"""Asyncio scheduler for autonomous agent cycles.

Drives :func:`agent.invoke` (or any callable) on a fixed cadence, persists
state to ``~/.temet-vault/state.json``, and shuts down cleanly on SIGINT /
SIGTERM. Designed to be runnable both standalone (``python -m src.scheduler``)
and under systemd as a long-lived user service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for cycle callables. Sync or async, both supported.
CycleFn = Callable[[], Any] | Callable[[], Awaitable[Any]]


class Scheduler:
    """Run a callable on a fixed minute-based cadence with persistent state.

    The scheduler is async-first: signals are handled via the running event
    loop, sleeps yield to other tasks, and a sync ``cycle_fn`` is awaited
    in a thread to avoid blocking the loop.

    Args:
        cycle_fn: Callable invoked once per tick. Receives no arguments. May
            return any value (logged) or raise (counted, then suppressed).
        interval_minutes: Minutes between ticks. Must be > 0.
        state_path: Path for the JSON state file. Defaults to
            ``~/.temet-vault/state.json``.
        max_cycles: If provided, stop after this many ticks. Useful for
            tests and ``--max-cycles`` CLI flags.
        run_immediately: If ``True``, fire one cycle before the first sleep.
            Defaults to ``True`` for fast feedback in dev.
    """

    def __init__(
        self,
        cycle_fn: CycleFn,
        *,
        interval_minutes: float = 5.0,
        state_path: Path | None = None,
        max_cycles: int | None = None,
        run_immediately: bool = True,
    ) -> None:
        if interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")
        self._cycle_fn: CycleFn = cycle_fn
        self.interval_seconds: float = float(interval_minutes) * 60.0
        self.state_path: Path = (
            state_path
            if state_path is not None
            else Path.home() / ".temet-vault" / "state.json"
        )
        self.max_cycles: int | None = max_cycles
        self.run_immediately: bool = run_immediately

        self._stop_event: asyncio.Event | None = None
        self._state: dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        """Load persisted state from disk, or return a fresh skeleton."""
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("state.json corrupt (%s) — resetting", exc)
        return {
            "last_run": None,
            "run_count": 0,
            "errors": 0,
            "last_error": None,
            "started_at": None,
        }

    def _save_state(self) -> None:
        """Atomically persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    @property
    def state(self) -> dict[str, Any]:
        """Read-only view of current scheduler state."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Cycle execution
    # ------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Execute a single cycle, swallowing and recording any exception."""
        try:
            result = self._cycle_fn()
            if asyncio.iscoroutine(result):
                result = await result
            logger.info("cycle ok (run #%d)", self._state["run_count"] + 1)
            if result is not None:
                logger.debug("cycle result: %r", result)
        except Exception as exc:  # noqa: BLE001 — scheduler must survive any failure
            self._state["errors"] += 1
            self._state["last_error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("cycle failed: %s", exc)
        finally:
            self._state["run_count"] += 1
            self._state["last_run"] = datetime.now(timezone.utc).isoformat()
            self._save_state()

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wire SIGINT/SIGTERM to ``self._stop_event`` if the loop supports it."""
        assert self._stop_event is not None

        def _stop() -> None:
            logger.info("shutdown signal received")
            assert self._stop_event is not None
            self._stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except (NotImplementedError, RuntimeError):
                # Windows or non-main-thread loops can't install handlers;
                # fall back to default behaviour (KeyboardInterrupt).
                logger.debug("could not install signal handler for %s", sig)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the scheduler loop until stopped or ``max_cycles`` reached.

        Returns when:
        - SIGINT/SIGTERM is delivered
        - ``self.stop()`` is called
        - ``max_cycles`` ticks have completed

        Always persists state, even on early exit.
        """
        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._install_signal_handlers(loop)

        self._state["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        cycles_done = 0
        first = True

        while not self._stop_event.is_set():
            if first and not self.run_immediately:
                first = False
            else:
                await self._run_cycle()
                cycles_done += 1
                first = False
                if self.max_cycles is not None and cycles_done >= self.max_cycles:
                    logger.info("reached max_cycles=%d, stopping", self.max_cycles)
                    break

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
            except asyncio.TimeoutError:
                # Normal path — interval elapsed, time for the next cycle.
                continue

        logger.info("scheduler stopped (cycles=%d)", cycles_done)

    def stop(self) -> None:
        """Request graceful shutdown. Safe to call before ``start()``."""
        if self._stop_event is not None:
            self._stop_event.set()


def run_forever(
    cycle_fn: CycleFn,
    interval_minutes: float = 5.0,
    *,
    state_path: Path | None = None,
) -> None:
    """Convenience entry point — block the calling thread on a Scheduler.

    Equivalent to::

        asyncio.run(Scheduler(cycle_fn, interval_minutes=...).start())
    """
    scheduler = Scheduler(
        cycle_fn,
        interval_minutes=interval_minutes,
        state_path=state_path,
    )
    asyncio.run(scheduler.start())
