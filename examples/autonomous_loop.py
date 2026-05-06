"""Autonomous mode demo — scheduler + agent + feedback in one script.

Each cycle:

1. Loads today's daily note (creates it if missing).
2. Scans for unfinished TODOs (`- [ ] ...` lines).
3. For each TODO, asks the agent to suggest a next step using vault context.
4. Either appends the suggestion to the daily note (default) or just logs
   it (``--dry-run``).

Usage::

    python examples/autonomous_loop.py                 # 5-min cadence forever
    python examples/autonomous_loop.py --interval 1    # 1-min cadence
    python examples/autonomous_loop.py --dry-run --max-cycles 2

Env::

    OBSIDIAN_VAULT_PATH   Vault root (default: data/sample_vault)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

# Allow ``python examples/autonomous_loop.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import build_agent  # noqa: E402
from src.feedback import FeedbackLoop  # noqa: E402
from src.memory import Memory  # noqa: E402
from src.retrieval import VaultRetriever  # noqa: E402
from src.scheduler import Scheduler  # noqa: E402
from src.tools import AppendTool, TaskTool  # noqa: E402

logger = logging.getLogger("autonomous_loop")

TODO_RE = re.compile(r"^\s*-\s*\[\s\]\s+(.+?)\s*$", re.MULTILINE)


def _resolve_vault() -> Path:
    """Default to the bundled sample vault if no env var is set."""
    env = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "data" / "sample_vault").resolve()


def _ensure_daily_note(
    vault_root: Path,
    task_tool: TaskTool,
    *,
    dry_run: bool,
) -> Path:
    """Return today's daily note path; create it only when not in dry-run."""
    path = task_tool._today_path()  # noqa: SLF001 — internal but stable
    if not path.exists() and not dry_run:
        # TaskTool.run with a placeholder bootstraps the file with frontmatter.
        task_tool.run("Revisar daily note (auto-created)")
    return path


def _extract_todos(daily_path: Path) -> list[str]:
    """Return every unfinished TODO from the daily note, in order."""
    if not daily_path.exists():
        return []
    text = daily_path.read_text(encoding="utf-8")
    return [m.group(1).strip() for m in TODO_RE.finditer(text)]


def make_cycle(
    vault_path: Path,
    *,
    dry_run: bool,
) -> callable:
    """Build the cycle callable bound to a fresh agent + feedback loop."""
    memory = Memory()
    retriever = VaultRetriever(vault_path)
    feedback = FeedbackLoop(memory)
    agent = build_agent(
        vault_path=vault_path,
        memory=memory,
        retriever=retriever,
        write_enabled=not dry_run,
        feedback=feedback,
    )
    task_tool = TaskTool(vault_path)
    append_tool = AppendTool(vault_path)

    def cycle() -> dict[str, object]:
        daily_path = _ensure_daily_note(vault_path, task_tool, dry_run=dry_run)
        todos = _extract_todos(daily_path)
        logger.info("daily=%s todos=%d", daily_path.name, len(todos))

        actions: list[str] = []
        for todo in todos[:3]:  # cap so each cycle stays cheap
            query = f"¿Cuál sería un próximo paso concreto para: {todo}?"
            result = agent.invoke({"query": query})
            response = str(result.get("response", "")).strip()
            actions.append(f"- TODO «{todo}» → {response[:160]}…")

            if not dry_run and response:
                rel = daily_path.relative_to(vault_path).as_posix()
                append_tool.run(
                    rel,
                    f"### Próximo paso sugerido para: {todo}\n\n{response}\n",
                )

        return {
            "daily_note": str(daily_path),
            "todos_seen": len(todos),
            "actions_taken": actions,
            "dry_run": dry_run,
        }

    return cycle


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Minutes between cycles (default: 5)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N cycles (default: run forever)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to vault — just log proposed actions",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Override scheduler state.json location",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    vault = _resolve_vault()
    if not vault.exists():
        logger.error("vault does not exist: %s", vault)
        return 1

    logger.info(
        "starting autonomous loop vault=%s interval=%smin dry_run=%s max_cycles=%s",
        vault,
        args.interval,
        args.dry_run,
        args.max_cycles,
    )

    cycle = make_cycle(vault, dry_run=args.dry_run)
    scheduler = Scheduler(
        cycle,
        interval_minutes=args.interval,
        state_path=args.state_path,
        max_cycles=args.max_cycles,
        run_immediately=True,
    )

    try:
        asyncio.run(scheduler.start())
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
