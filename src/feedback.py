"""Feedback loop — turn raw interactions into evolving strategy.

Reads the last N interactions from :class:`Memory`, asks the LLM to extract
recurring patterns / lessons, and appends those insights to ``strategy.md``.
The strategy file is meant to be re-injected into future agent prompts so
the system actually learns from its own logs (no fine-tuning needed).

Default cadence: trigger once every 10 interactions. Override with
``trigger_every``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm import DEFAULT_MODEL, OllamaUnavailableError, generate
from src.memory import Memory

logger = logging.getLogger(__name__)


PATTERN_PROMPT = (
    "Eres un meta-analista del agente. Te paso las últimas interacciones "
    "(query → response → fuentes). Extrae 3-5 patrones recurrentes que "
    "sirvan para mejorar respuestas futuras: temas frecuentes, gaps de "
    "conocimiento, sesgos del usuario, qué funcionó, qué no.\n\n"
    "Formato de salida (estricto, una línea por insight):\n"
    "- INSIGHT: <observación accionable>\n\n"
    "No incluyas otra cosa."
)


@dataclass
class FeedbackResult:
    """What :meth:`FeedbackLoop.tick` returned this call."""

    triggered: bool
    insights: list[str]
    strategy_path: Path | None
    reason: str


class FeedbackLoop:
    """Periodically extract patterns from memory and update strategy.md.

    Args:
        memory: The :class:`Memory` instance backing the agent.
        lessons_path: File where the agent already drops ad-hoc lessons.
            Used as additional context when extracting patterns.
        strategy_path: Where consolidated strategy lines are appended.
            Defaults to ``memory.base_dir / "strategy.md"``.
        trigger_every: Run extraction once every N interactions. Set to 0
            to disable auto-trigger (still works via :meth:`force`).
        sample_size: How many recent interactions to feed the LLM.
        model: Ollama model tag for pattern extraction.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        lessons_path: Path | None = None,
        strategy_path: Path | None = None,
        trigger_every: int = 10,
        sample_size: int = 50,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.memory: Memory = memory
        self.lessons_path: Path = (
            lessons_path
            if lessons_path is not None
            else memory.base_dir / "lessons.md"
        )
        self.strategy_path: Path = (
            strategy_path
            if strategy_path is not None
            else memory.base_dir / "strategy.md"
        )
        self.trigger_every: int = max(0, trigger_every)
        self.sample_size: int = max(1, sample_size)
        self.model: str = model

        self._interactions_since_last: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> FeedbackResult:
        """Increment the counter and run extraction if threshold reached."""
        self._interactions_since_last += 1
        if self.trigger_every == 0:
            return FeedbackResult(False, [], None, "auto-trigger disabled")
        if self._interactions_since_last < self.trigger_every:
            return FeedbackResult(
                False,
                [],
                None,
                f"counter {self._interactions_since_last}/{self.trigger_every}",
            )
        return self.force()

    def force(self) -> FeedbackResult:
        """Run extraction immediately, regardless of counter state."""
        try:
            insights = self.extract_patterns()
        except OllamaUnavailableError as exc:
            logger.warning("feedback skipped — Ollama down: %s", exc)
            self._interactions_since_last = 0
            return FeedbackResult(False, [], None, f"ollama unavailable: {exc}")

        path = self.update_strategy(insights) if insights else None
        self._interactions_since_last = 0
        return FeedbackResult(
            triggered=True,
            insights=insights,
            strategy_path=path,
            reason=f"extracted {len(insights)} insights",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def extract_patterns(self) -> list[str]:
        """Ask the LLM for patterns over the last ``sample_size`` interactions.

        Returns a list of insight strings (one per ``- INSIGHT:`` line).
        """
        rows = self.memory.recent_interactions(n=self.sample_size)
        if not rows:
            return []

        rendered = "\n".join(self._render_row(r) for r in rows)
        lessons_blob = ""
        if self.lessons_path.exists():
            try:
                lessons_blob = self.lessons_path.read_text(encoding="utf-8")[-4000:]
            except OSError as exc:  # pragma: no cover — defensive
                logger.warning("could not read lessons.md: %s", exc)

        prompt = (
            f"# Interacciones recientes ({len(rows)})\n\n{rendered}\n\n"
            f"# Lessons.md previo (opcional)\n\n{lessons_blob or '_(vacío)_'}\n"
        )

        raw = generate(prompt=prompt, model=self.model, system=PATTERN_PROMPT)
        return self._parse_insights(raw)

    def update_strategy(self, insights: list[str]) -> Path:
        """Append a timestamped block of insights to ``strategy.md``."""
        self.strategy_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        block_lines = [f"\n## {ts}\n"]
        for ins in insights:
            block_lines.append(f"- {ins}")
        block_lines.append("")
        with self.strategy_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(block_lines))
        return self.strategy_path

    @staticmethod
    def _render_row(row: dict[str, Any]) -> str:
        """Compact one-line(ish) rendering of an interaction record."""
        ts = str(row.get("timestamp", ""))[:19]
        q = str(row.get("query", "")).replace("\n", " ")[:200]
        r = str(row.get("response", "")).replace("\n", " ")[:200]
        srcs = row.get("metadata", {}).get("sources", []) or []
        titles = ", ".join(s.get("title", "?") for s in srcs[:3])
        return f"- [{ts}] Q: {q!r} | R: {r!r} | sources: {titles}"

    @staticmethod
    def _parse_insights(raw: str) -> list[str]:
        """Pull every ``- INSIGHT: ...`` line out of an LLM response."""
        insights: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("- INSIGHT:"):
                insights.append(stripped[len("- INSIGHT:") :].strip())
            elif stripped.startswith("INSIGHT:"):
                insights.append(stripped[len("INSIGHT:") :].strip())
        return [i for i in insights if i]
