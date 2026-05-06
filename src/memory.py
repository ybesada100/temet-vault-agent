"""Persistent memory for vault agent.

Markdown for human-readable session summaries, JSONL for structured
interaction logs. Append-only, debuggable, zero deps beyond stdlib.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Memory:
    """Append-only memory store backed by JSONL + markdown files.

    Layout under ``base_dir``::

        base_dir/
        ├── interactions.jsonl     # append-only log of every Q/A
        └── sessions/
            └── YYYY-MM-DD.md      # daily session summaries
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = (
            base_dir if base_dir is not None else Path.home() / ".temet-vault" / "memory"
        )
        self.sessions_dir: Path = self.base_dir / "sessions"
        self.interactions_path: Path = self.base_dir / "interactions.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create base + sessions directories if missing."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.interactions_path.exists():
            self.interactions_path.touch()

    def log_interaction(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a single interaction to the JSONL log.

        Args:
            query: User-supplied prompt.
            response: Assistant's reply.
            metadata: Free-form dict (sources, latency, model, etc.).
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {},
        }
        with self.interactions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent_interactions(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last ``n`` interactions, oldest-first.

        Robust to malformed lines — skips them silently.
        """
        if not self.interactions_path.exists():
            return []
        lines = self.interactions_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate corrupt lines rather than crashing the agent.
                continue
        return records

    def save_session_summary(self, summary: str) -> Path:
        """Append a markdown summary to today's session file.

        Returns:
            The path of the file written.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.sessions_dir / f"{today}.md"
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        block = f"\n## {timestamp}\n\n{summary.strip()}\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(block)
        return path
