"""Vault-write tools exposed to the agent.

Each tool is a thin wrapper around filesystem ops on the Obsidian vault.
They are also exported as LangChain :class:`StructuredTool` instances so
the agent can invoke them through the standard tool-calling interface.

Layout philosophy:
- Tool *classes* hold filesystem state (vault root, daily-note format).
- Tool *functions* are pure side-effects with structured args.
- StructuredTool wrappers come from :func:`as_langchain_tools`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class VaultPathError(ValueError):
    """Raised when a tool tries to touch a path outside the vault root."""


def _safe_join(vault_root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``vault_root`` and reject path-escapes.

    Always returns a path strictly inside ``vault_root``. Trailing ``.md``
    is added when missing so callers can pass either ``"Note"`` or
    ``"Note.md"``.
    """
    if not relative or relative.strip() in {"", ".", ".."}:
        raise VaultPathError(f"invalid relative path: {relative!r}")
    rel = relative.strip()
    if not rel.endswith(".md"):
        rel = f"{rel}.md"
    candidate = (vault_root / rel).resolve()
    root = vault_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise VaultPathError(
            f"path {relative!r} resolves outside vault {root}"
        ) from exc
    return candidate


def _slugify(title: str) -> str:
    """Filesystem-safe slug for note titles."""
    cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", "-", cleaned)
    return cleaned or "untitled"


def _frontmatter_block(meta: dict[str, object]) -> str:
    """Render a minimal YAML frontmatter block. No external YAML lib."""
    if not meta:
        return ""
    lines: list[str] = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
            lines.append(f"{key}: [{rendered}]")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Uniform return type for every tool — success flag + path + message."""

    ok: bool
    path: str
    message: str

    def __str__(self) -> str:  # pragma: no cover — trivial
        prefix = "✓" if self.ok else "✗"
        return f"{prefix} {self.path}: {self.message}"


class WriteTool:
    """Create a new markdown note with Obsidian-style frontmatter.

    Refuses to overwrite existing files unless ``overwrite=True`` is passed
    explicitly to :meth:`run` — this protects user notes from accidental
    clobbering by the agent.
    """

    name = "write_note"
    description = (
        "Create a new note in the Obsidian vault. Args: title (str), "
        "body (str, markdown), tags (list[str], optional), "
        "subdir (str, optional). Returns ToolResult."
    )

    def __init__(self, vault_root: Path) -> None:
        self.vault_root: Path = Path(vault_root).expanduser().resolve()

    def run(
        self,
        title: str,
        body: str,
        *,
        tags: list[str] | None = None,
        subdir: str | None = None,
        overwrite: bool = False,
    ) -> ToolResult:
        """Write a note. Returns a :class:`ToolResult`."""
        slug = _slugify(title)
        rel = f"{subdir.strip('/')}/{slug}" if subdir else slug
        path = _safe_join(self.vault_root, rel)

        if path.exists() and not overwrite:
            return ToolResult(False, str(path), "already exists (overwrite=False)")

        meta: dict[str, object] = {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "agent": "temet-vault-agent",
        }
        if tags:
            meta["tags"] = list(tags)

        content = _frontmatter_block(meta) + f"# {title}\n\n{body.rstrip()}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, str(path), f"wrote {len(content)} bytes")


class AppendTool:
    """Append a markdown block to an existing note.

    The note must already exist; use :class:`WriteTool` first if not. The
    appended block is separated by a blank line so it stays readable.
    """

    name = "append_to_note"
    description = (
        "Append a markdown block to an existing note. Args: note (str, "
        "filename or relative path), content (str). Returns ToolResult."
    )

    def __init__(self, vault_root: Path) -> None:
        self.vault_root: Path = Path(vault_root).expanduser().resolve()

    def run(self, note: str, content: str) -> ToolResult:
        """Append ``content`` to ``note``. Fails if the note does not exist."""
        path = _safe_join(self.vault_root, note)
        if not path.exists():
            return ToolResult(False, str(path), "note does not exist")
        block = "\n\n" + content.rstrip() + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(block)
        return ToolResult(True, str(path), f"appended {len(block)} bytes")


class LinkTool:
    """Create an Obsidian-style backlink between two notes.

    Appends a ``- [[target]]`` line under a ``## Backlinks`` heading in
    ``source``. Idempotent — refuses to add a link that already exists.
    """

    name = "link_notes"
    description = (
        "Add an Obsidian [[wikilink]] from source note to target note "
        "under a Backlinks section. Args: source (str), target (str). "
        "Returns ToolResult."
    )

    BACKLINKS_HEADER = "## Backlinks"

    def __init__(self, vault_root: Path) -> None:
        self.vault_root: Path = Path(vault_root).expanduser().resolve()

    def run(self, source: str, target: str) -> ToolResult:
        """Insert ``[[target]]`` into ``source``'s Backlinks section."""
        src_path = _safe_join(self.vault_root, source)
        if not src_path.exists():
            return ToolResult(False, str(src_path), "source note does not exist")

        target_title = Path(target).stem
        link_line = f"- [[{target_title}]]"

        text = src_path.read_text(encoding="utf-8")
        if link_line in text:
            return ToolResult(True, str(src_path), "link already present (idempotent)")

        if self.BACKLINKS_HEADER in text:
            text = text.rstrip() + f"\n{link_line}\n"
        else:
            text = text.rstrip() + f"\n\n{self.BACKLINKS_HEADER}\n\n{link_line}\n"

        src_path.write_text(text, encoding="utf-8")
        return ToolResult(True, str(src_path), f"linked → [[{target_title}]]")


class TaskTool:
    """Add a TODO checkbox to today's daily note.

    Daily notes live at ``daily/YYYY-MM-DD.md`` by default. Creates the file
    (with a frontmatter header) if it doesn't exist yet.
    """

    name = "add_task"
    description = (
        "Add a TODO checkbox to today's daily note. Args: task (str), "
        "tags (list[str], optional). Returns ToolResult."
    )

    def __init__(
        self,
        vault_root: Path,
        *,
        daily_dir: str = "daily",
        date_format: str = "%Y-%m-%d",
    ) -> None:
        self.vault_root: Path = Path(vault_root).expanduser().resolve()
        self.daily_dir: str = daily_dir
        self.date_format: str = date_format

    def _today_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime(self.date_format)
        return _safe_join(self.vault_root, f"{self.daily_dir}/{today}")

    def run(self, task: str, *, tags: list[str] | None = None) -> ToolResult:
        """Append ``- [ ] task`` to today's daily note (creating it if needed)."""
        path = self._today_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        tag_suffix = " " + " ".join(f"#{t.lstrip('#')}" for t in tags) if tags else ""
        line = f"- [ ] {task.strip()}{tag_suffix}\n"

        if not path.exists():
            today_str = datetime.now(timezone.utc).strftime(self.date_format)
            header = _frontmatter_block(
                {
                    "date": today_str,
                    "agent": "temet-vault-agent",
                    "tags": ["daily"],
                }
            )
            path.write_text(
                header + f"# {today_str}\n\n## Tasks\n\n{line}",
                encoding="utf-8",
            )
            return ToolResult(True, str(path), "created daily note + task")

        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return ToolResult(True, str(path), "appended task")


# ---------------------------------------------------------------------------
# LangChain integration
# ---------------------------------------------------------------------------


class WriteNoteArgs(BaseModel):
    """Arguments for :class:`WriteTool`."""

    title: str = Field(..., description="Note title (becomes the H1)")
    body: str = Field(..., description="Markdown body of the note")
    tags: list[str] | None = Field(default=None, description="Optional tag list")
    subdir: str | None = Field(default=None, description="Optional vault subdirectory")

    @field_validator("title")
    @classmethod
    def _non_empty_title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v


class AppendNoteArgs(BaseModel):
    """Arguments for :class:`AppendTool`."""

    note: str = Field(..., description="Existing note path/filename inside the vault")
    content: str = Field(..., description="Markdown content to append")


class LinkNotesArgs(BaseModel):
    """Arguments for :class:`LinkTool`."""

    source: str = Field(..., description="Note that should contain the new link")
    target: str = Field(..., description="Note to link to (used as [[target]])")


class AddTaskArgs(BaseModel):
    """Arguments for :class:`TaskTool`."""

    task: str = Field(..., description="What needs to be done")
    tags: list[str] | None = Field(default=None, description="Optional Obsidian tags")


def as_langchain_tools(
    vault_root: Path,
    *,
    include: Iterable[str] | None = None,
) -> list[StructuredTool]:
    """Build LangChain ``StructuredTool`` instances bound to ``vault_root``.

    Args:
        vault_root: Root of the Obsidian vault.
        include: Optional iterable of tool names to include. If ``None``,
            all four tools are returned.

    Returns:
        A list of compiled :class:`StructuredTool` objects ready to attach
        to a LangGraph agent.
    """
    write = WriteTool(vault_root)
    append = AppendTool(vault_root)
    link = LinkTool(vault_root)
    task = TaskTool(vault_root)

    tools: dict[str, StructuredTool] = {
        write.name: StructuredTool.from_function(
            func=write.run,
            name=write.name,
            description=write.description,
            args_schema=WriteNoteArgs,
        ),
        append.name: StructuredTool.from_function(
            func=append.run,
            name=append.name,
            description=append.description,
            args_schema=AppendNoteArgs,
        ),
        link.name: StructuredTool.from_function(
            func=link.run,
            name=link.name,
            description=link.description,
            args_schema=LinkNotesArgs,
        ),
        task.name: StructuredTool.from_function(
            func=task.run,
            name=task.name,
            description=task.description,
            args_schema=AddTaskArgs,
        ),
    }

    if include is None:
        return list(tools.values())
    return [tools[name] for name in include if name in tools]


__all__ = [
    "AppendTool",
    "LinkTool",
    "TaskTool",
    "ToolResult",
    "VaultPathError",
    "WriteTool",
    "as_langchain_tools",
]
