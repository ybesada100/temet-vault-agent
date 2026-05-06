"""BM25 retrieval over an Obsidian vault.

No embeddings. No external services. Just rank-bm25 over tokenized note bodies,
indexed once at startup. Fast and predictable on vaults < 10k notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

# Frontmatter regex — leading `---\n...\n---\n`.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# H1 heading: first `# ...` line in body.
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
# Tag line in YAML-ish frontmatter: `tags: [a, b]` or `tags: a` or list form.
_TAGS_INLINE_RE = re.compile(r"^tags\s*:\s*\[(.+?)\]\s*$", re.MULTILINE | re.IGNORECASE)
_TAGS_LINE_RE = re.compile(r"^tags\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_TAGS_LIST_ITEM_RE = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)
# Word tokenizer: lowercase alphanumerics + accents.
_TOKEN_RE = re.compile(r"[\wáéíóúñü]+", re.UNICODE)


@dataclass
class Note:
    """A parsed Obsidian note ready for retrieval/display."""

    title: str
    path: str
    body: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word characters. Cheap and correct enough for BM25."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _extract_frontmatter_and_body(raw: str) -> tuple[str, str]:
    """Split raw markdown into (frontmatter, body)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return "", raw
    return match.group(1), raw[match.end() :]


def _parse_tags(frontmatter: str) -> list[str]:
    """Extract tags from a YAML-ish frontmatter string.

    Handles three common forms::

        tags: [a, b, c]
        tags: a
        tags:
          - a
          - b
    """
    if not frontmatter:
        return []

    inline = _TAGS_INLINE_RE.search(frontmatter)
    if inline:
        return [t.strip().strip("'\"") for t in inline.group(1).split(",") if t.strip()]

    line = _TAGS_LINE_RE.search(frontmatter)
    if line:
        value = line.group(1).strip()
        if value and not value.startswith("["):
            # Could be a single tag or the start of a list — peek next lines.
            after = frontmatter[line.end() :]
            list_items = _TAGS_LIST_ITEM_RE.findall(after)
            if list_items and not value:
                return [item.strip().strip("'\"") for item in list_items]
            if value:
                return [value.strip("'\"")]
    return []


def _parse_note(path: Path, vault_root: Path) -> Note:
    """Load a single .md file into a Note dataclass."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = _extract_frontmatter_and_body(raw)
    tags = _parse_tags(frontmatter)

    h1 = _H1_RE.search(body)
    title = h1.group(1).strip() if h1 else path.stem

    rel = path.relative_to(vault_root).as_posix()
    return Note(title=title, path=rel, body=body.strip(), tags=tags)


class VaultRetriever:
    """Indexes all markdown files under a vault and serves BM25 queries."""

    def __init__(self, vault_path: Path) -> None:
        self.vault_path: Path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault path does not exist: {self.vault_path}")
        if not self.vault_path.is_dir():
            raise NotADirectoryError(f"Vault path is not a directory: {self.vault_path}")

        self.notes: list[Note] = self._load_notes()
        self._tokens: list[list[str]] = [
            _tokenize(f"{n.title} {' '.join(n.tags)} {n.body}") for n in self.notes
        ]
        # rank-bm25 raises on empty corpus — keep ``_bm25`` optional.
        self._bm25: BM25Okapi | None = (
            BM25Okapi(self._tokens) if self._tokens else None
        )

    def _load_notes(self) -> list[Note]:
        """Walk the vault for .md files, skipping `.obsidian/` and dotfolders."""
        notes: list[Note] = []
        for path in self.vault_path.rglob("*.md"):
            # Skip Obsidian internals and any dotted directory.
            if any(part.startswith(".") for part in path.relative_to(self.vault_path).parts):
                continue
            try:
                notes.append(_parse_note(path, self.vault_path))
            except OSError:
                # Unreadable file — skip without aborting indexing.
                continue
        return notes

    def search(self, query: str, top_k: int = 5) -> list[Note]:
        """Return the top-k notes scored against ``query``.

        Returns an empty list if the vault is empty or the query has no tokens.
        """
        if not self._bm25 or not self.notes:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.notes, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results: list[Note] = []
        for note, score in ranked[:top_k]:
            # BM25 can yield strictly negative scores; clip those out.
            # Zero scores can still be informative (e.g. tiny corpora where
            # the IDF degenerates); keep them so the LLM at least sees notes.
            if score < 0:
                continue
            results.append(
                Note(
                    title=note.title,
                    path=note.path,
                    body=note.body,
                    tags=list(note.tags),
                    score=float(score),
                )
            )
        return results
