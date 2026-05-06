"""Smoke tests for memory, retrieval, and the LangGraph agent.

Ollama calls are monkey-patched so the suite runs offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import agent as agent_mod
from src.agent import build_agent
from src.memory import Memory
from src.retrieval import VaultRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_memory(tmp_path: Path) -> Memory:
    """Memory rooted at a tmp dir to avoid touching the real ~/.temet-vault."""
    return Memory(base_dir=tmp_path / "memory")


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Build a tiny throwaway vault with three well-separated notes.

    BM25 needs a non-degenerate IDF, so we keep at least three documents.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Cuban-Economy.md").write_text(
        "---\ntags: [cuba, economy]\n---\n"
        "# Cuban Economy\n\n"
        "Las remesas y la dolarización informal sostienen la economía cubana. "
        "Las MIPYMES crecen rápido pero el estado las regula. "
        "El peso cubano perdió referencia y la gente piensa en dólares.\n",
        encoding="utf-8",
    )
    (vault / "LLM-Inference.md").write_text(
        "---\ntags: [llm, qwen]\n---\n"
        "# LLM Inference\n\n"
        "Qwen 30B-A3B en Ollama corre a 40 tokens por segundo en RTX 5070 con "
        "offload parcial. Modelo MoE de Alibaba.\n",
        encoding="utf-8",
    )
    (vault / "Obsidian-tips.md").write_text(
        "---\ntags: [obsidian, workflow]\n---\n"
        "# Obsidian tips\n\n"
        "Daily notes con plantillas Templater y queries Dataview son el "
        "núcleo del workflow.\n",
        encoding="utf-8",
    )
    # Obsidian internals must be skipped.
    obs = vault / ".obsidian"
    obs.mkdir()
    (obs / "config.json").write_text("{}", encoding="utf-8")
    return vault


@pytest.fixture()
def sample_vault_path() -> Path:
    """Path to the bundled sample vault."""
    return Path(__file__).resolve().parent.parent / "data" / "sample_vault"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_memory_persistence(tmp_memory: Memory) -> None:
    """Logged interactions must round-trip through JSONL."""
    tmp_memory.log_interaction(
        query="¿qué es Qwen?",
        response="Un modelo MoE de Alibaba.",
        metadata={"sources": []},
    )
    tmp_memory.log_interaction(
        query="¿y Llama?",
        response="Modelo dense de Meta.",
        metadata={"sources": []},
    )

    rows = tmp_memory.recent_interactions(n=10)
    assert len(rows) == 2
    assert rows[0]["query"] == "¿qué es Qwen?"
    assert rows[1]["response"] == "Modelo dense de Meta."
    assert all("timestamp" in r for r in rows)

    # File contains exactly two valid JSON lines.
    raw = tmp_memory.interactions_path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    for line in raw:
        json.loads(line)  # must parse


def test_memory_session_summary(tmp_memory: Memory) -> None:
    """save_session_summary writes a markdown file under sessions/."""
    path = tmp_memory.save_session_summary("Hoy hablamos del vault.")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Hoy hablamos del vault." in text


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_finds_relevant_note(tmp_vault: Path) -> None:
    """A query about Cuban remesas must rank the Cuban-Economy note first."""
    retriever = VaultRetriever(tmp_vault)
    assert len(retriever.notes) == 3  # .obsidian was skipped

    results = retriever.search("remesas cubanas mipymes", top_k=2)
    assert results, "expected at least one match"
    top = results[0]
    assert top.title == "Cuban Economy"
    assert "cuba" in top.tags
    assert top.score > 0


def test_retrieval_empty_query(tmp_vault: Path) -> None:
    """Empty / whitespace query returns no results gracefully."""
    retriever = VaultRetriever(tmp_vault)
    assert retriever.search("", top_k=3) == []
    assert retriever.search("   ", top_k=3) == []


def test_retrieval_missing_vault(tmp_path: Path) -> None:
    """Bad path raises a clear error."""
    with pytest.raises(FileNotFoundError):
        VaultRetriever(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# Agent (end-to-end with mocked Ollama)
# ---------------------------------------------------------------------------


def test_agent_full_pipeline(
    tmp_vault: Path,
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full graph runs without hitting Ollama; memory captures the interaction."""
    captured: dict[str, str] = {}

    def fake_generate(prompt: str, **kwargs: object) -> str:
        captured["prompt"] = prompt
        return "Respuesta de prueba citando [[Cuban Economy]]."

    monkeypatch.setattr(agent_mod, "generate", fake_generate)

    agent = build_agent(vault_path=tmp_vault, memory=tmp_memory)
    result = agent.invoke({"query": "¿qué dije sobre remesas?"})

    assert "Respuesta de prueba" in result["response"]
    assert "Cuban Economy" in result["response"]
    assert result["retrieved_notes"], "retrieval should have produced notes"
    assert result["intent"] in {"question", "search", "follow_up"}

    rows = tmp_memory.recent_interactions(n=5)
    assert len(rows) == 1
    assert rows[0]["query"] == "¿qué dije sobre remesas?"
    sources = rows[0]["metadata"]["sources"]
    assert any(s["title"] == "Cuban Economy" for s in sources)

    # System + context made it into the prompt sent to the LLM.
    assert "Cuban Economy" in captured["prompt"]
    assert "remesas" in captured["prompt"].lower()


def test_agent_handles_ollama_down(
    tmp_vault: Path,
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Ollama raises, the graph still completes with a friendly error."""
    from src.llm import OllamaUnavailableError

    def fail_generate(prompt: str, **kwargs: object) -> str:
        raise OllamaUnavailableError("no daemon")

    monkeypatch.setattr(agent_mod, "generate", fail_generate)

    agent = build_agent(vault_path=tmp_vault, memory=tmp_memory)
    result = agent.invoke({"query": "ping"})
    assert "Ollama" in result["response"]
    # Memory still captured the failed turn.
    assert len(tmp_memory.recent_interactions(n=5)) == 1
