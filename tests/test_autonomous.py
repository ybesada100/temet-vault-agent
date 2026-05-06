"""Tests for v0.2 autonomous mode: scheduler, tools, feedback loop.

All Ollama calls are monkey-patched so the suite runs offline. Filesystem
side-effects are confined to ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src import agent as agent_mod
from src import feedback as feedback_mod
from src.agent import build_agent
from src.feedback import FeedbackLoop
from src.memory import Memory
from src.scheduler import Scheduler
from src.tools import (
    AppendTool,
    LinkTool,
    TaskTool,
    VaultPathError,
    WriteTool,
    as_langchain_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Empty vault root for write-side tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture()
def tmp_memory(tmp_path: Path) -> Memory:
    return Memory(base_dir=tmp_path / "memory")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_runs_periodically(tmp_path: Path) -> None:
    """Scheduler executes the cycle N times and persists state.json."""
    counter: dict[str, int] = {"n": 0}

    def cycle() -> None:
        counter["n"] += 1

    state_path = tmp_path / "state.json"
    # 0.001 min = 60ms — keeps the test sub-second.
    scheduler = Scheduler(
        cycle,
        interval_minutes=0.001,
        state_path=state_path,
        max_cycles=3,
        run_immediately=True,
    )

    asyncio.run(scheduler.start())

    assert counter["n"] == 3
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["run_count"] == 3
    assert state["errors"] == 0
    assert state["last_run"] is not None
    assert state["started_at"] is not None


def test_scheduler_records_errors(tmp_path: Path) -> None:
    """A raising cycle is logged as an error but doesn't crash the loop."""

    def boom() -> None:
        raise RuntimeError("nope")

    scheduler = Scheduler(
        boom,
        interval_minutes=0.001,
        state_path=tmp_path / "state.json",
        max_cycles=2,
    )
    asyncio.run(scheduler.start())

    state = scheduler.state
    assert state["run_count"] == 2
    assert state["errors"] == 2
    assert state["last_error"] is not None
    assert "nope" in state["last_error"]


def test_scheduler_supports_async_cycle(tmp_path: Path) -> None:
    """An async cycle function is awaited correctly."""
    seen: list[int] = []

    async def cycle() -> None:
        await asyncio.sleep(0)
        seen.append(1)

    scheduler = Scheduler(
        cycle,
        interval_minutes=0.001,
        state_path=tmp_path / "state.json",
        max_cycles=2,
    )
    asyncio.run(scheduler.start())
    assert seen == [1, 1]


def test_scheduler_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        Scheduler(lambda: None, interval_minutes=0)


# ---------------------------------------------------------------------------
# WriteTool / AppendTool / LinkTool / TaskTool
# ---------------------------------------------------------------------------


def test_write_tool_creates_valid_markdown(tmp_vault: Path) -> None:
    """WriteTool produces a note with frontmatter, title, and body."""
    tool = WriteTool(tmp_vault)
    result = tool.run(
        title="My First Note",
        body="Hello **world**.",
        tags=["demo", "test"],
        subdir="agent-notes",
    )
    assert result.ok
    path = Path(result.path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "tags: [demo, test]" in text
    assert "agent: temet-vault-agent" in text
    assert "# My First Note" in text
    assert "Hello **world**." in text
    # Path lives under the agent-notes subdir of the vault.
    assert path.parent.name == "agent-notes"


def test_write_tool_refuses_overwrite(tmp_vault: Path) -> None:
    tool = WriteTool(tmp_vault)
    tool.run(title="Same", body="v1")
    second = tool.run(title="Same", body="v2")
    assert not second.ok
    assert "exists" in second.message
    # Overwrite=True should succeed.
    third = tool.run(title="Same", body="v3", overwrite=True)
    assert third.ok


def test_write_tool_blocks_path_escape(tmp_vault: Path) -> None:
    tool = WriteTool(tmp_vault)
    with pytest.raises(VaultPathError):
        tool.run(title="bad", body="x", subdir="../../etc")


def test_append_tool_appends_block(tmp_vault: Path) -> None:
    write = WriteTool(tmp_vault)
    append = AppendTool(tmp_vault)
    write.run(title="Journal", body="First entry.")
    res = append.run(note="Journal", content="Second entry.")
    assert res.ok
    text = (tmp_vault / "Journal.md").read_text(encoding="utf-8")
    assert "First entry." in text
    assert "Second entry." in text


def test_append_tool_missing_note(tmp_vault: Path) -> None:
    res = AppendTool(tmp_vault).run(note="ghost", content="hi")
    assert not res.ok


def test_link_tool_idempotent(tmp_vault: Path) -> None:
    write = WriteTool(tmp_vault)
    write.run(title="Source", body="body")
    write.run(title="Target", body="body")
    link = LinkTool(tmp_vault)
    res1 = link.run(source="Source", target="Target")
    res2 = link.run(source="Source", target="Target")
    assert res1.ok and res2.ok
    text = (tmp_vault / "Source.md").read_text(encoding="utf-8")
    assert text.count("[[Target]]") == 1
    assert "## Backlinks" in text


def test_task_tool_creates_daily_note(tmp_vault: Path) -> None:
    task = TaskTool(tmp_vault)
    res = task.run(task="Ship v0.2", tags=["project"])
    assert res.ok
    path = Path(res.path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "- [ ] Ship v0.2 #project" in text
    assert "## Tasks" in text


def test_as_langchain_tools_returns_structured_tools(tmp_vault: Path) -> None:
    tools = as_langchain_tools(tmp_vault)
    names = {t.name for t in tools}
    assert names == {"write_note", "append_to_note", "link_notes", "add_task"}
    # StructuredTool has a working .invoke surface.
    write = next(t for t in tools if t.name == "write_note")
    res = write.invoke({"title": "Lc Note", "body": "via langchain"})
    assert res.ok
    assert (tmp_vault / "Lc-Note.md").exists()


# ---------------------------------------------------------------------------
# FeedbackLoop
# ---------------------------------------------------------------------------


def _seed_interactions(memory: Memory, n: int) -> None:
    for i in range(n):
        memory.log_interaction(
            query=f"pregunta {i}",
            response=f"respuesta {i}",
            metadata={"sources": [{"title": f"Note{i}", "path": "x", "score": 1.0}]},
        )


def test_feedback_loop_extracts_patterns(
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force() calls the LLM, parses INSIGHT lines, appends to strategy.md."""
    canned = (
        "- INSIGHT: el usuario pregunta mucho sobre Cuba.\n"
        "- INSIGHT: faltan notas sobre macroeconomía.\n"
        "ruido extra que debe ignorarse\n"
        "- INSIGHT: respuestas más cortas funcionan mejor.\n"
    )

    def fake_generate(prompt: str, **_: object) -> str:
        return canned

    monkeypatch.setattr(feedback_mod, "generate", fake_generate)
    _seed_interactions(tmp_memory, 5)

    loop = FeedbackLoop(tmp_memory, trigger_every=10, sample_size=10)
    result = loop.force()

    assert result.triggered
    assert len(result.insights) == 3
    assert "Cuba" in result.insights[0]
    assert result.strategy_path is not None
    assert result.strategy_path.exists()
    text = result.strategy_path.read_text(encoding="utf-8")
    assert "Cuba" in text
    assert "macroeconomía" in text


def test_feedback_loop_auto_trigger(
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tick() returns triggered=True only after `trigger_every` calls."""
    monkeypatch.setattr(
        feedback_mod,
        "generate",
        lambda *a, **k: "- INSIGHT: ok\n",
    )
    _seed_interactions(tmp_memory, 3)

    loop = FeedbackLoop(tmp_memory, trigger_every=3)
    r1 = loop.tick()
    r2 = loop.tick()
    r3 = loop.tick()
    assert (r1.triggered, r2.triggered, r3.triggered) == (False, False, True)
    assert r3.insights == ["ok"]


def test_feedback_loop_handles_ollama_down(
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.llm import OllamaUnavailableError

    def boom(*_a: object, **_k: object) -> str:
        raise OllamaUnavailableError("offline")

    monkeypatch.setattr(feedback_mod, "generate", boom)
    _seed_interactions(tmp_memory, 1)
    result = FeedbackLoop(tmp_memory).force()
    assert not result.triggered
    assert "ollama" in result.reason.lower()


# ---------------------------------------------------------------------------
# Agent v0.2 — write decision + feedback wiring
# ---------------------------------------------------------------------------


def test_agent_writes_note_when_decision_says_so(
    tmp_vault: Path,
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When write_enabled and the meta-controller says WRITE, a note appears."""
    # Seed the vault with one note so retrieval has something to chew on.
    (tmp_vault / "Seed.md").write_text(
        "---\ntags: [seed]\n---\n# Seed\nbody\n", encoding="utf-8"
    )

    calls: list[str] = []

    def fake_generate(prompt: str, **kwargs: object) -> str:
        # Two invocations per turn: main answer + write decision.
        calls.append(prompt[:30])
        if "Pregunta\n" in prompt and "Respuesta generada" in prompt:
            return "WRITE: insight-test"
        return "Respuesta principal."

    monkeypatch.setattr(agent_mod, "generate", fake_generate)

    agent = build_agent(
        vault_path=tmp_vault,
        memory=tmp_memory,
        write_enabled=True,
    )
    result = agent.invoke({"query": "¿qué pasa?"})
    assert result["write_decision"] == "write"
    assert result["written_path"] is not None
    assert Path(result["written_path"]).exists()


def test_agent_skips_write_when_disabled(
    tmp_vault: Path,
    tmp_memory: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write_enabled=False keeps v0.1 behaviour — never writes."""
    (tmp_vault / "Seed.md").write_text(
        "---\ntags: [seed]\n---\n# Seed\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr(agent_mod, "generate", lambda *a, **k: "respuesta")

    agent = build_agent(
        vault_path=tmp_vault,
        memory=tmp_memory,
        write_enabled=False,
    )
    result = agent.invoke({"query": "ping"})
    assert result["write_decision"] == "skip"
    assert result["written_path"] is None
