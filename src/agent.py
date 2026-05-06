"""LangGraph state machine for the vault agent.

Pipeline (v0.2 — autonomous extensions enabled when ``write_enabled=True``)::

    parse_query
        → retrieve
        → build_context
        → generate
        → should_write_note  ─┬─ write_note ─┐
                              └─ skip_write ─┤
                                              ├─ log
                                              └─ update_feedback → END

Backward-compat: when ``write_enabled=False`` (default), the conditional
always routes to ``skip_write`` and ``feedback`` is a no-op — so existing
v0.1 callers and tests behave identically.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.feedback import FeedbackLoop
from src.llm import DEFAULT_MODEL, OllamaUnavailableError, generate
from src.memory import Memory
from src.retrieval import Note, VaultRetriever
from src.tools import WriteTool

logger = logging.getLogger(__name__)

QueryIntent = Literal["search", "question", "follow_up"]

SYSTEM_PROMPT = (
    "Eres un asistente que responde basándote en notas del vault Obsidian del "
    "usuario. Cita la nota fuente con [[título]]. Si no tienes contexto "
    "suficiente, dilo honestamente."
)

WRITE_DECISION_PROMPT = (
    "Eres el meta-controller de un agente. Te paso la pregunta del usuario "
    "y la respuesta que el agente acaba de generar. Decide si vale la pena "
    "guardar la respuesta como nueva nota en el vault Obsidian.\n\n"
    "Responde con UNA sola línea, exactamente uno de estos formatos:\n"
    "  WRITE: <título-corto-de-la-nota>\n"
    "  SKIP\n\n"
    "Reglas:\n"
    "- WRITE solo si la respuesta contiene insight original, síntesis nueva, "
    "  o un plan de acción que valga la pena conservar.\n"
    "- SKIP si la respuesta solo recita notas existentes, es trivial, o no "
    "  agrega información nueva."
)


class AgentState(TypedDict, total=False):
    """State carried through the graph."""

    query: str
    intent: QueryIntent
    retrieved_notes: list[Note]
    context: str
    response: str
    needs_clarification: bool
    model: str
    # v0.2 additions
    write_decision: Literal["write", "skip"]
    write_title: str | None
    written_path: str | None
    feedback_triggered: bool


def _detect_intent(query: str) -> QueryIntent:
    """Heuristic intent classifier — cheap, deterministic, no LLM."""
    q = query.strip().lower()
    if not q:
        return "question"
    follow_up_markers = (
        "y entonces",
        "y eso",
        "ampl",  # ampliar / amplía
        "explica más",
        "más detalle",
        "follow up",
        "follow-up",
        "and then",
    )
    if any(marker in q for marker in follow_up_markers):
        return "follow_up"
    if q.startswith(("buscar", "search", "find", "encuentra")):
        return "search"
    return "question"


def _format_context(notes: list[Note]) -> str:
    """Render retrieved notes as a markdown context block for the LLM."""
    if not notes:
        return "_(sin notas relevantes encontradas en el vault)_"
    blocks: list[str] = []
    for note in notes:
        tags = f" — tags: {', '.join(note.tags)}" if note.tags else ""
        body_excerpt = note.body if len(note.body) <= 1500 else note.body[:1500] + "…"
        blocks.append(
            f"### [[{note.title}]]{tags}\n"
            f"_path: `{note.path}` · score: {note.score:.2f}_\n\n"
            f"{body_excerpt}"
        )
    return "\n\n---\n\n".join(blocks)


_WRITE_LINE_RE = re.compile(r"^\s*WRITE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_write_decision(raw: str) -> tuple[str, str | None]:
    """Parse the meta-controller's response into (decision, title)."""
    if not raw:
        return ("skip", None)
    match = _WRITE_LINE_RE.search(raw)
    if match:
        title = match.group(1).strip().strip("\"'")
        if title:
            return ("write", title)
    return ("skip", None)


def build_agent(
    vault_path: Path,
    memory: Memory,
    *,
    retriever: VaultRetriever | None = None,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    write_enabled: bool = False,
    write_subdir: str = "agent-notes",
    feedback: FeedbackLoop | None = None,
) -> CompiledStateGraph:
    """Compile the LangGraph state machine.

    Args:
        vault_path: Root of the Obsidian vault.
        memory: :class:`Memory` instance for persistent logging.
        retriever: Optional pre-built retriever (useful for tests).
        model: Ollama model tag.
        top_k: Number of notes to retrieve per query.
        write_enabled: If ``True``, the agent may write a new note after
            answering. Defaults to ``False`` for backward-compat with v0.1.
        write_subdir: Subdirectory under the vault where agent-authored
            notes are saved. Default ``"agent-notes"``.
        feedback: Optional :class:`FeedbackLoop`. When provided, it is
            ticked after every interaction.

    Returns:
        A compiled LangGraph that consumes/produces :class:`AgentState`.
    """
    vault_retriever = retriever or VaultRetriever(vault_path)
    write_tool = WriteTool(vault_path) if write_enabled else None

    def parse_query(state: AgentState) -> AgentState:
        intent = _detect_intent(state.get("query", ""))
        return {"intent": intent, "needs_clarification": False}

    def retrieve(state: AgentState) -> AgentState:
        notes = vault_retriever.search(state["query"], top_k=top_k)
        return {"retrieved_notes": notes}

    def build_context(state: AgentState) -> AgentState:
        return {"context": _format_context(state.get("retrieved_notes", []))}

    def generate_node(state: AgentState) -> AgentState:
        prompt = (
            f"# Contexto del vault\n\n{state.get('context', '')}\n\n"
            f"# Pregunta del usuario\n\n{state['query']}\n\n"
            f"# Tu respuesta\n"
            f"Responde en español, conciso pero sustantivo. "
            f"Cita las notas usadas como [[título]]."
        )
        try:
            response = generate(
                prompt=prompt,
                model=state.get("model", model),
                system=SYSTEM_PROMPT,
            )
        except OllamaUnavailableError as exc:
            response = (
                "_(Ollama no está disponible — no pude generar respuesta.)_\n\n"
                f"Detalle: {exc}"
            )
        return {"response": response}

    def should_write_note(state: AgentState) -> AgentState:
        """Ask the LLM whether the answer is worth persisting."""
        if not write_enabled:
            return {"write_decision": "skip", "write_title": None}

        decision_prompt = (
            f"# Pregunta\n{state['query']}\n\n"
            f"# Respuesta generada\n{state.get('response', '')}\n"
        )
        try:
            raw = generate(
                prompt=decision_prompt,
                model=state.get("model", model),
                system=WRITE_DECISION_PROMPT,
                temperature=0.1,
            )
        except OllamaUnavailableError as exc:
            logger.warning("write-decision skipped — Ollama down: %s", exc)
            return {"write_decision": "skip", "write_title": None}

        decision, title = _parse_write_decision(raw)
        return {"write_decision": decision, "write_title": title}

    def write_note(state: AgentState) -> AgentState:
        """Persist the agent's response as a new vault note."""
        assert write_tool is not None  # guaranteed by routing
        title = state.get("write_title") or "agent-insight"
        body = (
            f"_Generado por temet-vault-agent en respuesta a:_\n\n"
            f"> {state['query']}\n\n"
            f"---\n\n"
            f"{state.get('response', '')}\n"
        )
        result = write_tool.run(
            title=title,
            body=body,
            tags=["agent", "auto"],
            subdir=write_subdir,
        )
        if not result.ok:
            logger.warning("write_note failed: %s", result.message)
            return {"written_path": None}
        return {"written_path": result.path}

    def skip_write(state: AgentState) -> AgentState:
        return {"written_path": None}

    def log_node(state: AgentState) -> AgentState:
        sources = [
            {"title": n.title, "path": n.path, "score": n.score}
            for n in state.get("retrieved_notes", [])
        ]
        memory.log_interaction(
            query=state["query"],
            response=state.get("response", ""),
            metadata={
                "intent": state.get("intent", "question"),
                "sources": sources,
                "model": state.get("model", model),
                "written_path": state.get("written_path"),
            },
        )
        return {}

    def update_feedback_node(state: AgentState) -> AgentState:
        if feedback is None:
            return {"feedback_triggered": False}
        result = feedback.tick()
        return {"feedback_triggered": result.triggered}

    def _route_write(state: AgentState) -> str:
        return "write_note" if state.get("write_decision") == "write" else "skip_write"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate", generate_node)
    graph.add_node("should_write_note", should_write_note)
    graph.add_node("write_note", write_note)
    graph.add_node("skip_write", skip_write)
    graph.add_node("log", log_node)
    graph.add_node("update_feedback", update_feedback_node)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "should_write_note")
    graph.add_conditional_edges(
        "should_write_note",
        _route_write,
        {"write_note": "write_note", "skip_write": "skip_write"},
    )
    graph.add_edge("write_note", "log")
    graph.add_edge("skip_write", "log")
    graph.add_edge("log", "update_feedback")
    graph.add_edge("update_feedback", END)

    return graph.compile()
