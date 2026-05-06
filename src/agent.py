"""LangGraph state machine for the vault agent.

Pipeline::

    parse_query → retrieve → build_context → generate → log → END
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.llm import DEFAULT_MODEL, OllamaUnavailableError, generate
from src.memory import Memory
from src.retrieval import Note, VaultRetriever

QueryIntent = Literal["search", "question", "follow_up"]

SYSTEM_PROMPT = (
    "Eres un asistente que responde basándote en notas del vault Obsidian del "
    "usuario. Cita la nota fuente con [[título]]. Si no tienes contexto "
    "suficiente, dilo honestamente."
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


def build_agent(
    vault_path: Path,
    memory: Memory,
    *,
    retriever: VaultRetriever | None = None,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
) -> CompiledStateGraph:
    """Compile the LangGraph state machine.

    Args:
        vault_path: Root of the Obsidian vault.
        memory: :class:`Memory` instance for persistent logging.
        retriever: Optional pre-built retriever (useful for tests).
        model: Ollama model tag.
        top_k: Number of notes to retrieve per query.

    Returns:
        A compiled LangGraph that consumes/produces :class:`AgentState`.
    """
    vault_retriever = retriever or VaultRetriever(vault_path)

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
            },
        )
        return {}

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate", generate_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "log")
    graph.add_edge("log", END)

    return graph.compile()
