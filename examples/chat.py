"""Conversational loop with in-memory turn history.

Each turn prepends the last few exchanges to the user query so the LLM has
short-term context. Persistent memory still goes through :class:`Memory`.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

# Allow `python examples/chat.py` from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Prompt  # noqa: E402

from src.agent import build_agent  # noqa: E402
from src.memory import Memory  # noqa: E402
from src.retrieval import VaultRetriever  # noqa: E402

HISTORY_TURNS = 4


def _format_history(history: deque[tuple[str, str]]) -> str:
    """Render the rolling history as a markdown preamble."""
    if not history:
        return ""
    lines = ["## Conversación previa"]
    for q, a in history:
        excerpt = a if len(a) <= 400 else a[:400] + "…"
        lines.append(f"- **Usuario:** {q}\n  **Asistente:** {excerpt}")
    return "\n".join(lines) + "\n\n"


def main() -> int:
    """Run an interactive chat session."""
    console = Console()
    vault_path = Path(
        os.environ.get(
            "OBSIDIAN_VAULT_PATH",
            str(Path(__file__).resolve().parent.parent / "data" / "sample_vault"),
        )
    ).expanduser()

    if not vault_path.exists():
        console.print(f"[red]Vault no encontrado:[/red] {vault_path}")
        return 1

    retriever = VaultRetriever(vault_path)
    memory = Memory()
    agent = build_agent(vault_path=vault_path, memory=memory, retriever=retriever)

    console.print(
        Panel.fit(
            f"[bold cyan]Chat con vault[/bold cyan]\n"
            f"Vault: [yellow]{vault_path}[/yellow] ({len(retriever.notes)} notas)\n"
            f"Escribe [yellow]/exit[/yellow] para salir.",
            border_style="cyan",
        )
    )

    history: deque[tuple[str, str]] = deque(maxlen=HISTORY_TURNS)

    while True:
        try:
            user_input = Prompt.ask("[bold green]›[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/dim]")
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            console.print("[dim]Hasta luego.[/dim]")
            return 0

        composed_query = _format_history(history) + user_input

        with console.status("[cyan]Pensando…[/cyan]", spinner="dots"):
            result = agent.invoke({"query": composed_query})

        response = result.get("response", "")
        console.print(
            Panel(Markdown(response), title="🤖 Respuesta", border_style="cyan")
        )

        sources = result.get("retrieved_notes", [])
        if sources:
            console.print("[bold]Fuentes:[/bold]")
            for note in sources:
                console.print(f"  • [[{note.title}]] — {note.path}")
            console.print()

        history.append((user_input, response))


if __name__ == "__main__":
    sys.exit(main())
