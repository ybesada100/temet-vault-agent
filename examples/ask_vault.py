"""One-shot query against the vault.

Usage::

    python examples/ask_vault.py "¿qué dije sobre Cuban economy?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python examples/ask_vault.py` from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402

from src.agent import build_agent  # noqa: E402
from src.memory import Memory  # noqa: E402
from src.retrieval import VaultRetriever  # noqa: E402


def main(argv: list[str]) -> int:
    """Entry point for one-shot queries."""
    console = Console()
    if len(argv) < 2:
        console.print(
            "[red]Uso:[/red] python examples/ask_vault.py "
            '"tu pregunta aquí"'
        )
        return 2

    query = " ".join(argv[1:]).strip()
    if not query:
        console.print("[red]La pregunta está vacía.[/red]")
        return 2

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

    console.print(f"[dim]Vault:[/dim] {vault_path} ({len(retriever.notes)} notas)")
    console.print(f"[dim]Pregunta:[/dim] {query}\n")

    with console.status("[cyan]Pensando…[/cyan]", spinner="dots"):
        result = agent.invoke({"query": query})

    console.print(
        Panel(Markdown(result.get("response", "")), title="🤖 Respuesta", border_style="cyan")
    )
    sources = result.get("retrieved_notes", [])
    if sources:
        console.print("\n[bold]Fuentes:[/bold]")
        for note in sources:
            console.print(f"  • [[{note.title}]] — {note.path} (score {note.score:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
