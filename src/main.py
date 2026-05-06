"""CLI entry point for temet-vault-agent.

Usage::

    python -m src.main

Reads ``OBSIDIAN_VAULT_PATH`` (default: ``data/sample_vault``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agent import build_agent
from src.memory import Memory
from src.retrieval import VaultRetriever

HELP_TEXT = """\
**Comandos disponibles**

- `/help`     — Mostrar esta ayuda
- `/recent`   — Mostrar últimas interacciones registradas
- `/exit`     — Salir
- Cualquier otra cosa se trata como pregunta para el vault.
"""


def _resolve_vault_path() -> Path:
    """Resolve the vault path from env or fall back to bundled sample vault."""
    env = os.environ.get("OBSIDIAN_VAULT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "data" / "sample_vault").resolve()


def _render_response(console: Console, response: str, sources: list[dict]) -> None:
    """Pretty-print the agent's response and the sources it used."""
    console.print(Panel(Markdown(response), title="🤖 Respuesta", border_style="cyan"))
    if not sources:
        console.print("[dim]Sin fuentes citadas (vault vacío o consulta sin matches).[/dim]\n")
        return
    table = Table(title="📚 Fuentes utilizadas", show_lines=False, header_style="bold magenta")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Título", style="bold")
    table.add_column("Path", style="cyan")
    table.add_column("Score", justify="right")
    for idx, src in enumerate(sources, start=1):
        table.add_row(
            str(idx),
            str(src.get("title", "?")),
            str(src.get("path", "?")),
            f"{float(src.get('score', 0.0)):.2f}",
        )
    console.print(table)
    console.print()


def _show_recent(console: Console, memory: Memory) -> None:
    """Render recent interactions as a table."""
    rows = memory.recent_interactions(n=10)
    if not rows:
        console.print("[dim]Sin interacciones registradas todavía.[/dim]")
        return
    table = Table(title="🕘 Recientes (últimas 10)", header_style="bold green")
    table.add_column("Cuándo", style="dim")
    table.add_column("Pregunta", overflow="fold")
    table.add_column("Respuesta (extracto)", overflow="fold")
    for row in rows:
        ts = str(row.get("timestamp", ""))[:19].replace("T", " ")
        q = str(row.get("query", ""))
        r = str(row.get("response", ""))
        if len(r) > 120:
            r = r[:117] + "…"
        table.add_row(ts, q, r)
    console.print(table)


def main() -> int:
    """CLI entry. Returns process exit code."""
    console = Console()
    vault_path = _resolve_vault_path()

    if not vault_path.exists():
        console.print(
            f"[red]✗[/red] Vault path no existe: [bold]{vault_path}[/bold]\n"
            "Configura [yellow]OBSIDIAN_VAULT_PATH[/yellow] o usa el sample vault."
        )
        return 1

    console.print(
        Panel.fit(
            f"[bold cyan]temet-vault-agent[/bold cyan]\n"
            f"Vault: [yellow]{vault_path}[/yellow]\n"
            f"Memoria: [yellow]~/.temet-vault/memory[/yellow]",
            border_style="cyan",
        )
    )

    try:
        retriever = VaultRetriever(vault_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        return 1

    memory = Memory()
    agent = build_agent(vault_path=vault_path, memory=memory, retriever=retriever)

    console.print(f"[green]✓[/green] Indexadas [bold]{len(retriever.notes)}[/bold] notas.\n")
    console.print(Markdown(HELP_TEXT))

    while True:
        try:
            user_input = Prompt.ask("[bold green]›[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/dim]")
            return 0

        if not user_input:
            continue

        match user_input.lower():
            case "/exit" | "/quit":
                console.print("[dim]Hasta luego.[/dim]")
                return 0
            case "/help":
                console.print(Markdown(HELP_TEXT))
                continue
            case "/recent":
                _show_recent(console, memory)
                continue

        with console.status("[cyan]Pensando…[/cyan]", spinner="dots"):
            result = agent.invoke({"query": user_input})

        sources = [
            {"title": n.title, "path": n.path, "score": n.score}
            for n in result.get("retrieved_notes", [])
        ]
        _render_response(console, result.get("response", ""), sources)


if __name__ == "__main__":
    sys.exit(main())
