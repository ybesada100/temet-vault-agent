# Temet Vault Agent

Agente local que conversa con tu vault de Obsidian usando **LangGraph + Ollama + Qwen 3.6**. 100% OSS, corre en tu máquina, memoria persistente. Desde **v0.2** también puede actuar autónomamente: programar ciclos, escribir notas nuevas y aprender de su propio historial.

> Construido pa' [@IgnotusBTC](https://x.com/IgnotusBTC) — todos los devs que quieran un agente sobre su vault.

## Stack

- **Orquestador**: [LangGraph](https://github.com/langchain-ai/langgraph) — DAG explícito, checkpointing nativo
- **LLM local**: [Ollama](https://ollama.ai/) + Qwen 3.6 30B-A3B (Q4_K_M)
- **Retrieval**: BM25 sobre vault Obsidian (sin embeddings, simple y rápido)
- **Memoria**: markdown + JSONL append-only (cero deps, debuggable)
- **Tools**: WriteTool / AppendTool / LinkTool / TaskTool — exportadas como `StructuredTool` de LangChain
- **Scheduler**: asyncio loop con state.json persistente y handlers SIGINT/SIGTERM
- **Feedback**: extracción de patrones cada N interacciones → `strategy.md`

## Hardware mínimo

- 16GB RAM (32GB recomendado para 30B-A3B Q4_K_M)
- GPU opcional (Ollama con GPU offload acelera)
- 25GB disk para el modelo

## Quick start

```bash
# 1. Clonar
git clone https://github.com/ybesada100/temet-vault-agent.git
cd temet-vault-agent

# 2. Install deps (con uv recomendado)
pip install uv
uv pip install -e .

# 3. Pull modelo en Ollama
ollama pull qwen3:30b-a3b-instruct-q4_K_M

# 4. Apuntar al vault (default usa data/sample_vault)
export OBSIDIAN_VAULT_PATH=~/Documents/MyVault

# 5a. Modo Q&A interactivo (v0.1)
python -m src.main

# 5b. Modo autónomo (v0.2) — un cycle cada 5 minutos
python examples/autonomous_loop.py --interval 5
```

## Estructura

```
src/
├── memory.py       # Persistencia markdown + jsonl
├── retrieval.py    # BM25 sobre notas .md del vault
├── agent.py        # LangGraph state machine (v0.2: + write/feedback nodes)
├── llm.py          # Ollama wrapper
├── main.py         # CLI Q&A interactivo
├── scheduler.py    # ⚡ asyncio loop + state persistence
├── feedback.py     # ⚡ FeedbackLoop (lessons → strategy)
└── tools/          # ⚡ WriteTool / AppendTool / LinkTool / TaskTool
examples/
├── ask_vault.py    # Una pregunta one-shot
├── chat.py         # Modo conversacional
└── autonomous_loop.py  # ⚡ Loop autónomo end-to-end
systemd/
├── temet-agent.service # ⚡ Long-running user service
└── temet-agent.timer   # ⚡ Alternativa: timer one-shot
data/sample_vault/  # Notas Obsidian de ejemplo
tests/              # 24 tests offline (Ollama mockeado)
```

## Cómo funciona el agente (LangGraph)

### v0.1 (sigue funcionando, default)

```
User query
    ↓
[parse_query]  Detecta tipo: search / question / follow_up
    ↓
[retrieve]     BM25 top-k sobre vault
    ↓
[build_context]
    ↓
[generate]     Qwen genera respuesta con contexto
    ↓
[log]          Append a memoria persistente
    ↓
END
```

### v0.2 — Autonomous Mode

Tres capas nuevas convierten el bot en un agente real:

```
                                ┌─ write_note ─┐
… → generate → should_write_note┤              ├─ log → update_feedback → END
                                └─ skip_write ─┘
```

1. **Loop scheduler** (`src/scheduler.py`)
   - `Scheduler(cycle_fn, interval_minutes=5)` corre `cycle_fn` cada N min
   - Async-friendly, soporta callables sync o async
   - Estado en `~/.temet-vault/state.json`: `last_run`, `run_count`, `errors`, `last_error`
   - Shutdown limpio con SIGINT/SIGTERM (también funciona bajo systemd)
   - `max_cycles=N` para tests / runs acotados

2. **Tools que escriben** (`src/tools/__init__.py`)
   - `WriteTool` — crea nota nueva con frontmatter (`tags`, `created`, `agent`)
   - `AppendTool` — agrega bloque a una nota existente
   - `LinkTool` — agrega `[[wikilink]]` bajo `## Backlinks` (idempotente)
   - `TaskTool` — agrega `- [ ] tarea` al daily note del día (lo crea si falta)
   - Todas exportables vía `as_langchain_tools(vault_path)` para usar con tool-calling LLMs
   - Path-safety: cualquier intento de escapar del vault root tira `VaultPathError`

3. **Memoria con feedback** (`src/feedback.py`)
   - `FeedbackLoop(memory)` lee las últimas 50 interacciones
   - Cada 10 interacciones (configurable), pide al LLM extraer patrones
   - Append timestampeado a `~/.temet-vault/memory/strategy.md`
   - `strategy.md` queda disponible para re-inyectar en prompts futuros (no fine-tuning, solo context engineering)

### Activar el modo autónomo desde código

```python
from pathlib import Path
from src.agent import build_agent
from src.feedback import FeedbackLoop
from src.memory import Memory

memory = Memory()
feedback = FeedbackLoop(memory, trigger_every=10)

agent = build_agent(
    vault_path=Path("~/Documents/MyVault").expanduser(),
    memory=memory,
    write_enabled=True,         # NEW — el agente puede crear notas
    feedback=feedback,          # NEW — patterns → strategy.md
    write_subdir="agent-notes", # dónde van las notas auto-generadas
)

result = agent.invoke({"query": "sintetiza lo que aprendí esta semana"})
print(result["written_path"])   # path de la nota nueva, o None si SKIP
```

> **Backward compat**: si NO pasas `write_enabled=True`, el grafo se comporta idéntico a v0.1 (los tests de v0.1 siguen pasando sin cambios).

## Quick start del autonomous loop

```bash
# Dry-run: 2 cycles, no escribe nada al vault
python examples/autonomous_loop.py --dry-run --max-cycles 2 --interval 0.05

# Producción: 5-min cadence, escribe sugerencias en el daily note
export OBSIDIAN_VAULT_PATH=~/Documents/MyVault
python examples/autonomous_loop.py --interval 5

# Verbose
python examples/autonomous_loop.py --interval 5 -v
```

Cada cycle el script:

1. Abre el daily note del día (lo crea si no existe).
2. Lee los `- [ ]` TODOs sin terminar.
3. Por cada TODO (cap a 3 por cycle), pregunta al agente "¿próximo paso?".
4. Append de la sugerencia bajo el TODO en el daily note.
5. Logea acción + actualiza `state.json`.

## Cómo correr como servicio systemd

```bash
# Copiar templates
mkdir -p ~/.config/systemd/user
cp systemd/temet-agent.service ~/.config/systemd/user/

# Editar paths reales (WorkingDirectory + OBSIDIAN_VAULT_PATH)
$EDITOR ~/.config/systemd/user/temet-agent.service

# Habilitar y arrancar
systemctl --user daemon-reload
systemctl --user enable --now temet-agent.service

# Logs en vivo
journalctl --user -u temet-agent -f
```

Variante con timer one-shot: ver `systemd/README.md`.

## Limitaciones honestas

- **Sin internet**: no hay tools de fetch / search / browse. El agente solo conoce lo que vive en tu vault + lo que ya generó.
- **Sin embeddings** (todavía): retrieval es puro BM25 — falla con sinónimos / paráfrasis. Roadmap v0.3.
- **Single-vault**: no hay multi-vault routing.
- **Modelo local fijo**: Qwen 30B-A3B por default. Cambia con `model="otra:tag"` en `build_agent()`.
- **Feedback recursivo**: si el LLM extrae insights tóxicos, se acumulan en `strategy.md`. Audita ese archivo periódicamente.
- **No human-in-the-loop por default**: el agente decide solo cuándo escribir. Si querés review previo, pon `write_enabled=False` y wrappea la decisión.

## Roadmap

- [x] v0.1: Single-shot Q&A sobre vault
- [x] **v0.2: Autonomous mode (scheduler + tools + feedback)**
- [ ] v0.3: Embedding retrieval (sentence-transformers)
- [ ] v0.4: Router cloud/local (MiniMax M2.7 para razonamiento largo)
- [ ] v0.5: Multi-agent (research → write → critique)
- [ ] v0.6: Webhooks (event-driven cuando vault cambia)

## Por qué LangGraph y no OpenClaw?

| | LangGraph | OpenClaw |
|---|-----------|----------|
| DAG explícito | ✅ | ❌ |
| Checkpointing nativo | ✅ | ❌ |
| Streaming | ✅ | parcial |
| Human-in-the-loop | ✅ nativo | manual |
| Observabilidad | LangSmith brutal | logs |
| Curva | media | baja |

OpenClaw gana en simplicidad. LangGraph gana en control fino + producción.

## Tests

```bash
pytest tests/ -v
# 24 passed in 0.4s — todo offline, Ollama mockeado.
```

## Licencia

MIT — adáptalo a tu setup.

---

Construido por [@IALabMiami](https://x.com/IALabMiami) en un DGX Spark con voz cubana refinada.
