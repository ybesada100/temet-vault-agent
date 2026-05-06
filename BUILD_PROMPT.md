# Build complete temet-vault-agent

Estás en el directorio `~/projects/temet-vault-agent`. Ya existe README.md, pyproject.toml, .gitignore.

Construye TODO el código fuente para que el repo sea **end-to-end funcional**.

## Archivos a crear

### `src/__init__.py`
Empty file (makes src a package).

### `src/memory.py`
Persistent memory with markdown + JSONL.
- Class `Memory(base_dir: Path)` — base_dir default `~/.temet-vault/memory/`
- Method `log_interaction(query: str, response: str, metadata: dict)` — appends to `interactions.jsonl`
- Method `recent_interactions(n: int = 10) -> List[dict]`
- Method `save_session_summary(summary: str)` — writes to `sessions/<date>.md`
- Auto-create directories.

### `src/retrieval.py`
BM25 retrieval over Obsidian vault.
- Class `VaultRetriever(vault_path: Path)`
- Walks all `.md` files in vault recursively (skip `.obsidian/`)
- Parses each note: extracts title (first H1 or filename), tags from frontmatter, body
- Builds BM25 index with rank_bm25
- Method `search(query: str, top_k: int = 5) -> List[Note]` returns notes ranked
- Class `Note(title: str, path: str, body: str, tags: list[str], score: float)`

### `src/llm.py`
Ollama wrapper.
- Function `generate(prompt: str, model: str = "qwen3:30b-a3b-instruct-q4_K_M", system: str = None, temperature: float = 0.4) -> str`
- Uses ollama Python lib
- Handle connection errors gracefully (suggest user run `ollama serve`)
- Stream-friendly version: `generate_stream(prompt, ...) -> Iterator[str]`

### `src/agent.py`
LangGraph agent with state machine.
- State TypedDict: query, retrieved_notes (List[Note]), context (str), response (str), needs_clarification (bool)
- Nodes:
  1. `parse_query`: detect intent (search vs question vs follow-up)
  2. `retrieve`: call VaultRetriever.search(state.query, top_k=5)
  3. `build_context`: format top notes as markdown context
  4. `generate`: call llm with system prompt + context + query
  5. `log`: save to Memory
- Edges: parse_query → retrieve → build_context → generate → log → END
- System prompt for LLM: "Eres un asistente que responde basándote en notas del vault Obsidian del usuario. Cita la nota fuente con [[título]]. Si no tienes contexto suficiente, dilo honestamente."
- Function `build_agent(vault_path: Path, memory: Memory) -> CompiledGraph`

### `src/main.py`
CLI entry point usando `rich` para pretty print.
- Reads OBSIDIAN_VAULT_PATH env var, default `data/sample_vault`
- Init: VaultRetriever + Memory + agent
- Loop: prompt user → run agent → pretty-print response with sources
- Commands: `/help`, `/recent`, `/exit`
- Show sources used (note titles) at the end of each response.

### `examples/ask_vault.py`
One-shot query example. Usage: `python examples/ask_vault.py "qué dije sobre Cuban economy?"`

### `examples/chat.py`
Conversational loop with history (each turn knows previous turns).

### `tests/test_basic.py`
Pytest tests:
- test_memory_persistence
- test_retrieval_finds_relevant_note
- test_agent_full_pipeline (mock Ollama with monkeypatch)

### `data/sample_vault/`
Crear 4 notas Obsidian de ejemplo, real y útil:
1. `Cuban-Economy.md` — análisis brutal de remesas, dolarización, mercado privado
2. `LLM-local-inference.md` — comparativa Qwen 30B-A3B vs Llama 70B en RTX 5070
3. `Obsidian-workflow.md` — sistema PARA + daily notes + dataview queries
4. `LangGraph-vs-OpenClaw.md` — comparativa técnica ya escrita en README, expandida

Cada una con frontmatter (`---\ntags: [...]\n---`), H1, contenido sustantivo (300-500 palabras cada una).

## Reglas

1. **Type hints** en todo
2. **Docstrings** breves pero claros
3. **Error handling** explícito (Ollama down, vault path missing, etc.)
4. **No mock data en producción** — sample_vault es solo para test
5. **Imports limpios** — no `import *`
6. Código en **inglés** (variables, funciones, comentarios técnicos), pero strings de UI/system prompts en **español refinado**
7. Después de crear todo, ejecuta `python -c "from src.agent import build_agent; print('OK imports')"` para validar que no hay errors de sintaxis

## Final step

Cuando termines, dame un summary de:
- Archivos creados (paths absolutos)
- Líneas totales de código
- Lo que probé (si testeaste algo)
- Posibles issues que detectaste

Empieza ahora.
