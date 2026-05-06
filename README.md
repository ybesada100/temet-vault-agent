# Temet Vault Agent

Agente local que conversa con tu vault de Obsidian usando **LangGraph + Ollama + Qwen 3.6**. 100% OSS, corre en tu máquina, memoria persistente.

> Construido pa' [@IgnotusBTC](https://x.com/IgnotusBTC) — todos los devs que quieran un agente sobre su vault.

## Stack

- **Orquestador**: [LangGraph](https://github.com/langchain-ai/langgraph) — DAG explícito, checkpointing nativo
- **LLM local**: [Ollama](https://ollama.ai/) + Qwen 3.6 30B-A3B (Q4_K_M)
- **Retrieval**: BM25 sobre vault Obsidian (sin embeddings, simple y rápido)
- **Memoria**: markdown + JSONL append-only (cero deps, debuggable)

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

# 5. Run
python -m src.main
```

## Estructura
src/
├── memory.py       # Persistencia markdown + jsonl
├── retrieval.py    # BM25 sobre notas .md del vault
├── agent.py        # LangGraph state machine
├── llm.py          # Ollama wrapper
└── main.py         # CLI entry
examples/
├── ask_vault.py    # Una pregunta one-shot
└── chat.py         # Modo conversacional
data/sample_vault/  # Notas Obsidian de ejemplo (4)
tests/              # Tests básicos
## Cómo funciona el agent (LangGraph)
User query
↓
[parse_query]  Detecta tipo: search / question / multi-hop
↓
[retrieve]     BM25 top-k sobre vault
↓
[generate]     Qwen genera respuesta con contexto
↓
[log]          Append a memoria persistente
↓
Response
## Roadmap

- [x] v0.1: Single-shot Q&A sobre vault
- [ ] v0.2: Embedding retrieval (sentence-transformers)
- [ ] v0.3: Router cloud/local (MiniMax M2.7 para razonamiento largo)
- [ ] v0.4: Multi-agent (research → write → critique)
- [ ] v0.5: Webhooks (event-driven cuando vault cambia)

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

## Licencia

MIT — adáptalo a tu setup.

---

Construido por [@IALabMiami](https://x.com/IALabMiami) en un DGX Spark con voz cubana refinada.
