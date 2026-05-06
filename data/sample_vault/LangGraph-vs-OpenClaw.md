---
tags: [langgraph, openclaw, agents, framework, comparison]
---

# LangGraph vs OpenClaw — comparativa técnica

Decisión recurrente al construir agentes en 2026: ¿LangGraph (LangChain stack) o OpenClaw (framework liviano)? Tras varios prototipos en producción, este es el veredicto detallado.

## Resumen ejecutivo

- **LangGraph** = control fino + producción. Curva media, ROI alto.
- **OpenClaw** = velocidad de prototipado + simplicidad. Curva baja, techo bajo.

Para un agente que vas a operar **6+ meses con telemetría real**: LangGraph. Para un demo de fin de semana o un proof-of-concept: OpenClaw.

## Tabla de decisión

| Dimensión | LangGraph | OpenClaw |
|-----------|-----------|----------|
| DAG explícito | ✅ Sí (StateGraph nodes/edges) | ❌ Implícito |
| Checkpointing nativo | ✅ Sí (memoria + reanudable) | ❌ Manual |
| Streaming de tokens | ✅ Sí (`.stream()`, `.astream()`) | ⚠️ Parcial |
| Human-in-the-loop | ✅ Nativo (interrupt) | ❌ Manual |
| Observabilidad | LangSmith + traces detallados | Logs simples |
| Curva de aprendizaje | Media (TypedDict + nodes) | Baja (decoradores) |
| Tamaño del runtime | Pesado (~150MB instalado) | Ligero (~15MB) |
| Tool use | Robusto (binding + retries) | Limitado |
| Compatibilidad cloud | LangSmith, LangGraph Cloud | DIY |
| Madurez | Alta (LangChain ecosystem) | Media |

## Cuándo usar LangGraph

- Pipelines de varios pasos donde el estado importa entre nodos.
- Agentes con human-in-the-loop (aprobación, edición, intervención).
- Multi-agent (research → write → critique con paso de mensajes).
- Necesitas reanudar conversaciones tras crash o cierre.
- Telemetría con LangSmith para debug en prod.

## Cuándo usar OpenClaw

- Single-shot: una pregunta, un retrieval, una respuesta.
- Prototipo de fin de semana sin compromiso de mantenimiento.
- Cuando el equipo no tiene experiencia con LangChain.
- Cuando deps mínimos importa (containers chicos, edge deploy).

## El gotcha de LangGraph

**El estado es TypedDict, no un objeto.** Los devs vienen del paradigma OOP esperando `state.foo = bar`, pero en LangGraph cada nodo retorna un **patch parcial** del estado:

```python
def my_node(state: AgentState) -> AgentState:
    return {"response": "hola"}  # Solo lo que cambia.
```

LangGraph hace el merge automáticamente. Si retornas el state completo, se sobrescribe todo y rompes la composición.

## El gotcha de OpenClaw

**No hay checkpointing.** Si el proceso muere a mitad de un agente complejo, perdiste todo. En prod, esto significa que cualquier cosa con sesiones largas necesita una capa de persistencia adicional, lo que termina siendo **reimplementar LangGraph mal**.

## Mi recomendación 2026

Usa **LangGraph** salvo que tengas razón fuerte para no hacerlo. La curva inicial es real (1-2 días para entender bien `StateGraph`, `TypedDict`, edges condicionales), pero el techo es alto. Una vez que el equipo lo internaliza, todos los agentes nuevos parten de un mismo template y se vuelve _commodity_ rápido.

OpenClaw lo dejaría para scripts y experiments. No para producción.

## Stack recomendado para agentes locales

- **Orquestador:** LangGraph 0.2+
- **LLM:** Ollama + Qwen 3 30B-A3B (ver [[LLM-local-inference]])
- **Retrieval:** BM25 sobre vault (ver [[Obsidian-workflow]])
- **Memoria:** JSONL + markdown append-only
- **Observabilidad:** LangSmith opcional, structlog default

Todo OSS, todo local, todo controlable.
