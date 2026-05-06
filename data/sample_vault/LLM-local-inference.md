---
tags: [llm, inference, qwen, llama, hardware, ollama]
---

# LLM Local Inference — Qwen 30B-A3B vs Llama 70B en RTX 5070

Comparativa práctica corriendo modelos cuantizados en hardware accesible (RTX 5070 con 12GB VRAM + 64GB RAM DDR5). Foco en **latencia real, calidad de output, y costo operativo**, no en benchmarks sintéticos.

## Setup

- **GPU:** RTX 5070 12GB (Ada/Blackwell, no Hopper)
- **CPU:** Ryzen 9 7950X
- **RAM:** 64GB DDR5-6000
- **Runtime:** Ollama 0.5+ con backend llama.cpp + GPU offload parcial

## Qwen 3 30B-A3B Q4_K_M

Modelo MoE — solo 3B parámetros activos por token, pero 30B totales en memoria. Diseño nuevo de Alibaba (2024-2025).

- **VRAM usado:** ~7.5 GB con offload parcial. El resto va a RAM.
- **Tokens/s (prompt corto):** 38–45 t/s en generación.
- **Tokens/s (prompt largo, 4k contexto):** 22–28 t/s.
- **Calidad:** Excelente en español, razonamiento multi-paso, tool use. A nivel de Llama 3.1 70B en muchas tareas, a fracción del costo computacional.
- **Latencia primer token:** ~1.2s.

Veredicto: **el sweet spot actual** para inferencia local. MoE = comer pastel y tenerlo: parámetros totales como 30B, costo de inferencia como 3B activos.

## Llama 3.1 70B Q4_K_M

Dense, 70B parámetros, todos activos siempre.

- **VRAM usado:** Imposible cargarlo todo en 12GB. Offload mayoritario a RAM (~40GB).
- **Tokens/s (prompt corto):** 4–6 t/s. Inviable interactivo.
- **Tokens/s (prompt largo):** 2–3 t/s.
- **Calidad:** Marginalmente mejor que Qwen 30B-A3B en algunas tareas de razonamiento puro, peor en español, más rígido.
- **Latencia primer token:** 8–15s.

Veredicto: **no usar local en esta clase de hardware.** Sólo tiene sentido con dual GPU o offload mínimo (>32GB VRAM).

## Otros que probé

| Modelo | VRAM | t/s | Notas |
|--------|------|-----|-------|
| Llama 3.2 3B Q8 | 3.5 GB | 95 t/s | Brutal para tareas simples / clasificación |
| Mistral 7B Q5_K_M | 5.5 GB | 65 t/s | Sólido baseline general |
| DeepSeek-Coder 33B Q4 | 17 GB | 12 t/s | Mejor que Qwen para código puro |
| Phi-3.5-mini Q8 | 4 GB | 80 t/s | Razonamiento sorprendente para 3.8B |

## Lecciones

1. **MoE > Dense** en hardware consumer. Qwen 3 30B-A3B redefine el techo.
2. **Q4_K_M es el cuantizado óptimo** para calidad/tamaño en 2026.
3. **GPU offload parcial > full CPU.** Ollama lo hace bien automático.
4. **Contexto >8k empieza a doler.** Latencia no escala lineal en MoE tampoco.
5. **No persigas el modelo más grande.** Persigue el modelo correcto para la tarea.

## Default actual

Para agentes locales sobre vault Obsidian: **`qwen3:30b-a3b-instruct-q4_K_M`**. Punto. Razón: el balance latencia/calidad/RAM no lo iguala nadie en open-weight a esta fecha.
