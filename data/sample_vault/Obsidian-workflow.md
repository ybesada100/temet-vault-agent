---
tags: [obsidian, workflow, productivity, para, dataview]
---

# Obsidian Workflow — sistema PARA + daily notes + dataview

Mi setup actual de Obsidian, refinado a lo largo de tres años. No es teoría: es lo que uso a diario para escribir, investigar, y operar negocios.

## Principio rector

**Notas que se escriben solas no sirven.** El vault no es archivo, es taller. Si una nota no se vincula con otras, no existe.

## Estructura PARA (modificada)

Tomo PARA de Tiago Forte, pero adaptado:

```
00-inbox/         # Captura rápida, sin estructura
01-projects/      # Activos, con deadline
02-areas/         # Responsabilidades continuas (salud, finanzas, equipo)
03-resources/     # Conocimiento por dominio
04-archive/       # Lo que ya no toco
05-daily/         # Daily notes (auto-generadas)
99-meta/          # Templates, queries, configuración
```

Diferencia clave vs PARA puro: **inbox al frente** (orden alfabético importa) y **daily notes en su propia rama** porque crecen a miles.

## Daily notes — la columna vertebral

Cada día genero una nota con plantilla:

```markdown
# 2026-05-06

## Foco del día
- [ ] 

## Sesiones de trabajo
### 09:00 — 

## Capturas y links
- 

## Reflexión nocturna
- 
```

Las daily notes **no almacenan conocimiento permanente**. Son scratch pad. Si algo merece quedarse, se promueve a `03-resources/` con contexto y links.

## Dataview — el secreto

Dataview convierte el vault en una base de datos. Queries que uso a diario:

### Tareas pendientes por proyecto

```dataview
TASK
FROM "01-projects"
WHERE !completed
GROUP BY file.link
SORT file.mtime DESC
```

### Notas creadas esta semana

```dataview
TABLE file.cday as "Creada", length(file.outlinks) as "Links"
FROM ""
WHERE file.cday >= date(today) - dur(7 days)
SORT file.cday DESC
```

### Notas huérfanas (sin links de entrada)

```dataview
LIST
FROM ""
WHERE length(file.inlinks) = 0 AND length(file.outlinks) > 0
```

Esa última es la **brújula del vault**: notas sin inlinks son notas que nadie está usando. O las conecto, o las archivo.

## Plugins core

- **Templater** — plantillas dinámicas (daily notes, meeting notes)
- **Dataview** — queries
- **Obsidian Git** — backup automático a repo privado
- **Quick Switcher++** — navegación rápida por título
- **Excalidraw** — sketches inline

Nada de grafos vistosos sin propósito. El graph view es lindo pero rara vez accionable.

## Reglas de oro

1. **Una idea por nota.** Si una nota tiene dos H1 diferentes, son dos notas.
2. **Linkea agresivamente.** Cada concepto importante = `[[link]]`.
3. **Tags son taxonomía, no resumen.** 5 tags max por nota.
4. **No pierdas tiempo en Folder Hierarchy.** PARA + tags + links es suficiente.
5. **Revisa el inbox semanalmente.** Lo que no se procesa en 7 días, muere.

## Cómo conecta con un agente LLM

El vault así estructurado es **ideal para retrieval**: títulos descriptivos, frontmatter consistente, links explícitos. Un agente con BM25 sobre este vault encuentra lo relevante el 80% de las veces sin embeddings. Con embeddings, sube a 95%.
