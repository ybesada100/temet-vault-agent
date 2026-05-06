"""Ollama wrapper.

Thin layer over the ``ollama`` Python client. Surfaces friendly errors when
the daemon isn't running, and exposes both blocking and streaming variants.
"""

from __future__ import annotations

from collections.abc import Iterator

import ollama
from ollama import ResponseError

DEFAULT_MODEL = "qwen3:30b-a3b-instruct-q4_K_M"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable."""


def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    """Compose the chat message list for ``ollama.chat``."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.4,
) -> str:
    """Run a single completion against an Ollama-served model.

    Args:
        prompt: User prompt.
        model: Ollama model tag (must be pulled locally).
        system: Optional system message.
        temperature: Sampling temperature.

    Returns:
        Generated text.

    Raises:
        OllamaUnavailableError: If the Ollama daemon is unreachable or the
            model cannot be loaded.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=_build_messages(prompt, system),
            options={"temperature": temperature},
        )
    except ConnectionError as exc:
        raise OllamaUnavailableError(
            "No se pudo conectar al daemon de Ollama. "
            "Ejecuta `ollama serve` en otra terminal y reintenta."
        ) from exc
    except ResponseError as exc:
        raise OllamaUnavailableError(
            f"Ollama respondió con error: {exc}. "
            f"Verifica que el modelo `{model}` esté disponible (`ollama pull {model}`)."
        ) from exc

    # ollama-python returns either a dict-like or a typed object depending on version.
    message = response["message"] if isinstance(response, dict) else response.message
    content = message["content"] if isinstance(message, dict) else message.content
    return content or ""


def generate_stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.4,
) -> Iterator[str]:
    """Streaming variant of :func:`generate`.

    Yields successive content chunks. Raises :class:`OllamaUnavailableError`
    on connection / model errors before the first token is produced.
    """
    try:
        stream = ollama.chat(
            model=model,
            messages=_build_messages(prompt, system),
            options={"temperature": temperature},
            stream=True,
        )
    except ConnectionError as exc:
        raise OllamaUnavailableError(
            "No se pudo conectar al daemon de Ollama. "
            "Ejecuta `ollama serve` en otra terminal y reintenta."
        ) from exc
    except ResponseError as exc:
        raise OllamaUnavailableError(
            f"Ollama respondió con error: {exc}. "
            f"Verifica que el modelo `{model}` esté disponible (`ollama pull {model}`)."
        ) from exc

    for chunk in stream:
        message = chunk["message"] if isinstance(chunk, dict) else chunk.message
        if message is None:
            continue
        content = message["content"] if isinstance(message, dict) else message.content
        if content:
            yield content
