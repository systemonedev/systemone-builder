from __future__ import annotations

from systemone.adapters.base import ModelAdapter
from systemone.adapters.ollama import OllamaAdapter
from systemone.adapters.openai_compat import OpenAICompatAdapter

ADAPTERS: dict[str, type[ModelAdapter]] = {"openai": OpenAICompatAdapter, "ollama": OllamaAdapter}


def build_adapter(kind: str, base_url: str, model: str, timeout_s: float, api_key: str | None = None) -> ModelAdapter:
    try:
        cls = ADAPTERS[kind]
    except KeyError:
        raise ValueError(f"unknown adapter kind {kind!r}; available: {sorted(ADAPTERS)}") from None
    return cls(base_url, model, timeout_s=timeout_s, api_key=api_key)
