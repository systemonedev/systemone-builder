"""BYOM adapter registry.

Built-in kinds:

* ``openai`` - any OpenAI-compatible server: vLLM, SGLang, llama.cpp
  ``llama-server``, LM Studio, TGI (Messages API), LocalAI ...
* ``ollama`` - Ollama native API (vision images, ``format`` JSON schema)

Third-party adapters register through the ``systemone.adapters`` entry-point
group, e.g. in the plugin's ``pyproject.toml``::

    [project.entry-points."systemone.adapters"]
    mlx = "my_pkg.adapters:MLXAdapter"

The class must subclass :class:`~systemone.adapters.base.ModelAdapter`.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from systemone.adapters.base import ModelAdapter
from systemone.adapters.ollama import OllamaAdapter
from systemone.adapters.openai_compat import OpenAICompatAdapter

log = logging.getLogger(__name__)

ADAPTERS: dict[str, type[ModelAdapter]] = {"openai": OpenAICompatAdapter, "ollama": OllamaAdapter}
_plugins_loaded = False


def load_plugins() -> dict[str, type[ModelAdapter]]:
    global _plugins_loaded
    if not _plugins_loaded:
        for ep in entry_points(group="systemone.adapters"):
            try:
                cls = ep.load()
                if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
                    raise TypeError(f"{ep.value} is not a ModelAdapter subclass")
                ADAPTERS.setdefault(ep.name, cls)
            except Exception as exc:  # a broken plugin must not take the API down
                log.error("failed to load adapter plugin %s: %s", ep.name, exc)
        _plugins_loaded = True
    return ADAPTERS


def build_adapter(kind: str, base_url: str, model: str, timeout_s: float, api_key: str | None = None) -> ModelAdapter:
    registry = load_plugins()
    try:
        cls = registry[kind]
    except KeyError:
        raise ValueError(f"unknown adapter kind {kind!r}; available: {sorted(registry)}") from None
    return cls(base_url, model, timeout_s=timeout_s, api_key=api_key)
