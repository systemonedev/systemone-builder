"""BYOM (Bring Your Own Model) adapter interface.

Every model endpoint - the vLLM student on GPU 0, the vLLM triage server on
GPU 1, the Ollama oracle on the Mac - is reached through a
:class:`ModelAdapter`. Adapters stream responses so time-to-first-token is
measured precisely, and surface per-token logprobs when the backend provides
them (used by the confidence scorer).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenLogprob:
    token: str
    logprob: float


@dataclass
class Generation:
    text: str
    model: str
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    logprobs: list[TokenLogprob] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class AdapterError(RuntimeError):
    pass


class ModelAdapter(abc.ABC):
    kind: str = "base"

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        images: list[str] | None = None,
        logprobs: bool = False,
        model: str | None = None,
    ) -> Generation: ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "base_url": self.base_url, "model": self.model}
