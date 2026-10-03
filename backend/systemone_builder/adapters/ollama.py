"""Ollama adapter - the Mac M4 Max asynchronous oracle (``qwen3.8:27b``)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from systemone_builder.adapters.base import AdapterError, Generation, ModelAdapter, TokenLogprob


class OllamaAdapter(ModelAdapter):
    kind = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: float = 600.0, api_key: str | None = None,
                 keep_alive: str = "30m", num_ctx: int = 16384) -> None:
        super().__init__(base_url, model, timeout_s, api_key)
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0))

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        images: list[str] | None = None,
        logprobs: bool = False,
        model: str | None = None,
    ) -> Generation:
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = list(images)
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": msgs,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if json_schema is not None:
            body["format"] = json_schema
        if logprobs:
            body["logprobs"] = True
        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        lps: list[TokenLogprob] = []
        final: dict[str, Any] = {}
        try:
            async with self.client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:500]
                    raise AdapterError(f"{self.base_url} HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise AdapterError(f"ollama: {chunk['error']}")
                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        parts.append(content)
                    for lp in chunk.get("logprobs") or []:
                        lps.append(TokenLogprob(lp.get("token", ""), float(lp.get("logprob", 0.0))))
                    if chunk.get("done"):
                        final = chunk
        except httpx.HTTPError as exc:
            raise AdapterError(f"{self.base_url}: {exc!r}") from exc
        return Generation(
            text="".join(parts),
            model=final.get("model", body["model"]),
            ttft_ms=ttft,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=final.get("prompt_eval_count"),
            completion_tokens=final.get("eval_count"),
            logprobs=lps or None,
            finish_reason=final.get("done_reason"),
            raw={k: final.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
        )

    async def health(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/api/version")
            return {"ok": r.status_code == 200, **(r.json() if r.status_code == 200 else {})}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": repr(exc)}

    async def list_models(self) -> list[str]:
        r = await self.client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    async def pull(self, model: str | None = None) -> None:
        async with self.client.stream("POST", "/api/pull", json={"model": model or self.model}) as resp:
            async for _ in resp.aiter_lines():
                pass

    async def aclose(self) -> None:
        await self.client.aclose()
