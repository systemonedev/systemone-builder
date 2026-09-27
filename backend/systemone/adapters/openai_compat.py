"""OpenAI-compatible adapter (vLLM student/triage servers, LM Studio, TGI ...)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from systemone.adapters.base import AdapterError, Generation, ModelAdapter, TokenLogprob


class OpenAICompatAdapter(ModelAdapter):
    kind = "openai"

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0, api_key: str | None = None) -> None:
        super().__init__(base_url, model, timeout_s, api_key)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Keep-alive pool: avoids TCP/TLS setup on the sub-100ms path.
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=128),
        )

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
    ) -> Generation:
        if images:
            messages = [*messages[:-1], _with_images(messages[-1], images)]
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action_output", "schema": json_schema, "strict": True},
            }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 1

        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        lps: list[TokenLogprob] = []
        usage: dict[str, Any] = {}
        finish: str | None = None
        used_model = body["model"]
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:500]
                    raise AdapterError(f"{self.base_url} HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    used_model = chunk.get("model", used_model)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            parts.append(delta)
                        for lp in ((choice.get("logprobs") or {}).get("content") or []):
                            lps.append(TokenLogprob(lp.get("token", ""), float(lp.get("logprob", 0.0))))
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
        except httpx.HTTPError as exc:
            raise AdapterError(f"{self.base_url}: {exc!r}") from exc
        details = usage.get("prompt_tokens_details") or {}
        return Generation(
            text="".join(parts),
            model=used_model,
            ttft_ms=ttft,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cached_tokens=details.get("cached_tokens"),
            logprobs=lps or None,
            finish_reason=finish,
        )

    async def health(self) -> dict[str, Any]:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            r = await self.client.get(f"{root}/health")
            return {"ok": r.status_code == 200, "status": r.status_code}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": repr(exc)}

    async def list_models(self) -> list[str]:
        r = await self.client.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def metrics_text(self) -> str:
        """Raw Prometheus metrics from a vLLM server (prefix-cache stats)."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        r = await self.client.get(f"{root}/metrics")
        r.raise_for_status()
        return r.text

    async def post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return await self.client.post(f"{root}{path}", json=payload)

    async def aclose(self) -> None:
        await self.client.aclose()


def _with_images(message: dict[str, Any], images: list[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": message.get("content", "")}]
    for b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return {**message, "content": content}
