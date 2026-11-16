"""Anthropic Messages API adapter for benchmarking frontier Claude models."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from systemone_builder.adapters.base import AdapterError, Generation, ModelAdapter


class AnthropicAdapter(ModelAdapter):
    kind = "anthropic"

    def __init__(self, base_url: str = "https://api.anthropic.com", model: str = "claude-sonnet-4-5",
                 timeout_s: float = 60.0, api_key: str | None = None) -> None:
        super().__init__(base_url, model, timeout_s, api_key)
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        images: list[str] | None = None,
        logprobs: bool = False,
        model: str | None = None,
    ) -> Generation:
        if not self.api_key:
            raise AdapterError("no Anthropic API key: set ANTHROPIC_API_KEY in .env")
        system_parts: list[str] = []
        chat_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(str(m.get("content", "")))
            else:
                chat_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if images and chat_msgs:
            last = chat_msgs[-1]
            blocks: list[dict[str, Any]] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
                for b64 in images
            ]
            blocks.append({"type": "text", "text": str(last.get("content", ""))})
            chat_msgs = [*chat_msgs[:-1], {"role": last["role"], "content": blocks}]

        body: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_msgs,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if json_schema is not None:
            body["tools"] = [{
                "name": "action_output",
                "description": "Emit the structured decision output matching the required schema.",
                "input_schema": json_schema,
            }]
            body["tool_choice"] = {"type": "tool", "name": "action_output"}

        path = "/messages" if self.base_url.endswith("/v1") else "/v1/messages"
        t0 = time.perf_counter()
        try:
            resp = await self.client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise AdapterError(f"{self.base_url}: {exc!r}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            raise AdapterError(f"Anthropic HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        text = ""
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and block.get("name") == "action_output":
                text = json.dumps(block.get("input", {}))
                break
            if block.get("type") == "text":
                text += block.get("text", "")
        usage = data.get("usage") or {}
        return Generation(
            text=text.strip(),
            model=data.get("model", body["model"]),
            ttft_ms=latency_ms,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            cached_tokens=usage.get("cache_read_input_tokens"),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": bool(self.api_key), "model": self.model}

    async def list_models(self) -> list[str]:
        return [self.model]

    async def aclose(self) -> None:
        await self.client.aclose()
