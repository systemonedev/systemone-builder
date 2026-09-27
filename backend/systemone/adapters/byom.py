"""Bring-Your-Own-Model configuration and validation.

Roles and their defaults come from :mod:`systemone.config` (``S1_*`` env).
An optional ``systemone.yaml`` (path in ``S1_BYOM_FILE``) overrides them::

    student:
      base_model: meta-llama/Llama-3.2-1B-Instruct   # any HF causal LM Unsloth supports
      url: http://student:8000/v1
    triage:
      adapter: openai
      url: http://triage:8000/v1
      model: Qwen/Qwen2.5-14B-Instruct-AWQ
    oracle:
      adapter: ollama
      url: http://192.168.1.50:11434
      model: qwen3.8:27b
      vision_model: qwen2.5vl:32b
      api_key_env: ORACLE_KEY        # read the key from this env var

Endpoints can also be swapped at runtime through ``PUT /api/v1/byom/{role}``;
the change is persisted in Redis and survives restarts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel

from systemone.config import Settings

Role = Literal["student", "triage", "oracle"]

# model_type values (config.json) that Unsloth's FastLanguageModel supports.
UNSLOTH_MODEL_TYPES = {
    "llama", "mistral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "gemma", "gemma2", "gemma3", "gemma3_text",
    "phi3", "phi", "cohere", "granite", "falcon", "gpt_oss", "olmo2", "smollm3", "mixtral", "deepseek_v3",
}


class EndpointConfig(BaseModel):
    adapter: str
    url: str
    model: str
    api_key_env: str | None = None
    vision_model: str | None = None
    base_model: str | None = None  # student only: HF id to fine-tune

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


def defaults(s: Settings) -> dict[str, EndpointConfig]:
    return {
        "student": EndpointConfig(adapter=s.student_adapter, url=s.student_url, model=s.student_served_name, base_model=s.student_base_model),
        "triage": EndpointConfig(adapter=s.triage_adapter, url=s.triage_url, model=s.triage_model),
        "oracle": EndpointConfig(adapter=s.oracle_adapter, url=s.oracle_url, model=s.oracle_model, vision_model=s.oracle_vision_model),
    }


def load_file(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text())
    return {k: v for k, v in (data or {}).items() if k in ("student", "triage", "oracle") and isinstance(v, dict)}


def resolve(s: Settings, overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, EndpointConfig]:
    out = defaults(s)
    for layer in (load_file(s.byom_file), overrides or {}):
        for role, patch in layer.items():
            out[role] = out[role].model_copy(update={k: v for k, v in patch.items() if v is not None})
    return out


async def validate_hf_model(model_id: str, token: str | None = None) -> dict[str, Any]:
    """Check that a Hugging Face model exists and Unsloth can fine-tune it."""
    if Path(model_id).exists():
        cfg_path = Path(model_id) / "config.json"
        if not cfg_path.exists():
            return {"ok": False, "model": model_id, "error": "local path has no config.json"}
        cfg = json.loads(cfg_path.read_text())
    else:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                r = await c.get(url, headers=headers)
        except httpx.HTTPError as exc:
            return {"ok": False, "model": model_id, "error": f"hub unreachable: {exc!r}"}
        if r.status_code in (401, 403):
            return {"ok": False, "model": model_id, "error": "gated or private model: set HF_TOKEN and accept the license"}
        if r.status_code != 200:
            return {"ok": False, "model": model_id, "error": f"config.json not found (HTTP {r.status_code})"}
        cfg = r.json()
    mtype = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type")
    params_hint = cfg.get("num_hidden_layers"), cfg.get("hidden_size")
    supported = mtype in UNSLOTH_MODEL_TYPES
    return {
        "ok": supported,
        "model": model_id,
        "model_type": mtype,
        "architectures": cfg.get("architectures"),
        "layers_hidden": params_hint,
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "error": None if supported else f"model_type {mtype!r} is not in the Unsloth-supported set",
    }


class AdapterHandle:
    """Stable reference to a role's adapter whose backend can be hot-swapped.

    Every component (router, factory, evaluator, telemetry) holds the handle,
    so ``PUT /byom/{role}`` takes effect everywhere at once.
    """

    def __init__(self, role: str, inner: Any, config: EndpointConfig) -> None:
        self.role = role
        self._inner = inner
        self.config = config

    @property
    def inner(self) -> Any:
        return self._inner

    async def swap(self, inner: Any, config: EndpointConfig) -> None:
        old, self._inner, self.config = self._inner, inner, config
        await old.aclose()

    def describe(self) -> dict[str, Any]:
        return {**self._inner.describe(), "role": self.role, "config": self.config.model_dump(exclude={"api_key_env"})}

    def __getattr__(self, name: str) -> Any:  # generate, health, list_models, metrics_text, ...
        return getattr(self._inner, name)
