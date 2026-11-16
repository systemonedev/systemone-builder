"""Build System One engines from settings."""

from __future__ import annotations

from systemone_builder.adapters.byom import resolve
from systemone_builder.adapters.factory import build_adapter
from systemone_builder.config import Settings
from systemone_builder.system_one.engines import (
    DirectClefEngine,
    DirectKenningEngine,
    DirectKenningXLEngine,
    JevEngine,
    LLMJsonEngine,
    LocalLogprobEngine,
    SystemOneEngine,
)

ENGINES = ("kenning", "clef", "clef-direct", "xl", "jev", "glide", "local", "llm", "openai", "anthropic", "gemini", "openrouter", "ollama")


def _build_frontier_llm(provider: str, model: str, s: Settings) -> SystemOneEngine:
    timeout = max(s.request_timeout_s, 90.0)
    if provider == "openai":
        return LLMJsonEngine(
            build_adapter("openai", s.openai_url, model, timeout, s.openai_api_key),
            name=f"openai ({model})",
        )
    if provider == "anthropic":
        return LLMJsonEngine(
            build_adapter("anthropic", s.anthropic_url, model, timeout, s.anthropic_api_key),
            name=f"anthropic ({model})",
        )
    if provider == "gemini":
        return LLMJsonEngine(
            build_adapter("openai", s.gemini_url, model, timeout, s.gemini_api_key),
            name=f"gemini ({model})",
        )
    if provider == "openrouter":
        return LLMJsonEngine(
            build_adapter("openai", s.openrouter_url, model, timeout, s.openrouter_api_key),
            name=f"openrouter ({model})",
        )
    if provider == "ollama":
        return LLMJsonEngine(
            build_adapter("ollama", s.oracle_url, model, s.oracle_timeout_s),
            name=f"ollama ({model})",
        )
    raise ValueError(f"unknown LLM provider {provider!r}; choose from openai, anthropic, gemini, openrouter, ollama")


def build_engine(kind: str, s: Settings) -> SystemOneEngine:
    if ":" in kind:
        prefix, target = kind.split(":", 1)
        prefix, target = prefix.strip(), target.strip()
        if prefix == "kenning" and target:
            return DirectKenningEngine(target)
        if prefix == "clef" and target:
            return DirectClefEngine(target)
        if prefix == "xl" and target:
            return DirectKenningXLEngine(target)
        if prefix == "llm" and "/" in target:
            prov, mod = target.split("/", 1)
            return _build_frontier_llm(prov.strip(), mod.strip(), s)
        if prefix == "llm" and target:
            o = resolve(s)["oracle"]
            return LLMJsonEngine(build_adapter(o.adapter, o.url, target, s.oracle_timeout_s, o.api_key()),
                                 name=f"llm ({target})")
        if prefix in ("openai", "anthropic", "gemini", "openrouter", "ollama") and target:
            return _build_frontier_llm(prefix, target, s)
    if kind == "clef-direct":
        return DirectClefEngine("Cloudflare/clef-flash")
    if kind == "xl":
        return DirectKenningXLEngine("Qwen/Qwen2.5-7B-Instruct")
    if kind == "clef":
        # Cloudflare Clef (Apache-2.0), served locally by kenning.clef_serve (compose profile "clef")
        return JevEngine(None, base_url=s.clef_url, model="clef-flash", timeout_s=max(s.request_timeout_s, 120),
                         require_key=False, name="clef-flash")
    if kind == "kenning":
        # Kenning (compose service "kenning") serves the System One wire format,
        # so the same HTTP client is reused, without a key
        return JevEngine(None, base_url=s.kenning_url, model="kenning", timeout_s=s.request_timeout_s,
                         require_key=False, name="kenning")
    if kind == "local":
        return LocalLogprobEngine(s.system_one_local_url or s.triage_url, s.system_one_local_model or s.triage_model,
                                  timeout_s=s.request_timeout_s, concurrency=s.system_one_concurrency)
    if kind == "jev":
        # Opt-in: TypeSafe Jev with the user's own TYPESAFE_API_KEY (their TypeSafe agreement applies).
        return JevEngine(s.typesafe_api_key, base_url=s.typesafe_url, model=s.typesafe_model,
                         timeout_s=s.request_timeout_s)
    if kind == "glide":
        # Opt-in: Fastino GLiDE with the user's own FASTINO_API_KEY. Same wire format; benchmark only.
        return JevEngine(s.fastino_api_key, base_url=s.fastino_url, model=s.fastino_model,
                         timeout_s=max(s.request_timeout_s, 60), name="glide")
    if kind == "llm":
        o = resolve(s)["oracle"]
        return LLMJsonEngine(build_adapter(o.adapter, o.url, o.model, s.oracle_timeout_s, o.api_key()))
    raise ValueError(f"unknown engine {kind!r}; choose from {', '.join(ENGINES)} (or <provider>:<model>)")


def api_engine(s: Settings) -> SystemOneEngine:
    """The engine behind POST /api/v1/systemone (S1_SYSTEM_ONE_BACKEND)."""
    kind = {"kenning": "kenning", "model": "kenning", "logprob": "local"}.get(s.system_one_backend, s.system_one_backend)
    return build_engine(kind, s)
