"""Build System One engines from settings."""

from __future__ import annotations

from systemone_builder.adapters.byom import resolve
from systemone_builder.adapters.factory import build_adapter
from systemone_builder.config import Settings
from systemone_builder.system_one.engines import JevEngine, LLMJsonEngine, LocalLogprobEngine, SystemOneEngine

ENGINES = ("kenning", "jev", "local", "llm")


def build_engine(kind: str, s: Settings) -> SystemOneEngine:
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
    if kind == "llm":
        o = resolve(s)["oracle"]
        return LLMJsonEngine(build_adapter(o.adapter, o.url, o.model, s.oracle_timeout_s, o.api_key()))
    raise ValueError(f"unknown engine {kind!r}; choose from {', '.join(ENGINES)}")


def api_engine(s: Settings) -> SystemOneEngine:
    """The engine behind POST /api/v1/systemone (S1_SYSTEM_ONE_BACKEND)."""
    kind = {"kenning": "kenning", "model": "kenning", "logprob": "local"}.get(s.system_one_backend, s.system_one_backend)
    return build_engine(kind, s)
