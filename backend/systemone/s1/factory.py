"""Build System One engines from settings."""

from __future__ import annotations

from systemone.adapters.byom import resolve
from systemone.adapters.factory import build_adapter
from systemone.config import Settings
from systemone.s1.engines import JevEngine, LLMJsonEngine, LocalLogprobEngine, SystemOneEngine

ENGINES = ("local", "jev", "llm")


def build_engine(kind: str, s: Settings) -> SystemOneEngine:
    if kind == "local":
        return LocalLogprobEngine(s.system_one_local_url or s.triage_url, s.system_one_local_model or s.triage_model,
                                  timeout_s=s.request_timeout_s, concurrency=s.system_one_concurrency)
    if kind == "jev":
        return JevEngine(s.typesafe_api_key, base_url=s.typesafe_url, model=s.typesafe_model,
                         timeout_s=s.request_timeout_s)
    if kind == "llm":
        o = resolve(s)["oracle"]
        return LLMJsonEngine(build_adapter(o.adapter, o.url, o.model, s.oracle_timeout_s, o.api_key()))
    raise ValueError(f"unknown engine {kind!r}; choose from {', '.join(ENGINES)}")
