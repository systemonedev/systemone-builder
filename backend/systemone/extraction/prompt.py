"""Prefix-stable prompt construction.

Layout (most stable -> most volatile):

1. system message: domain instructions, output contract, action vocabulary
2. few-shot exchanges (static per domain)
3. user message: canonical state JSON with keys in ``prompt_key_order``

Canonical JSON (fixed key order, compact separators, no NaN) guarantees that
two semantically identical states render to byte-identical prompts, which is
what vLLM's automatic prefix caching needs.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from systemone.domains.spec import DomainSpec


def _sort(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort(v) for v in obj]
    return obj


def canonical_state(state: dict[str, Any], key_order: list[str]) -> str:
    ordered: dict[str, Any] = {}
    for k in key_order:
        if k in state:
            ordered[k] = _sort(state[k])
    for k in sorted(state):
        if k not in ordered:
            ordered[k] = _sort(state[k])
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def state_hash(canonical: str) -> str:
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def format_action(action: dict[str, Any] | None, action_field: str = "action") -> str:
    """Compact temporal-buffer form: ``CLICK(12)``, ``TYPE(2, 'admin')``."""
    if not action:
        return "NOOP()"
    name = action.get(action_field) or action.get("action") or "UNKNOWN"
    args: list[str] = []
    if action.get("target_id") is not None:
        args.append(str(action["target_id"]))
    if action.get("text") is not None:
        args.append(repr(action["text"]))
    if action.get("coordinates"):
        c = action["coordinates"]
        args.extend([str(c.get("x")), str(c.get("y"))])
    if action.get("target_ioc"):
        args.append(",".join(action["target_ioc"]))
    return f"{name}({', '.join(args)})"


class PromptBuilder:
    def __init__(self, domain: DomainSpec) -> None:
        self.domain = domain
        self._static = self._build_static()
        self.static_prefix_hash = hashlib.sha256(json.dumps(self._static).encode()).hexdigest()[:16]

    def _contract(self) -> str:
        d = self.domain
        schema = json.dumps(d.guided_json_schema(), separators=(",", ":"), sort_keys=True)
        return (
            f"{d.system_prompt}\n\n"
            f"Supported actions: {', '.join(d.supported_actions)}.\n"
            f"Respond with a single JSON object (action_output) matching this JSON schema and nothing else:\n{schema}"
        )

    def _build_static(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self._contract()}]
        for shot in self.domain.few_shots:
            msgs.append({"role": "user", "content": canonical_state(shot["state"], self.domain.prompt_key_order)})
            msgs.append({"role": "assistant", "content": json.dumps(_sort(shot["action"]), separators=(",", ":"))})
        return msgs

    def messages(self, state: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        canon = canonical_state(state, self.domain.prompt_key_order)
        return [*self._static, {"role": "user", "content": canon}], canon

    def teacher_messages(self, state: dict[str, Any], extra: str | None = None) -> list[dict[str, str]]:
        """System-2 prompt: think step by step, then emit the same contract."""
        d = self.domain
        sys = (
            f"{self._contract()}\n\nYou are the System-2 teacher. First reason step by step inside "
            "<think>...</think> about the state, the goal and the risks. Then output the final action_output "
            "JSON object on its own after the closing </think> tag."
        )
        user = canonical_state(state, d.prompt_key_order)
        if extra:
            user = f"{user}\n\n{extra}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


class PrefixTracker:
    """Tracks prompt-prefix overlap between consecutive requests per session."""

    def __init__(self, max_sessions: int = 1024) -> None:
        self._last: OrderedDict[str, str] = OrderedDict()
        self.max_sessions = max_sessions

    def observe(self, session_id: str, prompt: str) -> float:
        prev = self._last.pop(session_id, None)
        self._last[session_id] = prompt
        while len(self._last) > self.max_sessions:
            self._last.popitem(last=False)
        if not prev:
            return 0.0
        n = min(len(prev), len(prompt))
        i = 0
        while i < n and prev[i] == prompt[i]:
            i += 1
        return i / max(len(prompt), 1)
