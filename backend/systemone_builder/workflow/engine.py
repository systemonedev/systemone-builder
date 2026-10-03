"""Prompt-to-Workflow Engine (Module F, Phase 6).

A user describes the specialization in natural language ("triage failed SSH
logins on my bastion hosts and block brute-forcers", "click through our
Jira-like tracker to close stale tickets"). The engine asks the Teacher to
produce a complete :class:`DomainSpec` - state extractor configuration,
JSON data schemas, action vocabulary, routing threshold, system prompt,
few-shot examples and synthetic-factory scenarios - validates it, repairs it
through a feedback loop, and returns a draft the user can review and activate.

Activation registers the domain and (optionally) bootstraps the workspace:
the synthetic factory starts generating seed trajectories immediately.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from pydantic import ValidationError

from systemone_builder.adapters.base import AdapterError, ModelAdapter
from systemone_builder.datastore.store import JsonStore
from systemone_builder.domains.builtin import BUILTIN_DOMAINS
from systemone_builder.domains.spec import DomainSpec
from systemone_builder.routing.parse import parse_action

log = logging.getLogger(__name__)

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "name", "description", "kind", "supported_actions", "action_field", "state_schema",
                 "action_schema", "threshold", "system_prompt", "few_shots", "prompt_key_order", "extractor", "factory"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "kind": {"type": "string", "enum": ["computer_use", "secops", "custom"]},
        "supported_actions": {"type": "array", "items": {"type": "string"}},
        "action_field": {"type": "string"},
        "state_schema": {"type": "object"},
        "action_schema": {"type": "object"},
        "threshold": {"type": "number"},
        "system_prompt": {"type": "string"},
        "few_shots": {"type": "array", "items": {"type": "object", "properties": {"state": {"type": "object"}, "action": {"type": "object"}}}},
        "prompt_key_order": {"type": "array", "items": {"type": "string"}},
        "extractor": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["dom", "log", "passthrough"]},
                "drop_fields": {"type": "array", "items": {"type": "string"}},
                "preserve_fields": {"type": "array", "items": {"type": "string"}},
                "volatile_query_params": {"type": "array", "items": {"type": "string"}},
                "custom_patterns": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                "tokenize_ips": {"type": "boolean"},
                "log_format": {"type": "string"},
            },
        },
        "factory": {
            "type": "object",
            "properties": {
                "scenarios": {"type": "array", "items": {"type": "string"}},
                "seed_states_per_scenario": {"type": "integer"},
                "temperature": {"type": "number"},
            },
        },
    },
}

GUI_HINTS = re.compile(r"\b(click|browser|web ?page|website|dom|gui|form|button|ui|screen|desktop|app|navigate|login page|tab)\b", re.I)
SEC_HINTS = re.compile(r"\b(log|ids|suricata|zeek|snort|siem|alert|attack|threat|malware|phish|brute|ssh|firewall|ioc|cve|soc|secops|intrusion|vuln|edr|waf)\b", re.I)


def classify(prompt: str) -> str:
    gui, sec = len(GUI_HINTS.findall(prompt)), len(SEC_HINTS.findall(prompt))
    if gui == sec == 0:
        return "custom"
    return "computer_use" if gui > sec else "secops"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    s = s if s and s[0].isalpha() else f"d_{s}"
    return s[:48] or f"domain_{uuid.uuid4().hex[:6]}"


class WorkflowEngine:
    def __init__(self, teacher: ModelAdapter, store: JsonStore, max_repairs: int = 2) -> None:
        self.teacher = teacher
        self.store = store
        self.max_repairs = max_repairs

    def _system_prompt(self, template: DomainSpec | None) -> str:
        base = (
            "You are the Prompt-to-Workflow engine of systemone-builder, a framework that distills slow System-2 "
            "reasoning into sub-100ms System-1 reflex models. From the user's description of a specialization, "
            "design a complete domain specification as JSON.\n\n"
            "Requirements:\n"
            "- id: snake_case, 2-64 chars, starts with a letter.\n"
            "- kind: computer_use (GUI/DOM automation), secops (security log/event triage) or custom.\n"
            "- supported_actions: UPPER_SNAKE_CASE verbs; MUST include ESCALATE (hand-off to the teacher).\n"
            "- action_field: the key of action_output holding the chosen action.\n"
            "- state_schema / action_schema: JSON Schema (draft 2020-12). action_schema MUST require "
            "confidence_score (number 0..1) and the action_field. Keep states compact; they are serialized into a "
            "latency-critical prompt.\n"
            "- threshold: routing confidence threshold, 0.6-0.99; higher for high-stakes/irreversible actions.\n"
            "- system_prompt: concise policy for the reflex model.\n"
            "- few_shots: 1-3 {state, action} examples valid against the schemas.\n"
            "- prompt_key_order: state keys ordered from most stable to most volatile (prefix-cache friendly).\n"
            "- extractor.type: dom for web GUIs, log for log/event streams, passthrough for JSON states. List "
            "volatile fields to drop (timestamps, request ids, counters) and regex custom_patterns "
            "[[pattern, placeholder]] for domain-specific volatile tokens.\n"
            "- factory.scenarios: 6-12 diverse training scenarios including edge cases and escalation cases.\n"
        )
        if template is not None:
            base += (
                "\nStart from this built-in starter template and customize it to the user's specialization. Keep its "
                "state_schema/action_schema contract compatible (you may add optional fields and actions):\n"
                + json.dumps(template.model_dump(mode="json", exclude={"source"}))
            )
        return base

    async def generate(self, prompt: str, kind_hint: str | None = None) -> dict[str, Any]:
        kind = kind_hint or classify(prompt)
        template = BUILTIN_DOMAINS.get(kind)
        messages = [
            {"role": "system", "content": self._system_prompt(template)},
            {"role": "user", "content": f"Specialization:\n{prompt}"},
        ]
        attempts: list[dict[str, Any]] = []
        spec: DomainSpec | None = None
        for attempt in range(self.max_repairs + 1):
            gen = await self.teacher.generate(messages, json_schema=DRAFT_SCHEMA, max_tokens=6000, temperature=0.3)
            raw, _ = parse_action(gen.text)
            errors = self._validate(raw, template)
            attempts.append({"attempt": attempt, "errors": errors})
            if not errors:
                spec = DomainSpec.model_validate({**self._normalize(raw, template), "source": "generated"})
                break
            messages += [
                {"role": "assistant", "content": gen.text[-8000:]},
                {"role": "user", "content": "The specification is invalid:\n- " + "\n- ".join(errors[:20]) + "\nReturn the corrected full JSON."},
            ]
        draft_id = uuid.uuid4().hex[:12]
        draft = {
            "id": draft_id,
            "prompt": prompt,
            "kind": kind,
            "template": template.id if template else None,
            "status": "ready" if spec else "invalid",
            "spec": spec.model_dump(mode="json") if spec else None,
            "attempts": attempts,
            "created_at": time.time(),
        }
        await self.store.hset(("workflow", "drafts"), draft_id, draft)
        return draft

    def _normalize(self, raw: dict[str, Any], template: DomainSpec | None) -> dict[str, Any]:
        d = dict(raw)
        d["id"] = slugify(str(d.get("id") or d.get("name") or "domain"))
        acts = [str(a).upper() for a in d.get("supported_actions") or []]
        if "ESCALATE" not in acts:
            acts.append("ESCALATE")
        d["supported_actions"] = acts
        d["threshold"] = float(min(max(float(d.get("threshold", 0.8)), 0.5), 0.99))
        if template is not None and d.get("kind") == template.kind:
            # preserve the template contract so built-in validators/graders apply
            d["action_field"] = template.action_field
            d["extractor"] = {**template.extractor.model_dump(), **(d.get("extractor") or {}), "type": template.extractor.type}
        ext = d.get("extractor") or {}
        ext["custom_patterns"] = [p for p in ext.get("custom_patterns") or [] if isinstance(p, list) and len(p) == 2]
        d["extractor"] = ext
        return d

    def _validate(self, raw: Any, template: DomainSpec | None) -> list[str]:
        if not isinstance(raw, dict):
            return ["output is not a JSON object"]
        try:
            spec = DomainSpec.model_validate({**self._normalize(raw, template), "source": "generated"})
        except (ValidationError, ValueError, TypeError) as exc:
            if isinstance(exc, ValidationError):
                return [f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            return [str(exc)]
        errors: list[str] = []
        req = set(spec.action_schema.get("required") or [])
        if "confidence_score" not in req:
            errors.append("action_schema must require confidence_score")
        if spec.action_field not in (spec.action_schema.get("properties") or {}):
            errors.append(f"action_schema must define the action_field {spec.action_field!r}")
        for i, shot in enumerate(spec.few_shots):
            if not isinstance(shot, dict) or "state" not in shot or "action" not in shot:
                errors.append(f"few_shots[{i}] must have state and action")
                continue
            errors += [f"few_shots[{i}].state {e}" for e in spec.validate_state(shot["state"])]
            v = spec.validate_action(shot["action"], shot["state"])
            errors += [f"few_shots[{i}].action {e}" for e in v.errors]
        if len(spec.factory.scenarios) < 3:
            errors.append("factory.scenarios needs at least 3 scenarios")
        return errors

    async def drafts(self) -> list[dict[str, Any]]:
        items = list((await self.store.hgetall(("workflow", "drafts"))).values())
        items.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return items

    async def get(self, draft_id: str) -> dict[str, Any] | None:
        return await self.store.hget(("workflow", "drafts"), draft_id)


async def generate_safe(engine: WorkflowEngine, prompt: str, kind_hint: str | None) -> dict[str, Any]:
    try:
        return await engine.generate(prompt, kind_hint)
    except AdapterError as exc:
        return {"status": "error", "error": str(exc), "prompt": prompt}
