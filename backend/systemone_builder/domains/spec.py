"""Domain specification: everything needed to distill one specialization.

A :class:`DomainSpec` is the unit produced by the Prompt-to-Workflow engine
and shipped as a starter template. It bundles the data contract (JSON
schemas), the supported action vocabulary, the state extractor configuration,
the prompt, the routing threshold and the synthetic-factory parameters.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, Field, ValidationError, field_validator

from systemone_builder.contracts import computer_use, secops

ExtractorType = Literal["dom", "log", "passthrough"]


class ExtractorConfig(BaseModel):
    type: ExtractorType = "passthrough"
    # FuzzyScrubber policy overrides
    drop_fields: list[str] = Field(default_factory=list)
    preserve_fields: list[str] = Field(default_factory=list)
    volatile_query_params: list[str] = Field(default_factory=list)
    custom_patterns: list[tuple[str, str]] = Field(default_factory=list)
    tokenize_ips: bool = False
    tokenize_emails: bool = True
    normalize_counters: bool = True
    max_string_len: int = 512
    # DOM specifics
    max_nodes: int = 150
    temporal_buffer_len: int = 8
    # log specifics
    log_format: str = "auto"
    payload_max: int = 512

    @field_validator("custom_patterns")
    @classmethod
    def _valid_regex(cls, v: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for pattern, _ in v:
            re.compile(pattern)
        return v


class FactoryConfig(BaseModel):
    """Synthetic data factory parameters (Module B)."""

    scenarios: list[str] = Field(default_factory=list)
    seed_states_per_scenario: int = 20
    temperature: float = 0.7
    cot_max_tokens: int = 1024
    judge_min_score: float = 0.7
    # Fraction of confident student actions sampled for oracle re-labelling.
    replay_sample_rate: float = 0.05


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    grounding_errors: list[str] = Field(default_factory=list)
    action: dict[str, Any] | None = None

    @property
    def hallucinated(self) -> bool:
        return bool(self.grounding_errors)


class DomainSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str
    description: str = ""
    kind: Literal["computer_use", "secops", "custom"] = "custom"
    state_schema: dict[str, Any]
    action_schema: dict[str, Any]
    supported_actions: list[str]
    # Field of action_output carrying the chosen action.
    action_field: str = "action"
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    # Acceptance threshold for the GPU 1 triage tier (defaults to ``threshold``).
    triage_threshold: float | None = Field(None, ge=0.0, le=1.0)
    triage_enabled: bool = True
    system_prompt: str
    few_shots: list[dict[str, Any]] = Field(default_factory=list)
    # Serialization order of top-level state keys: stable keys first so that
    # consecutive prompts share the longest possible prefix.
    prompt_key_order: list[str] = Field(default_factory=list)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    factory: FactoryConfig = Field(default_factory=FactoryConfig)
    source: Literal["builtin", "template", "generated", "user"] = "user"

    @field_validator("state_schema", "action_schema")
    @classmethod
    def _valid_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        jsonschema.Draft202012Validator.check_schema(v)
        return v

    # ------------------------------------------------------------ validation
    def validate_state(self, state: dict[str, Any]) -> list[str]:
        v = jsonschema.Draft202012Validator(self.state_schema)
        return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in v.iter_errors(state)]

    def validate_action(self, action: Any, state: dict[str, Any] | None = None) -> ValidationResult:
        if not isinstance(action, dict):
            return ValidationResult(ok=False, errors=["action_output is not a JSON object"])
        action = {**action}
        action.setdefault("supported_actions", list(self.supported_actions))
        errors = [
            f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
            for e in jsonschema.Draft202012Validator(self.action_schema).iter_errors(action)
        ]
        chosen = action.get(self.action_field)
        if chosen not in self.supported_actions:
            errors.append(f"{self.action_field} {chosen!r} not in supported_actions {self.supported_actions}")
        grounding: list[str] = []
        try:
            if self.kind == "computer_use":
                action = computer_use.ComputerUseAction.model_validate(action).model_dump(exclude_none=True)
                if state is not None:
                    grounding = computer_use.grounding_errors(state, action)
            elif self.kind == "secops":
                action = secops.SecOpsAction.model_validate(action).model_dump()
                if state is not None:
                    grounding = secops.grounding_errors(state, action)
        except ValidationError as exc:
            errors.extend(f"{'/'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())
        return ValidationResult(ok=not errors, errors=errors, grounding_errors=grounding, action=action)

    def guided_json_schema(self) -> dict[str, Any]:
        """Schema handed to vLLM guided decoding / Ollama ``format``."""
        schema = dict(self.action_schema)
        props = dict(schema.get("properties", {}))
        props[self.action_field] = {**props.get(self.action_field, {"type": "string"}), "enum": list(self.supported_actions)}
        schema["properties"] = props
        return schema

    def action_label(self, action: dict[str, Any] | None) -> str | None:
        return None if not action else action.get(self.action_field)

    @property
    def effective_triage_threshold(self) -> float:
        return self.threshold if self.triage_threshold is None else self.triage_threshold

    def escalate_action(self, confidence: float, hint: dict[str, Any] | None = None) -> dict[str, Any]:
        """The halt action returned while a query is escalated."""
        if self.kind == "computer_use":
            return {"confidence_score": confidence, "action": "ESCALATE", "target_id": None, "coordinates": None,
                    "supported_actions": list(self.supported_actions)}
        if self.kind == "secops":
            verdict = (hint or {}).get("verdict") if (hint or {}).get("verdict") in ("BENIGN", "SUSPICIOUS", "MALICIOUS") else "SUSPICIOUS"
            return {"confidence_score": confidence, "verdict": verdict, "immediate_action": "ESCALATE",
                    "target_ioc": [], "supported_actions": list(self.supported_actions)}
        return {"confidence_score": confidence, self.action_field: "ESCALATE", "supported_actions": list(self.supported_actions)}
