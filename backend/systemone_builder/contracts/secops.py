"""Cybersecurity Reflex Schema with actionable payloads (spec section 5.2)."""

from __future__ import annotations

import ipaddress
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SECOPS_ACTIONS = ["DROP_AND_BLACKLIST_IP", "ESCALATE", "ALLOW"]


class SecOpsState(BaseModel):
    # Extra log fields (dest_ip, signature, http fields, ...) are allowed.
    model_config = ConfigDict(extra="allow")

    source: str
    src_ip: str | None = None
    payload_snippet: str = ""


class SecOpsAction(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["BENIGN", "SUSPICIOUS", "MALICIOUS"]
    immediate_action: str
    target_ioc: list[str] = Field(default_factory=list)
    supported_actions: list[str] = Field(default_factory=lambda: list(SECOPS_ACTIONS))

    @model_validator(mode="after")
    def _check(self) -> "SecOpsAction":
        if self.immediate_action not in self.supported_actions:
            raise ValueError(f"immediate_action {self.immediate_action!r} not in supported_actions")
        if self.immediate_action == "DROP_AND_BLACKLIST_IP":
            if not self.target_ioc:
                raise ValueError("DROP_AND_BLACKLIST_IP requires target_ioc")
            for ioc in self.target_ioc:
                ipaddress.ip_address(ioc)  # raises ValueError on non-IP
        return self


def grounding_errors(state: dict[str, Any], action: dict[str, Any]) -> list[str]:
    """Every IOC must literally appear in the observed state (no invented IPs)."""
    blob = str(state)
    return [f"target_ioc {ioc} not observed in state" for ioc in action.get("target_ioc", []) if ioc not in blob]
