"""Replay-buffer record contract (Module A)."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Tier(str, Enum):
    STUDENT = "student"  # GPU 0 reflex model
    TRIAGE = "triage"  # GPU 1 synchronous fast-slow escalation
    ORACLE = "oracle"  # Mac M4 Max asynchronous teacher
    HUMAN = "human"  # DPO corrections studio


class Outcome(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    ESCALATED = "escalated"


class ReplayRecord(BaseModel):
    """One action taken by the system plus the state that produced it."""

    seq: int | None = None  # assigned by the buffer
    ts: float = Field(default_factory=time.time)
    domain: str
    session_id: str = "default"
    state: dict[str, Any]  # scrubbed state_input
    state_hash: str | None = None
    action: dict[str, Any] | None = None  # action_output
    confidence: float | None = None
    tier: Tier = Tier.STUDENT
    route_path: list[Tier] = Field(default_factory=list)
    latency_ms: float | None = None
    ttft_ms: float | None = None
    outcome: Outcome = Outcome.PENDING
    delta: dict[str, Any] | None = None  # state delta observed after the action
    meta: dict[str, Any] = Field(default_factory=dict)
