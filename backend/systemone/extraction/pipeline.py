"""State extraction pipeline: raw observation -> scrubbed ``state_input``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from systemone.domains.spec import DomainSpec
from systemone.extraction.dom import DomExtractor
from systemone.extraction.fuzzy import EntityVault, FuzzyScrubber, ScrubPolicy
from systemone.extraction.logs import LogExtractor
from systemone.extraction.prompt import PromptBuilder, canonical_state, state_hash

ObservationKind = Literal["state", "html", "ax_tree", "elements", "log", "suricata_eve", "syslog", "json", "text"]


class Observation(BaseModel):
    """What a client (browser driver, log shipper, IDS) sends for one step."""

    kind: ObservationKind = "state"
    data: Any
    url: str | None = None
    temporal_buffer: list[str] | None = None
    screenshot_b64: str | None = None
    viewport: tuple[int, int] | None = None  # (width, height) for DOM filtering
    goal: str | None = None  # task instruction (stable across a session)
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ExtractionResult:
    state: dict[str, Any]
    canonical: str
    state_hash: str
    index: dict[int, dict[str, Any]] = field(default_factory=dict)
    vault: EntityVault | None = None
    screenshot_b64: str | None = None
    validation_errors: list[str] = field(default_factory=list)


def policy_for(domain: DomainSpec) -> ScrubPolicy:
    e = domain.extractor
    return ScrubPolicy(
        drop_fields=set(e.drop_fields),
        preserve_fields=set(e.preserve_fields),
        normalize_counters=e.normalize_counters,
        tokenize_ips=e.tokenize_ips,
        tokenize_emails=e.tokenize_emails,
        max_string_len=e.max_string_len,
        extra_volatile_params=set(e.volatile_query_params),
        custom_patterns=list(e.custom_patterns),
    )


class StateExtractor:
    def __init__(self, domain: DomainSpec) -> None:
        self.domain = domain
        self.scrubber = FuzzyScrubber(policy_for(domain))
        self.dom = DomExtractor(self.scrubber, max_nodes=domain.extractor.max_nodes)
        self.logs = LogExtractor(self.scrubber, payload_max=domain.extractor.payload_max)
        self.prompts = PromptBuilder(domain)

    def extract(self, obs: Observation) -> ExtractionResult:
        vault = EntityVault() if (self.domain.extractor.tokenize_ips or self.domain.extractor.tokenize_emails) else None
        index: dict[int, dict[str, Any]] = {}
        kind = obs.kind
        if kind in ("html", "ax_tree", "elements"):
            self.dom.viewport = obs.viewport
            if kind == "html":
                ex = self.dom.from_html(str(obs.data), vault)
            elif kind == "ax_tree":
                ex = self.dom.from_ax_tree(obs.data, vault)
            else:
                ex = self.dom.from_elements(list(obs.data), vault)
            index = ex.index
            state: dict[str, Any] = {
                "url": self.scrubber.scrub_url(obs.url or "about:blank", vault),
                "viewport_tree": ex.viewport_tree,
                "temporal_buffer": list(obs.temporal_buffer or [])[-self.domain.extractor.temporal_buffer_len :],
            }
        elif kind in ("log", "suricata_eve", "syslog", "json", "text"):
            fmt = self.domain.extractor.log_format if kind == "log" else kind
            state = self.logs.extract(obs.data, fmt, vault)
        else:  # pre-structured state_input: scrub only
            if not isinstance(obs.data, dict):
                raise ValueError("observation kind 'state' requires a JSON object")
            state = self.scrubber.scrub(obs.data, vault)
            if obs.temporal_buffer is not None:
                state["temporal_buffer"] = list(obs.temporal_buffer)[-self.domain.extractor.temporal_buffer_len :]
            if self.domain.kind == "computer_use":
                state.setdefault("temporal_buffer", [])
        if obs.goal:
            state["goal"] = self.scrubber.scrub_text(obs.goal, vault)
        canon = canonical_state(state, self.domain.prompt_key_order)
        return ExtractionResult(
            state=state,
            canonical=canon,
            state_hash=state_hash(canon),
            index=index,
            vault=vault if vault and vault.reverse else None,
            screenshot_b64=obs.screenshot_b64,
            validation_errors=self.domain.validate_state(state),
        )
