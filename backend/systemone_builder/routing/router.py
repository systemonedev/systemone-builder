"""Fast-Slow router (Module D, Phase 3).

::

    observation --extract/scrub--> state_input
         |
         v
    [Student  GPU 0]  conf >= threshold ------------------------------> execute
         | halt (low confidence / invalid / ESCALATE)
         v
    [Triage   GPU 1]  conf >= triage_threshold -----------------------> execute
         | halt                                                         (label -> SFT)
         v
    [Oracle   Mac  ]  async queue; caller gets ESCALATE (or waits) ---> label -> SFT

Every decision is written to the replay buffer with its full route so the
synthetic factory can distill teacher answers back into the student, and is
published on the event bus for the routing visualizer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from systemone_builder.adapters.base import AdapterError, Generation, ModelAdapter
from systemone_builder.contracts.replay import Outcome, ReplayRecord, Tier
from systemone_builder.datastore.replay_buffer import ReplayBuffer
from systemone_builder.datastore.store import JsonStore
from systemone_builder.domains.spec import DomainSpec, ValidationResult
from systemone_builder.extraction.pipeline import ExtractionResult, Observation, StateExtractor
from systemone_builder.extraction.prompt import PrefixTracker, format_action
from systemone_builder.routing.confidence import ConfidenceReport, ConfidenceScorer
from systemone_builder.routing.oracle import OracleEscalationService
from systemone_builder.routing.parse import parse_action
from systemone_builder.telemetry.bus import EventBus

log = logging.getLogger(__name__)


@dataclass
class TierOutcome:
    tier: Tier
    accepted: bool
    threshold: float
    action: dict[str, Any] | None = None
    report: ConfidenceReport | None = None
    validation: ValidationResult | None = None
    generation: Generation | None = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        g = self.generation
        return {
            "tier": self.tier.value,
            "accepted": self.accepted,
            "threshold": self.threshold,
            "confidence": self.report.confidence if self.report else 0.0,
            "self_reported": self.report.self_reported if self.report else None,
            "token_confidence": self.report.token_confidence if self.report else None,
            "gates": self.report.gates if self.report else [],
            "errors": (self.validation.errors + self.validation.grounding_errors) if self.validation else [],
            "ttft_ms": g.ttft_ms if g else None,
            "latency_ms": g.latency_ms if g else None,
            "prompt_tokens": g.prompt_tokens if g else None,
            "cached_tokens": g.cached_tokens if g else None,
            "model": g.model if g else None,
            "error": self.error,
        }


class Decision(BaseModel):
    decision_id: str
    domain: str
    session_id: str
    seq: int | None
    tier: Tier
    halted: bool
    action: dict[str, Any]
    confidence: float
    threshold: float
    route: list[dict[str, Any]]
    escalation_id: str | None = None
    oracle: dict[str, Any] | None = None
    state_hash: str
    prefix_overlap: float
    execution: dict[str, Any] | None = None
    latency_ms: float
    state_input: dict[str, Any] = Field(default_factory=dict)


class FastSlowRouter:
    SESSION_HISTORY = 32

    def __init__(
        self,
        *,
        student: ModelAdapter,
        triage: ModelAdapter | None,
        oracle: OracleEscalationService,
        scorer: ConfidenceScorer,
        replay: ReplayBuffer,
        store: JsonStore,
        bus: EventBus,
        prefix: PrefixTracker,
        extractor_for: Any,
        domain_for: Any,
        student_model_name: Any,
        student_max_tokens: int = 192,
        student_gate: Any = None,
        vision: Any = None,
    ) -> None:
        self.student = student
        self.triage = triage
        self.oracle = oracle
        self.scorer = scorer
        self.replay = replay
        self.store = store
        self.bus = bus
        self.prefix = prefix
        self._extractor_for = extractor_for
        self._domain_for = domain_for
        self._student_model_name = student_model_name
        self.student_max_tokens = student_max_tokens
        # StudentLifecycleOrchestrator: acquire()/release() around GPU 0 calls
        self.student_gate = student_gate
        self.vision = vision

    # ----------------------------------------------------------- sessions
    async def session_history(self, domain_id: str, session_id: str) -> list[str]:
        return list(await self.store.peek(("session", domain_id, session_id, "actions"), -self.SESSION_HISTORY, -1))

    async def _push_history(self, domain_id: str, session_id: str, label: str) -> None:
        key = self.store.key("session", domain_id, session_id, "actions")
        pipe = self.store.r.pipeline()
        pipe.rpush(key, json.dumps(label))
        pipe.ltrim(key, -self.SESSION_HISTORY, -1)
        pipe.expire(key, 86_400)
        await pipe.execute()

    async def reset_session(self, domain_id: str, session_id: str) -> None:
        await self.store.delete("session", domain_id, session_id, "actions")

    # ------------------------------------------------------------- tiers
    async def _run_tier(
        self, tier: Tier, adapter: ModelAdapter, domain: DomainSpec, messages: list[dict[str, str]],
        state: dict[str, Any], threshold: float, model: str | None = None,
    ) -> TierOutcome:
        try:
            gen = await adapter.generate(
                messages,
                json_schema=domain.guided_json_schema(),
                max_tokens=self.student_max_tokens,
                temperature=0.0,
                logprobs=True,
                model=model,
            )
        except AdapterError as exc:
            log.warning("%s tier unavailable: %s", tier.value, exc)
            return TierOutcome(tier, False, threshold, error=str(exc),
                               report=ConfidenceReport(0.0, None, None, gates=["tier_unavailable"]))
        action, _ = parse_action(gen.text)
        validation = domain.validate_action(action, state) if action is not None else None
        report = self.scorer.score(domain, gen.text, action, validation, gen.logprobs)
        normalized = validation.action if validation and validation.action else action
        if normalized is not None:
            # The router's calibrated score replaces the raw self-report.
            normalized["confidence_score"] = report.confidence
        accepted = report.confidence >= threshold and not report.gates
        return TierOutcome(tier, accepted, threshold, normalized, report, validation, gen)

    # -------------------------------------------------------------- main
    async def act(
        self,
        domain_id: str,
        obs: Observation,
        *,
        session_id: str = "default",
        wait_for_oracle: bool = False,
        oracle_timeout_s: float = 120.0,
    ) -> Decision:
        t0 = time.perf_counter()
        decision_id = uuid.uuid4().hex[:12]
        domain: DomainSpec = self._domain_for(domain_id)
        extractor: StateExtractor = self._extractor_for(domain_id)

        if obs.kind == "screenshot":
            obs = await self.screenshot_to_elements(obs)
        if obs.temporal_buffer is None and domain.kind == "computer_use":
            obs = obs.model_copy(update={"temporal_buffer": await self.session_history(domain_id, session_id)})
        ex: ExtractionResult = extractor.extract(obs)
        messages, _ = extractor.prompts.messages(ex.state)
        overlap = self.prefix.observe(f"{domain_id}:{session_id}", "".join(m["content"] for m in messages))
        self.bus.publish("routing", "request", decision_id=decision_id, domain=domain_id, session_id=session_id,
                         state_hash=ex.state_hash, prefix_overlap=overlap)

        route: list[TierOutcome] = []
        if self.student_gate is None or self.student_gate.acquire():
            try:
                student_model = await self._student_model_name()
                s = await self._run_tier(Tier.STUDENT, self.student, domain, messages, ex.state, domain.threshold, student_model)
            finally:
                if self.student_gate is not None:
                    self.student_gate.release()
        else:  # GPU 0 is in a training cycle: straight to triage
            s = TierOutcome(Tier.STUDENT, False, domain.threshold, error="student paused for training",
                            report=ConfidenceReport(0.0, None, None, gates=["student_training"]))
        route.append(s)
        self.bus.publish("routing", "tier_result", decision_id=decision_id, domain=domain_id, **s.summary())
        final: TierOutcome | None = s if s.accepted else None

        if final is None and self.triage is not None and domain.triage_enabled:
            self.bus.publish("routing", "escalate", decision_id=decision_id, domain=domain_id, frm="student", to="triage",
                             confidence=s.report.confidence if s.report else 0.0)
            t = await self._run_tier(Tier.TRIAGE, self.triage, domain, messages, ex.state, domain.effective_triage_threshold)
            route.append(t)
            self.bus.publish("routing", "tier_result", decision_id=decision_id, domain=domain_id, **t.summary())
            if t.accepted:
                final = t

        escalation_id: str | None = None
        oracle_result: dict[str, Any] | None = None
        best_conf = max((o.report.confidence for o in route if o.report), default=0.0)
        if final is not None:
            tier, action, halted = final.tier, dict(final.action or {}), False
        else:
            tier, halted = Tier.ORACLE, True
            hint = next((o.action for o in reversed(route) if o.action), None)
            action = domain.escalate_action(best_conf, hint)

        # Record first so the oracle label can be attached to this seq.
        rec = await self.replay.append(
            ReplayRecord(
                domain=domain_id,
                session_id=session_id,
                state=ex.state,
                state_hash=ex.state_hash,
                action=action,
                confidence=action.get("confidence_score", best_conf),
                tier=tier,
                route_path=[o.tier for o in route] + ([Tier.ORACLE] if halted else []),
                latency_ms=(time.perf_counter() - t0) * 1000,
                ttft_ms=s.generation.ttft_ms if s.generation else None,
                outcome=Outcome.ESCALATED if halted else Outcome.PENDING,
                meta={
                    "decision_id": decision_id,
                    "student_action": s.action,
                    "student_confidence": s.report.confidence if s.report else 0.0,
                    "label_source": tier.value if tier != Tier.STUDENT else None,
                    "entities": ex.vault.to_dict() if ex.vault else {},
                    "prefix_overlap": overlap,
                    "cached_tokens": s.generation.cached_tokens if s.generation else None,
                    "prompt_tokens": s.generation.prompt_tokens if s.generation else None,
                },
            )
        )

        if halted:
            reasons = sorted({g for o in route for g in (o.report.gates if o.report else [])} or {"low_confidence"})
            escalation_id = await self.oracle.submit(
                domain_id, ex.state, seq=rec.seq, reason=reasons,
                student_action=s.action, triage_action=route[1].action if len(route) > 1 else None,
                screenshot_b64=ex.screenshot_b64,
            )
            await self.replay.update(rec.seq, meta={**rec.meta, "escalation_id": escalation_id})
            if wait_for_oracle:
                oracle_result = await self.oracle.wait_for(escalation_id, oracle_timeout_s)
                if oracle_result and oracle_result.get("status") == "done" and oracle_result.get("action"):
                    action, halted = dict(oracle_result["action"]), False
                    await self.replay.update(rec.seq, action=action, outcome=Outcome.PENDING,
                                             confidence=action.get("confidence_score"))

        await self._push_history(domain_id, session_id, format_action(action, domain.action_field))
        executed = ex.vault.rehydrate(action) if ex.vault else action
        execution = self._execution_handle(executed, ex)
        latency = (time.perf_counter() - t0) * 1000
        decision = Decision(
            decision_id=decision_id, domain=domain_id, session_id=session_id, seq=rec.seq, tier=tier,
            halted=halted, action=executed, confidence=float(action.get("confidence_score", best_conf) or 0.0),
            threshold=domain.threshold, route=[o.summary() for o in route], escalation_id=escalation_id,
            oracle=oracle_result, state_hash=ex.state_hash, prefix_overlap=overlap, execution=execution,
            latency_ms=latency, state_input=ex.state,
        )
        await self._count(domain_id, decision)
        self.bus.publish("routing", "decision", decision_id=decision_id, domain=domain_id, seq=rec.seq, tier=tier.value,
                         halted=halted, confidence=decision.confidence, latency_ms=latency,
                         action_label=domain.action_label(executed), route=[o.summary() for o in route])
        return decision

    async def screenshot_to_elements(self, obs: Observation) -> Observation:
        if self.vision is None:
            raise ValueError("no vision parser configured (S1_ORACLE_VISION_MODEL)")
        if not obs.screenshot_b64 and not isinstance(obs.data, str):
            raise ValueError("screenshot observation requires screenshot_b64 or base64 data")
        shot = obs.screenshot_b64 or obs.data
        elements = await self.vision.parse(shot, obs.viewport)
        return obs.model_copy(update={"kind": "elements", "data": elements, "screenshot_b64": shot})

    @staticmethod
    def _execution_handle(action: dict[str, Any], ex: ExtractionResult) -> dict[str, Any] | None:
        tid = action.get("target_id")
        if tid is not None and tid in ex.index:
            handle = dict(ex.index[tid])
            bbox = handle.get("bbox")
            if bbox:
                handle["center"] = {"x": bbox[0] + bbox[2] // 2, "y": bbox[1] + bbox[3] // 2}
            return handle
        if action.get("coordinates"):
            return {"center": action["coordinates"]}
        return None

    async def _count(self, domain_id: str, d: Decision) -> None:
        pipe = self.store.r.pipeline()
        pipe.hincrby(self.store.key("routing", "stats", domain_id), "total", 1)
        pipe.hincrby(self.store.key("routing", "stats", domain_id), f"final_{d.tier.value}", 1)
        if d.halted:
            pipe.hincrby(self.store.key("routing", "stats", domain_id), "halted", 1)
        if len(d.route) > 1 or d.halted:
            pipe.hincrby(self.store.key("routing", "stats", domain_id), "escalations", 1)
        await pipe.execute()

    async def stats(self, domain_id: str) -> dict[str, int]:
        raw = await self.store.r.hgetall(self.store.key("routing", "stats", domain_id))
        return {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}
