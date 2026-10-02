"""Contextual DPO feedback loop (Module E, Phase 5).

Flow for every executed action whose outcome is reported::

    POST /feedback {seq, outcome?, post_observation?}
        -> extract post-state with the domain extractor
        -> compute the state delta (error banners, new alerts, no-op ...)
        -> infer failure if the client did not say
        -> update the replay record (outcome + delta)
        -> failure? create a correction candidate and send
           (initial state + failed action + failure delta) to the Teacher
        -> teacher proposes the corrected ("chosen") action with CoT
        -> candidate waits in the DPO Corrections Studio for a human, or is
           auto-approved when the judge is confident enough
        -> approved: (state, chosen, rejected) preference pair -> dpo.jsonl
           and the chosen action is also added as an SFT sample
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from systemone_builder.adapters.base import AdapterError
from systemone_builder.contracts.replay import Outcome
from systemone_builder.datastore.replay_buffer import ReplayBuffer
from systemone_builder.datastore.store import JsonStore
from systemone_builder.domains.registry import DomainRegistry
from systemone_builder.dpo.delta import compute_delta, infer_failure
from systemone_builder.extraction.pipeline import Observation
from systemone_builder.factory.dataset import DatasetStore, DPOPair, SFTSample
from systemone_builder.factory.synthetic import SyntheticFactory
from systemone_builder.telemetry.bus import EventBus

log = logging.getLogger(__name__)

CandidateStatus = Literal["pending_teacher", "pending_review", "approved", "rejected", "teacher_failed"]


class Feedback(BaseModel):
    seq: int | None = None
    decision_id: str | None = None
    outcome: Literal["success", "failure"] | None = None
    post_observation: Observation | None = None
    error_message: str | None = None  # client-side execution error (e.g. element detached)
    note: str | None = None


class Correction(BaseModel):
    """Human decision in the DPO Corrections Studio."""

    approve: bool = True
    chosen: dict[str, Any] | None = None  # edited action (overrides the teacher's)
    reviewer: str | None = None
    note: str | None = None


class DPOLoop:
    def __init__(
        self,
        replay: ReplayBuffer,
        store: JsonStore,
        datasets: DatasetStore,
        domains: DomainRegistry,
        factory: SyntheticFactory,
        bus: EventBus,
        extractor_for: Any,
        auto_approve_min_judge: float | None = 0.9,
        on_new_pairs: Any = None,
    ) -> None:
        self.replay = replay
        self.store = store
        self.datasets = datasets
        self.domains = domains
        self.factory = factory
        self.bus = bus
        self.extractor_for = extractor_for
        self.auto_approve_min_judge = auto_approve_min_judge
        self.on_new_pairs = on_new_pairs
        self._tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------ feedback
    async def _resolve_seq(self, fb: Feedback) -> int:
        if fb.seq is not None:
            return fb.seq
        if fb.decision_id:
            for rec in await self.replay.latest(self.replay.capacity):
                if rec.meta.get("decision_id") == fb.decision_id:
                    return rec.seq  # type: ignore[return-value]
        raise KeyError("feedback must reference a retained replay seq or decision_id")

    async def feedback(self, fb: Feedback) -> dict[str, Any]:
        seq = await self._resolve_seq(fb)
        rec = await self.replay.get(seq)
        if rec is None:
            raise KeyError(f"seq {seq} is no longer in the replay window")
        domain = self.domains.get(rec.domain)
        delta: dict[str, Any] | None = None
        if fb.post_observation is not None:
            post = self.extractor_for(rec.domain).extract(fb.post_observation).state
            delta = compute_delta(domain.kind, rec.state, post)
            delta["post_state"] = post
        if fb.error_message:
            delta = {**(delta or {}), "error_signals": [*(delta or {}).get("error_signals", []), f"execution error: {fb.error_message}"]}
        if fb.outcome is not None:
            failed = fb.outcome == "failure"
        else:
            failed = infer_failure(domain.kind, delta or {})
        outcome = Outcome.FAILURE if failed else Outcome.SUCCESS
        await self.replay.update(seq, outcome=outcome, delta=delta, meta={**rec.meta, "feedback_note": fb.note})
        self.bus.publish("dpo", "feedback", seq=seq, domain=rec.domain, outcome=outcome.value,
                         error_signals=(delta or {}).get("error_signals", []))
        await self.store.incr("dpo", rec.domain, outcome.value)
        candidate_id = None
        if failed and rec.action and domain.action_label(rec.action) != "ESCALATE":
            candidate_id = await self._create_candidate(rec.domain, seq, rec.state, rec.action, delta, fb.note)
        return {"seq": seq, "outcome": outcome.value, "delta": delta, "candidate_id": candidate_id}

    # ---------------------------------------------------------- candidates
    async def _create_candidate(self, domain_id: str, seq: int, state: dict[str, Any], rejected: dict[str, Any],
                                delta: dict[str, Any] | None, note: str | None) -> str:
        cid = uuid.uuid4().hex[:12]
        cand = {
            "id": cid, "domain": domain_id, "seq": seq, "state": state, "rejected": rejected,
            "delta": {k: v for k, v in (delta or {}).items() if k != "post_state"},
            "post_state": (delta or {}).get("post_state"), "note": note, "status": "pending_teacher",
            "created_at": time.time(),
        }
        await self.store.hset(("dpo", "candidates"), cid, cand)
        self.bus.publish("dpo", "candidate", id=cid, domain=domain_id, seq=seq, status="pending_teacher")
        task = asyncio.create_task(self._ask_teacher(cid))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return cid

    async def _ask_teacher(self, cid: str) -> None:
        cand = await self.store.hget(("dpo", "candidates"), cid)
        domain = self.domains.get(cand["domain"])
        hint = (
            "STATE-DELTA INJECTION: the reflex model executed the action below and it FAILED.\n"
            f"Failed action: {json.dumps(cand['rejected'])}\n"
            f"Immediate consequence (state delta): {json.dumps(cand['delta'])}\n"
            + (f"Operator note: {cand['note']}\n" if cand.get("note") else "")
            + "Explain why it failed and give the action that should have been taken from the ORIGINAL state."
        )
        try:
            chosen, cot, errs = await self.factory.teach(domain, cand["state"], hint)
        except AdapterError as exc:
            chosen, cot, errs = None, None, [str(exc)]
        if chosen is not None and _same_action(domain.action_field, chosen, cand["rejected"]):
            chosen, errs = None, ["teacher repeated the failed action"]
        cand.update(chosen=chosen, cot=cot, teacher_errors=errs)
        if chosen is None:
            cand["status"] = "teacher_failed"
        else:
            cand["status"] = "pending_review"
            if self.factory.judge is not None and self.auto_approve_min_judge is not None:
                try:
                    verdict = await self.factory.judge.review(domain, cand["state"], chosen, cot)
                    cand["judge"] = verdict
                except AdapterError as exc:
                    cand["judge"] = {"score": 0.0, "correct": False, "issues": [str(exc)]}
        await self.store.hset(("dpo", "candidates"), cid, cand)
        self.bus.publish("dpo", "candidate", id=cid, domain=cand["domain"], seq=cand["seq"], status=cand["status"])
        judge = cand.get("judge") or {}
        if (cand["status"] == "pending_review" and self.auto_approve_min_judge is not None
                and judge.get("correct") and judge.get("score", 0) >= self.auto_approve_min_judge):
            await self.review(cid, Correction(approve=True, reviewer="auto:judge"))

    async def retry_teacher(self, cid: str) -> None:
        cand = await self.store.hget(("dpo", "candidates"), cid)
        if cand is None:
            raise KeyError(cid)
        cand["status"] = "pending_teacher"
        await self.store.hset(("dpo", "candidates"), cid, cand)
        await self._ask_teacher(cid)

    async def list(self, status: str | None = None, domain: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        items = list((await self.store.hgetall(("dpo", "candidates"))).values())
        items = [c for c in items if (status is None or c["status"] == status) and (domain is None or c["domain"] == domain)]
        items.sort(key=lambda c: c.get("created_at", 0), reverse=True)
        return items[:limit]

    async def get(self, cid: str) -> dict[str, Any] | None:
        return await self.store.hget(("dpo", "candidates"), cid)

    async def review(self, cid: str, corr: Correction) -> dict[str, Any]:
        cand = await self.store.hget(("dpo", "candidates"), cid)
        if cand is None:
            raise KeyError(cid)
        if cand["status"] in ("approved", "rejected"):
            raise ValueError(f"candidate already {cand['status']}")
        domain = self.domains.get(cand["domain"])
        if not corr.approve:
            cand.update(status="rejected", reviewer=corr.reviewer, review_note=corr.note, reviewed_at=time.time())
            await self.store.hset(("dpo", "candidates"), cid, cand)
            self.bus.publish("dpo", "reviewed", id=cid, status="rejected")
            return cand
        chosen = corr.chosen or cand.get("chosen")
        if chosen is None:
            raise ValueError("no chosen action: edit one in the studio or retry the teacher")
        v = domain.validate_action(chosen, cand["state"])
        if not v.ok or v.hallucinated:
            raise ValueError(f"chosen action invalid: {v.errors + v.grounding_errors}")
        chosen = v.action or chosen
        source = "human" if corr.chosen or not (corr.reviewer or "").startswith("auto:") else "oracle"
        pair = DPOPair(domain=cand["domain"], state=cand["state"], rejected=cand["rejected"], chosen=chosen,
                       delta=cand.get("delta"), cot=cand.get("cot"), source=source, replay_seq=cand["seq"])
        await self.datasets.add_dpo(pair)
        await self.datasets.add_sft(SFTSample(domain=cand["domain"], state=cand["state"], action=chosen, cot=cand.get("cot"),
                                              source=f"dpo_{source}", replay_seq=cand["seq"],
                                              judge_score=(cand.get("judge") or {}).get("score")))
        cand.update(status="approved", chosen=chosen, reviewer=corr.reviewer, review_note=corr.note,
                    reviewed_at=time.time(), edited=corr.chosen is not None)
        await self.store.hset(("dpo", "candidates"), cid, cand)
        self.bus.publish("dpo", "reviewed", id=cid, status="approved", source=source)
        if self.on_new_pairs:
            await self.on_new_pairs(cand["domain"])
        return cand

    async def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"candidates": {}}
        for c in (await self.store.hgetall(("dpo", "candidates"))).values():
            out["candidates"][c["status"]] = out["candidates"].get(c["status"], 0) + 1
        for d in self.domains.all():
            out[d.id] = {
                "success": await self.store.counter("dpo", d.id, "success"),
                "failure": await self.store.counter("dpo", d.id, "failure"),
                "pairs": await self.datasets.count(d.id, "dpo"),
            }
        return out


def _same_action(field: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = {field, "target_id", "coordinates", "text", "target_ioc", "verdict"}
    return all(a.get(k) == b.get(k) for k in keys)
