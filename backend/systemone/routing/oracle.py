"""Asynchronous oracle escalation (Mac M4 Max, Ollama).

Queries that neither the student nor the triage tier can answer confidently
are queued in Redis and processed by :class:`OracleEscalationService` against
the System-2 teacher. Results are persisted, attached to the replay record as
a distillation label and broadcast on the event bus. Callers may block on a
result (``wait_for``) or poll it later.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from systemone.adapters.base import AdapterError, ModelAdapter
from systemone.datastore.replay_buffer import ReplayBuffer
from systemone.datastore.store import JsonStore
from systemone.domains.registry import DomainRegistry
from systemone.extraction.prompt import PromptBuilder
from systemone.routing.parse import parse_action
from systemone.telemetry.bus import EventBus

log = logging.getLogger(__name__)

DEEP_ANALYSIS = {
    "secops": (
        "Escalated for deep out-of-band threat analysis. Consider attack technique (MITRE ATT&CK), "
        "false-positive likelihood, blast radius of containment, and whether the source is internal."
    ),
    "computer_use": (
        "Escalated because the reflex model was uncertain. Consider the goal, the action history, "
        "error banners or dialogs, and whether a pixel-coordinate fallback is required."
    ),
}


class OracleEscalationService:
    def __init__(
        self,
        adapter: ModelAdapter,
        store: JsonStore,
        replay: ReplayBuffer,
        domains: DomainRegistry,
        bus: EventBus,
        concurrency: int = 1,
        max_tokens: int = 2048,
    ) -> None:
        self.adapter = adapter
        self.store = store
        self.replay = replay
        self.domains = domains
        self.bus = bus
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self._waiters: dict[str, list[asyncio.Future[dict[str, Any]]]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._wake = asyncio.Event()

    # --------------------------------------------------------------- submit
    async def submit(self, domain_id: str, state: dict[str, Any], *, seq: int | None, reason: list[str],
                     student_action: dict[str, Any] | None, triage_action: dict[str, Any] | None,
                     screenshot_b64: str | None = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "domain": domain_id, "state": state, "seq": seq, "reason": reason,
            "student_action": student_action, "triage_action": triage_action, "status": "queued",
            "submitted_at": time.time(), "screenshot_b64": screenshot_b64,
        }
        await self.store.hset(("oracle", "jobs"), job_id, {k: v for k, v in job.items() if k != "screenshot_b64"})
        await self.store.push(("oracle", "queue"), job)
        self.bus.publish("routing", "oracle_enqueued", job_id=job_id, domain=domain_id, seq=seq, reason=reason)
        self._wake.set()
        return job_id

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await self.store.hget(("oracle", "jobs"), job_id)

    async def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        jobs = list((await self.store.hgetall(("oracle", "jobs"))).values())
        jobs.sort(key=lambda j: j.get("submitted_at", 0), reverse=True)
        return [j for j in jobs if status is None or j.get("status") == status][:limit]

    async def wait_for(self, job_id: str, timeout_s: float) -> dict[str, Any] | None:
        job = await self.get(job_id)
        if job and job.get("status") in ("done", "failed"):
            return job
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(job_id, []).append(fut)
        try:
            return await asyncio.wait_for(fut, timeout_s)
        except asyncio.TimeoutError:
            return await self.get(job_id)
        finally:
            lst = self._waiters.get(job_id, [])
            if fut in lst:
                lst.remove(fut)

    # --------------------------------------------------------------- worker
    def start(self) -> None:
        for i in range(self.concurrency):
            self._tasks.append(asyncio.create_task(self._loop(), name=f"oracle-worker-{i}"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _loop(self) -> None:
        while True:
            jobs = await self.store.pop_many(("oracle", "queue"), 1)
            if not jobs:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self.process(jobs[0])
            except asyncio.CancelledError:
                raise
            except Exception:  # never let one job kill the worker
                log.exception("oracle job %s crashed", jobs[0].get("id"))

    async def process(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = job["id"]
        domain = self.domains.get(job["domain"])
        record = {k: v for k, v in job.items() if k != "screenshot_b64"}
        record.update(status="running", started_at=time.time())
        await self.store.hset(("oracle", "jobs"), job_id, record)
        self.bus.publish("routing", "oracle_started", job_id=job_id, domain=domain.id)

        extra = [DEEP_ANALYSIS.get(domain.kind, "Escalated for System-2 analysis.")]
        extra.append(f"Escalation reasons: {', '.join(job.get('reason') or ['low_confidence'])}.")
        for tier in ("student", "triage"):
            if job.get(f"{tier}_action"):
                extra.append(f"The {tier} model proposed (uncertain): {job[f'{tier}_action']}")
        messages = PromptBuilder(domain).teacher_messages(job["state"], "\n".join(extra))
        images = [job["screenshot_b64"]] if job.get("screenshot_b64") else None
        try:
            gen = await self.adapter.generate(messages, max_tokens=self.max_tokens, temperature=0.2, images=images)
            action, cot = parse_action(gen.text)
            validation = domain.validate_action(action, job["state"]) if action is not None else None
            ok = validation is not None and validation.ok and not validation.hallucinated
            record.update(
                status="done" if ok else "failed",
                action=validation.action if validation else action,
                cot=cot,
                errors=(validation.errors + validation.grounding_errors) if validation else ["unparseable teacher output"],
                latency_ms=gen.latency_ms,
                model=gen.model,
            )
        except AdapterError as exc:
            record.update(status="failed", errors=[str(exc)])
        record["finished_at"] = time.time()
        await self.store.hset(("oracle", "jobs"), job_id, record)

        if record["status"] == "done" and job.get("seq") is not None:
            rec = await self.replay.get(job["seq"])
            if rec is not None:
                meta = {**rec.meta, "oracle_action": record["action"], "oracle_cot": record.get("cot"),
                        "label_source": "oracle", "escalation_id": job_id}
                await self.replay.update(job["seq"], meta=meta)
        self.bus.publish("routing", "oracle_result", job_id=job_id, domain=domain.id, status=record["status"],
                         action=record.get("action"), latency_ms=record.get("latency_ms"))
        for fut in self._waiters.pop(job_id, []):
            if not fut.done():
                fut.set_result(record)
        return record
