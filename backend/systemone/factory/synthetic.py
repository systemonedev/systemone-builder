"""Synthetic Distillation & Validation Engine (Module B, Phase 4).

Runs against the Mac M4 Max oracle so the Linux GPUs stay dedicated to
serving and training. Three producers feed the SFT dataset:

1. **Replay distillation** (continuous): pulls new replay-buffer records and
   turns them into ``(state -> CoT -> action)`` samples.

   * records answered by the oracle already carry a CoT label -> judged, kept
   * records answered by the GPU 1 triage tier -> re-derived by the oracle
     with full reasoning, then judged
   * a sample of confident student actions -> re-derived by the oracle
     (``replay_sample_rate``) to keep the student honest on its own traffic

2. **Seed synthesis** (on demand): the oracle invents realistic ``state_input``
   observations for each scenario of the domain, then labels them.

3. **Vision parsing**: screenshots of GUIs without accessibility tags are
   converted into ``viewport_tree`` nodes with bounding boxes.

Every candidate passes schema validation, grounding checks and an
LLM-as-a-Judge review before it enters the dataset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

from systemone.adapters.base import AdapterError, ModelAdapter
from systemone.contracts.replay import ReplayRecord, Tier
from systemone.datastore.replay_buffer import ReplayBuffer
from systemone.datastore.store import JsonStore
from systemone.domains.registry import DomainRegistry
from systemone.domains.spec import DomainSpec
from systemone.extraction.pipeline import Observation, StateExtractor
from systemone.extraction.prompt import PromptBuilder, canonical_state
from systemone.factory.dataset import DatasetStore, SFTSample
from systemone.routing.parse import parse_action
from systemone.telemetry.bus import EventBus

log = logging.getLogger(__name__)

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["score", "correct", "issues"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "correct": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
}

VISION_SCHEMA = {
    "type": "object",
    "required": ["elements"],
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "name", "bbox"],
                "properties": {
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
            },
        }
    },
}


class Judge:
    """LLM-as-a-Judge validation on the oracle endpoint."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    async def review(self, domain: DomainSpec, state: dict[str, Any], action: dict[str, Any], cot: str | None) -> dict[str, Any]:
        sys = (
            "You are a strict reviewer of training data for a reflex agent.\n"
            f"Domain: {domain.name}. {domain.description}\nPolicy: {domain.system_prompt}\n"
            "Given a state and a proposed action (with the reasoning that produced it), judge whether the "
            "action is the correct, safe next step. Penalize actions that reference elements or IOCs that are "
            "not in the state, unsafe containment of internal/benign traffic, and actions that ignore visible "
            "errors. Return JSON: {score: 0..1, correct: bool, issues: [str]}."
        )
        user = json.dumps({"state": state, "action": action, "reasoning": cot or ""}, ensure_ascii=False)
        gen = await self.adapter.generate(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            json_schema=JUDGE_SCHEMA, max_tokens=512, temperature=0.0,
        )
        verdict, _ = parse_action(gen.text)
        if not verdict or "score" not in verdict:
            return {"score": 0.0, "correct": False, "issues": ["judge output unparseable"]}
        verdict["score"] = float(min(max(verdict["score"], 0.0), 1.0))
        return verdict


class VisionParser:
    """Screenshot -> viewport_tree for GUIs lacking accessibility tags."""

    def __init__(self, adapter: ModelAdapter, model: str | None = None) -> None:
        self.adapter = adapter
        self.model = model

    async def parse(self, screenshot_b64: str, viewport: tuple[int, int] | None = None) -> list[dict[str, Any]]:
        dims = f" The screenshot is {viewport[0]}x{viewport[1]} pixels." if viewport else ""
        messages = [
            {"role": "system", "content": (
                "You convert GUI screenshots into an accessibility-style element list. List every visible "
                "interactive element (button, link, textbox, checkbox, combobox, tab, menuitem ...) and "
                "important context (headings, alerts, dialogs). bbox is [x, y, width, height] in screenshot "
                "pixels. Use the visible label as name." + dims)},
            {"role": "user", "content": "Extract the elements of this screenshot."},
        ]
        gen = await self.adapter.generate(messages, json_schema=VISION_SCHEMA, max_tokens=4096, images=[screenshot_b64], model=self.model)
        obj, _ = parse_action(gen.text)
        elements = (obj or {}).get("elements") or []
        return [
            {"role": str(e.get("role", "")).lower(), "name": str(e.get("name", "")), "bbox": e.get("bbox"), "visible": True}
            for e in elements if isinstance(e, dict) and e.get("bbox") and len(e["bbox"]) == 4
        ]


class SyntheticFactory:
    def __init__(
        self,
        oracle: ModelAdapter,
        replay: ReplayBuffer,
        store: JsonStore,
        datasets: DatasetStore,
        domains: DomainRegistry,
        bus: EventBus,
        *,
        batch_size: int = 8,
        poll_interval_s: float = 5.0,
        use_judge: bool = True,
        on_new_samples: Any = None,
    ) -> None:
        self.oracle = oracle
        self.replay = replay
        self.store = store
        self.datasets = datasets
        self.domains = domains
        self.bus = bus
        self.batch_size = batch_size
        self.poll_interval_s = poll_interval_s
        self.judge = Judge(oracle) if use_judge else None
        self.on_new_samples = on_new_samples
        self._task: asyncio.Task[None] | None = None
        self._jobs: dict[str, asyncio.Task[Any]] = {}

    # ----------------------------------------------------------- labelling
    async def teach(self, domain: DomainSpec, state: dict[str, Any], hint: str | None = None,
                    temperature: float = 0.2) -> tuple[dict[str, Any] | None, str | None, list[str]]:
        """Ask the oracle for a (CoT, action) label for ``state``."""
        messages = PromptBuilder(domain).teacher_messages(state, hint)
        gen = await self.oracle.generate(messages, max_tokens=domain.factory.cot_max_tokens + 256, temperature=temperature)
        action, cot = parse_action(gen.text)
        if action is None:
            return None, cot, ["unparseable teacher output"]
        v = domain.validate_action(action, state)
        errs = v.errors + v.grounding_errors
        return (v.action if v.ok and not v.hallucinated else None), cot, errs

    async def accept(self, domain: DomainSpec, state: dict[str, Any], action: dict[str, Any], cot: str | None,
                     source: str, replay_seq: int | None = None) -> bool:
        score = None
        if self.judge is not None:
            verdict = await self.judge.review(domain, state, action, cot)
            score = verdict["score"]
            if not verdict.get("correct") or score < domain.factory.judge_min_score:
                await self.store.incr("factory", domain.id, "rejected")
                self.bus.publish("factory", "rejected", domain=domain.id, score=score, issues=verdict.get("issues"), source=source)
                return False
        added = await self.datasets.add_sft(SFTSample(domain=domain.id, state=state, cot=cot, action=action,
                                                      source=source, judge_score=score, replay_seq=replay_seq))
        if added:
            await self.store.incr("factory", domain.id, "accepted")
            self.bus.publish("factory", "sample", domain=domain.id, source=source, judge_score=score, replay_seq=replay_seq)
        return added

    # -------------------------------------------------- replay distillation
    def _needs_label(self, rec: ReplayRecord, domain: DomainSpec) -> str | None:
        label = rec.meta.get("label_source")
        if label == "oracle" and rec.meta.get("oracle_action"):
            return "oracle"
        if rec.tier == Tier.TRIAGE:
            return "triage"
        if rec.tier == Tier.STUDENT and rec.outcome.value != "failure" and random.random() < domain.factory.replay_sample_rate:
            return "student_audit"
        return None

    async def distill_record(self, rec: ReplayRecord) -> bool:
        try:
            domain = self.domains.get(rec.domain)
        except KeyError:
            return False
        kind = self._needs_label(rec, domain)
        if kind is None:
            return False
        if kind == "oracle":
            return await self.accept(domain, rec.state, rec.meta["oracle_action"], rec.meta.get("oracle_cot"), "oracle", rec.seq)
        hint = None
        if rec.action and kind == "triage":
            hint = f"A faster model answered {json.dumps(rec.action)}. Verify it from first principles."
        action, cot, errs = await self.teach(domain, rec.state, hint)
        if action is None:
            self.bus.publish("factory", "teacher_invalid", domain=domain.id, seq=rec.seq, errors=errs)
            return False
        return await self.accept(domain, rec.state, action, cot, "oracle" if kind == "triage" else "oracle_audit", rec.seq)

    async def step(self) -> int:
        """Process one batch of new replay records. Returns samples added."""
        cursor = int(await self.store.get("factory", "cursor", default=0))
        records = await self.replay.range(cursor + 1, limit=self.batch_size)
        # Don't label escalations still waiting on the oracle.
        ready: list[ReplayRecord] = []
        for r in records:
            if r.meta.get("escalation_id") and not r.meta.get("oracle_action"):
                job = await self.store.hget(("oracle", "jobs"), r.meta["escalation_id"])
                if job and job.get("status") in ("queued", "running"):
                    break
            ready.append(r)
        added = 0
        for rec in ready:
            try:
                added += int(await self.distill_record(rec))
            except AdapterError as exc:
                log.warning("oracle unavailable during distillation: %s", exc)
                break  # retry from this cursor next poll
            await self.store.set(rec.seq, "factory", "cursor")
        if added and self.on_new_samples:
            await self.on_new_samples({r.domain for r in ready})
        return added

    async def _loop(self) -> None:
        self.bus.publish("factory", "started")
        while True:
            try:
                n = await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("factory step failed")
                n = 0
            if n == 0:
                await asyncio.sleep(self.poll_interval_s)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="synthetic-factory")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for t in self._jobs.values():
            t.cancel()
        self.bus.publish("factory", "stopped")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------- seed synthesis
    async def synthesize_states(self, domain: DomainSpec, scenario: str, n: int) -> list[dict[str, Any]]:
        schema = {"type": "object", "required": ["states"],
                  "properties": {"states": {"type": "array", "items": domain.state_schema}}}
        examples = [canonical_state(s["state"], domain.prompt_key_order) for s in domain.few_shots[:2]]
        messages = [
            {"role": "system", "content": (
                f"You generate realistic, diverse observations for training a reflex agent.\nDomain: {domain.name}. "
                f"{domain.description}\nEach observation is a state_input JSON object matching this schema:\n"
                f"{json.dumps(domain.state_schema)}\nExamples:\n" + "\n".join(examples) +
                "\nVary difficulty: include easy, ambiguous and adversarial cases, and cases where the correct "
                "action is ESCALATE. Do not include answers.")},
            {"role": "user", "content": f"Scenario: {scenario}\nGenerate {n} distinct state_input objects as {{\"states\": [...]}}."},
        ]
        gen = await self.oracle.generate(messages, json_schema=schema, max_tokens=min(1200 * n, 16000),
                                         temperature=domain.factory.temperature)
        obj, _ = parse_action(gen.text)
        states = [s for s in (obj or {}).get("states", []) if isinstance(s, dict)]
        extractor = StateExtractor(domain)
        out = []
        for s in states:
            if domain.validate_state(s):
                continue
            # Run synthetic states through the same scrubber as live traffic.
            out.append(extractor.extract(Observation(kind="state", data=s)).state)
        return out

    async def synthesize(self, domain_id: str, scenarios: list[str] | None = None, per_scenario: int | None = None) -> dict[str, Any]:
        domain = self.domains.get(domain_id)
        scenarios = scenarios or domain.factory.scenarios
        per = per_scenario or domain.factory.seed_states_per_scenario
        report = {"domain": domain_id, "generated_states": 0, "accepted": 0, "rejected": 0, "scenarios": len(scenarios)}
        for scenario in scenarios:
            remaining = per
            while remaining > 0:
                chunk = min(remaining, 5)
                remaining -= chunk
                try:
                    states = await self.synthesize_states(domain, scenario, chunk)
                except AdapterError as exc:
                    report["error"] = str(exc)
                    return report
                report["generated_states"] += len(states)
                for st in states:
                    action, cot, _ = await self.teach(domain, st, f"Scenario: {scenario}")
                    if action is not None and await self.accept(domain, st, action, cot, "synthetic"):
                        report["accepted"] += 1
                    else:
                        report["rejected"] += 1
                self.bus.publish("factory", "synthesis_progress", **report, scenario=scenario)
        if report["accepted"] and self.on_new_samples:
            await self.on_new_samples({domain_id})
        return report

    def launch_synthesis(self, domain_id: str, scenarios: list[str] | None, per_scenario: int | None) -> str:
        job_id = f"synth-{domain_id}-{int(time.time())}"

        async def run() -> None:
            await self.store.hset(("factory", "jobs"), job_id, {"id": job_id, "status": "running", "domain": domain_id})
            try:
                rep = await self.synthesize(domain_id, scenarios, per_scenario)
                await self.store.hset(("factory", "jobs"), job_id, {"id": job_id, "status": "done", **rep})
            except Exception as exc:
                log.exception("synthesis job failed")
                await self.store.hset(("factory", "jobs"), job_id, {"id": job_id, "status": "failed", "error": repr(exc)})

        self._jobs[job_id] = asyncio.create_task(run(), name=job_id)
        return job_id

    async def status(self) -> dict[str, Any]:
        per_domain = {}
        for d in self.domains.all():
            per_domain[d.id] = {
                "accepted": await self.store.counter("factory", d.id, "accepted"),
                "rejected": await self.store.counter("factory", d.id, "rejected"),
                **await self.datasets.stats(d.id),
            }
        return {
            "running": self.running,
            "cursor": await self.store.get("factory", "cursor", default=0),
            "domains": per_domain,
            "jobs": list((await self.store.hgetall(("factory", "jobs"))).values()),
        }
