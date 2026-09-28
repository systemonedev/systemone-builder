"""Evaluation & Accuracy Suite - benchmarking sandbox (Module G, Phase 5).

Runs a model against held-out, *real* samples the student never trained on
(e.g. 50 novel GUI states or log files) and reports:

* accuracy (full action match), action-label accuracy, per-action confusion
* success rate on autonomous actions (precision when not escalating),
  coverage (share handled without escalation), escalation rate
* hallucination ratio (ungrounded ids / IOCs, schema-invalid, unparseable)
* latency and TTFT distributions (p50/p90/p95/p99, histogram, % under 100ms)
* confidence calibration (ECE, Brier, reliability table) + a fitted Platt
  calibration that can be pushed to the live router
* prefix-cache reuse (cached / prompt tokens)
* a deployment-readiness verdict against explicit gates
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from systemone.adapters.base import AdapterError, ModelAdapter
from systemone.datastore.store import JsonStore
from systemone.domains.registry import DomainRegistry
from systemone.domains.spec import DomainSpec
from systemone.evaluation.metrics import calibration, distribution, full_match
from systemone.extraction.prompt import PromptBuilder
from systemone.factory.dataset import DatasetStore
from systemone.routing.confidence import Calibration, ConfidenceScorer
from systemone.routing.parse import parse_action
from systemone.telemetry.bus import EventBus

log = logging.getLogger(__name__)

Target = Literal["student", "triage", "oracle"]


class ReadinessGates(BaseModel):
    min_accuracy: float = 0.90
    min_success_rate: float = 0.97
    max_hallucination_ratio: float = 0.02
    max_p95_latency_ms: float = 100.0
    max_ece: float = 0.10
    min_samples: int = 30


class EvalRequest(BaseModel):
    domain: str
    target: Target = "student"
    model: str | None = None  # override served model name (e.g. a LoRA id)
    limit: int | None = Field(None, ge=1)
    warmup: int = Field(3, ge=0, le=50)
    gates: ReadinessGates = Field(default_factory=ReadinessGates)
    apply_calibration: bool = False
    tags: list[str] | None = None


class EvaluationSandbox:
    def __init__(
        self,
        adapters: dict[str, ModelAdapter],
        datasets: DatasetStore,
        domains: DomainRegistry,
        store: JsonStore,
        bus: EventBus,
        scorer: ConfidenceScorer,
        results_dir: Path,
        student_model_name: Any,
        max_tokens: int = 192,
    ) -> None:
        self.adapters = adapters
        self.datasets = datasets
        self.domains = domains
        self.store = store
        self.bus = bus
        self.scorer = scorer
        self.results_dir = results_dir
        self.student_model_name = student_model_name
        self.max_tokens = max_tokens
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        # report_id -> {"domain", "target", "done", "total", "started_at"} while running
        self.running: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------------- jobs
    def launch(self, req: EvalRequest) -> str:
        report_id = f"eval-{req.domain}-{req.target}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

        async def run() -> None:
            await self.store.hset(("eval", "reports"), report_id, {"id": report_id, "status": "running", "domain": req.domain,
                                                                   "target": req.target, "created_at": time.time()})
            try:
                report = await self.run(req, report_id)
                await self.store.hset(("eval", "reports"), report_id, _summary(report))
            except Exception as exc:
                self.running.pop(report_id, None)
                log.exception("evaluation %s failed", report_id)
                await self.store.hset(("eval", "reports"), report_id, {"id": report_id, "status": "failed", "error": repr(exc),
                                                                       "domain": req.domain, "target": req.target})
                self.bus.publish("eval", "failed", id=report_id, error=repr(exc))

        self._tasks[report_id] = asyncio.create_task(run(), name=report_id)
        return report_id

    async def reports(self, domain: str | None = None) -> list[dict[str, Any]]:
        items = [r for r in (await self.store.hgetall(("eval", "reports"))).values() if domain is None or r.get("domain") == domain]
        items.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return items

    def load_report(self, report_id: str) -> dict[str, Any] | None:
        for p in self.results_dir.glob(f"*/{report_id}.json"):
            return json.loads(p.read_text())
        return None

    # ---------------------------------------------------------------- run
    async def run(self, req: EvalRequest, report_id: str | None = None) -> dict[str, Any]:
        report_id = report_id or f"eval-{uuid.uuid4().hex[:8]}"
        domain = self.domains.get(req.domain)
        samples = list(self.datasets.iter(domain.id, "heldout"))
        if req.tags:
            samples = [s for s in samples if set(req.tags) & set(s.get("tags") or [])]
        if req.limit:
            samples = samples[: req.limit]
        if not samples:
            raise ValueError(f"no held-out samples for domain {domain.id!r}; import real samples first")
        adapter = self.adapters[req.target]
        model = req.model or (await self.student_model_name() if req.target == "student" else None)
        pb = PromptBuilder(domain)
        self.bus.publish("eval", "started", id=report_id, domain=domain.id, target=req.target, n=len(samples))
        self.running[report_id] = {"domain": domain.id, "target": req.target, "done": 0, "total": len(samples),
                                   "started_at": time.time()}

        # Warm the prefix cache / CUDA graphs so latency reflects steady state.
        for s in samples[: req.warmup]:
            try:
                await self._infer(adapter, domain, pb, s["state"], model)
            except AdapterError:
                break

        rows: list[dict[str, Any]] = []
        for i, s in enumerate(samples):
            rows.append(await self._evaluate_one(adapter, domain, pb, s, model))
            self.running[report_id]["done"] = i + 1
            if (i + 1) % 10 == 0 or i + 1 == len(samples):
                self.bus.publish("eval", "progress", id=report_id, done=i + 1, total=len(samples))

        report = self._aggregate(domain, req, rows, report_id, model)
        if req.apply_calibration and report["calibration"].get("fitted"):
            cal = report["calibration"]["fitted"]
            self.scorer.calibrations[domain.id] = Calibration(**cal)
            await self.store.hset(("routing", "calibration"), domain.id, cal)
            report["calibration"]["applied"] = True
        out = self.results_dir / domain.id / f"{report_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        self.running.pop(report_id, None)
        self.bus.publish("eval", "finished", **_summary(report))
        return report

    async def _infer(self, adapter: ModelAdapter, domain: DomainSpec, pb: PromptBuilder, state: dict[str, Any], model: str | None):
        messages, _ = pb.messages(state)
        return await adapter.generate(messages, json_schema=domain.guided_json_schema(), max_tokens=self.max_tokens,
                                      temperature=0.0, logprobs=True, model=model)

    async def _evaluate_one(self, adapter: ModelAdapter, domain: DomainSpec, pb: PromptBuilder, sample: dict[str, Any],
                            model: str | None) -> dict[str, Any]:
        state = sample["state"]
        expected = [sample["expected"], *(sample.get("acceptable") or [])]
        row: dict[str, Any] = {"id": sample.get("id"), "tags": sample.get("tags") or [],
                               "expected_label": domain.action_label(sample["expected"])}
        try:
            gen = await self._infer(adapter, domain, pb, state, model)
        except AdapterError as exc:
            row.update(error=str(exc), parsed=False, valid=False, hallucinated=False, confidence=0.0,
                       label_ok=False, correct=False, escalated=True)
            return row
        action, _ = parse_action(gen.text)
        v = domain.validate_action(action, state) if action is not None else None
        rep = self.scorer.score(domain, gen.text, action, v, gen.logprobs)
        pred = (v.action if v and v.action else action) or None
        label_ok, correct = full_match(domain, pred, expected, state)
        pred_label = domain.action_label(pred) if pred else None
        escalated = pred_label == "ESCALATE" or rep.confidence < domain.threshold
        row.update(
            parsed=action is not None,
            valid=bool(v and v.ok),
            hallucinated=bool(v and v.hallucinated) or (action is not None and not (v and v.ok)),
            errors=(v.errors + v.grounding_errors) if v else ["unparseable"],
            predicted=pred,
            predicted_label=pred_label,
            confidence=rep.confidence,
            raw_confidence=(action or {}).get("confidence_score"),
            label_ok=label_ok,
            correct=correct,
            escalated=escalated,
            latency_ms=gen.latency_ms,
            ttft_ms=gen.ttft_ms,
            prompt_tokens=gen.prompt_tokens,
            cached_tokens=gen.cached_tokens,
        )
        return row

    def _aggregate(self, domain: DomainSpec, req: EvalRequest, rows: list[dict[str, Any]], report_id: str,
                   model: str | None) -> dict[str, Any]:
        n = len(rows)
        correct = [r["correct"] for r in rows]
        autonomous = [r for r in rows if not r["escalated"]]
        # Escalating is the *right* answer when the expected label is ESCALATE.
        acc = sum(correct) / n
        success = (sum(r["correct"] for r in autonomous) / len(autonomous)) if autonomous else None
        halluc = sum(r["hallucinated"] for r in rows) / n
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        confs = [r["confidence"] for r in rows]
        # calibration of the *raw* self-report is what Platt scaling corrects
        raw = [float(r["raw_confidence"]) for r in rows if isinstance(r.get("raw_confidence"), (int, float))]
        raw_ok = [r["correct"] for r in rows if isinstance(r.get("raw_confidence"), (int, float))]
        fitted = Calibration.fit(confs, correct)
        cal = calibration(confs, correct)
        cal["raw_self_report"] = calibration(raw, raw_ok) if raw else None
        cal["fitted"] = {"a": fitted.a, "b": fitted.b} if (fitted.a, fitted.b) != (1.0, 0.0) else None

        confusion: dict[str, dict[str, int]] = {}
        for r in rows:
            confusion.setdefault(r["expected_label"] or "?", {})
            k = r.get("predicted_label") or "<invalid>"
            confusion[r["expected_label"] or "?"][k] = confusion[r["expected_label"] or "?"].get(k, 0) + 1
        by_tag: dict[str, dict[str, Any]] = {}
        for r in rows:
            for t in r["tags"]:
                b = by_tag.setdefault(t, {"n": 0, "correct": 0})
                b["n"] += 1
                b["correct"] += int(r["correct"])
        for b in by_tag.values():
            b["accuracy"] = b["correct"] / b["n"]
        prompt_toks = sum(r.get("prompt_tokens") or 0 for r in rows)
        cached_toks = sum(r.get("cached_tokens") or 0 for r in rows)

        lat_d, ttft_d = distribution(lat), distribution(ttft)
        g = req.gates
        checks = {
            "samples": (n >= g.min_samples, f"{n} >= {g.min_samples}"),
            "accuracy": (acc >= g.min_accuracy, f"{acc:.3f} >= {g.min_accuracy}"),
            "success_rate": (success is not None and success >= g.min_success_rate, f"{success if success is None else round(success, 3)} >= {g.min_success_rate}"),
            "hallucination_ratio": (halluc <= g.max_hallucination_ratio, f"{halluc:.3f} <= {g.max_hallucination_ratio}"),
            "p95_latency_ms": (lat_d.get("p95") is not None and lat_d["p95"] <= g.max_p95_latency_ms, f"{lat_d.get('p95')} <= {g.max_p95_latency_ms}"),
            "ece": (cal["ece"] is not None and cal["ece"] <= g.max_ece, f"{cal['ece']} <= {g.max_ece}"),
        }
        return {
            "id": report_id,
            "status": "done",
            "domain": domain.id,
            "target": req.target,
            "model": model,
            "threshold": domain.threshold,
            "created_at": time.time(),
            "n": n,
            "metrics": {
                "accuracy": acc,
                "action_accuracy": sum(r["label_ok"] for r in rows) / n,
                "success_rate": success,
                "coverage": len(autonomous) / n,
                "escalation_rate": 1 - len(autonomous) / n,
                "hallucination_ratio": halluc,
                "invalid_output_ratio": sum(not r.get("valid") for r in rows) / n,
                "unparseable_ratio": sum(not r.get("parsed") for r in rows) / n,
                "errors": sum(1 for r in rows if r.get("error")),
                "prefix_cache_token_ratio": (cached_toks / prompt_toks) if prompt_toks else None,
            },
            "latency_ms": lat_d,
            "ttft_ms": ttft_d,
            "calibration": cal,
            "confusion": confusion,
            "by_tag": by_tag,
            "readiness": {
                "ready_for_deployment": all(ok for ok, _ in checks.values()),
                "checks": {k: {"pass": ok, "detail": d} for k, (ok, d) in checks.items()},
                "gates": g.model_dump(),
            },
            "rows": rows,
        }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": report["id"], "status": report["status"], "domain": report["domain"], "target": report["target"],
        "model": report.get("model"), "n": report["n"], "created_at": report["created_at"],
        "accuracy": report["metrics"]["accuracy"], "success_rate": report["metrics"]["success_rate"],
        "hallucination_ratio": report["metrics"]["hallucination_ratio"], "p95_latency_ms": report["latency_ms"].get("p95"),
        "ready_for_deployment": report["readiness"]["ready_for_deployment"],
    }
