"""Phase 4 API: synthetic factory, datasets, training lifecycle, vision."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from systemone_builder.api.deps import get_rt
from systemone_builder.factory.dataset import SFTSample
from systemone_builder.runtime import Runtime
from systemone_builder.training.lifecycle import LifecycleError

router = APIRouter(tags=["factory", "training"])


def pipeline_on(rt: Runtime = Depends(get_rt)) -> Runtime:
    """Guard for actions that drive the vLLM student (compose profile "pipeline")."""
    if not rt.settings.pipeline:
        raise HTTPException(409, "the generative pipeline is off: set S1_PIPELINE=1 and start the "
                                 "\"pipeline\" compose profile (docker compose --profile pipeline up -d)")
    return rt


def _domain(rt: Runtime, domain_id: str):
    try:
        return rt.domains.get(domain_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


# ------------------------------------------------------------------ factory
class SynthesizeRequest(BaseModel):
    domain: str
    scenarios: list[str] | None = None
    per_scenario: int | None = Field(None, ge=1, le=500)


@router.get("/factory/status")
async def factory_status(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return await rt.factory.status()


@router.post("/factory/start")
async def factory_start(rt: Runtime = Depends(pipeline_on)) -> dict[str, Any]:
    rt.factory.start()
    return {"running": rt.factory.running}


@router.post("/factory/stop")
async def factory_stop(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    await rt.factory.stop()
    return {"running": rt.factory.running}


@router.post("/factory/synthesize", status_code=202)
async def synthesize(req: SynthesizeRequest, rt: Runtime = Depends(pipeline_on)) -> dict[str, Any]:
    _domain(rt, req.domain)
    return {"job_id": rt.factory.launch_synthesis(req.domain, req.scenarios, req.per_scenario)}


# ----------------------------------------------------------------- datasets
@router.get("/datasets/{domain_id}")
async def dataset_stats(domain_id: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    _domain(rt, domain_id)
    return await rt.datasets.stats(domain_id)


@router.post("/datasets/{domain_id}/reconcile")
async def dataset_reconcile(domain_id: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    """Rebuild counters and dedup keys from the dataset files on disk."""
    _domain(rt, domain_id)
    return await rt.datasets.reconcile(domain_id)


@router.get("/datasets/{domain_id}/{split}")
async def dataset_tail(domain_id: str, split: Literal["sft", "dpo", "heldout"], n: int = 50,
                       rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    _domain(rt, domain_id)
    return rt.datasets.tail(domain_id, split, n)


class ManualSample(BaseModel):
    state: dict[str, Any]
    action: dict[str, Any]
    cot: str | None = None


@router.post("/datasets/{domain_id}/sft", status_code=201)
async def add_sample(domain_id: str, sample: ManualSample, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    domain = _domain(rt, domain_id)
    v = domain.validate_action(sample.action, sample.state)
    if not v.ok or v.hallucinated:
        raise HTTPException(422, {"errors": v.errors, "grounding_errors": v.grounding_errors})
    added = await rt.datasets.add_sft(SFTSample(domain=domain_id, state=sample.state, action=v.action or sample.action,
                                                cot=sample.cot, source="human"))
    return {"added": added}


@router.post("/datasets/{domain_id}/sft/bulk", status_code=201)
async def add_samples_bulk(domain_id: str, samples: list[ManualSample], source: str = "import",
                           rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    domain = _domain(rt, domain_id)
    added, errors = 0, []
    for i, sample in enumerate(samples):
        v = domain.validate_action(sample.action, sample.state)
        if not v.ok or v.hallucinated:
            errors.append({"index": i, "errors": v.errors + v.grounding_errors})
            continue
        added += await rt.datasets.add_sft(SFTSample(domain=domain_id, state=sample.state, action=v.action or sample.action,
                                                     cot=sample.cot, source=source))
    return {"added": added, "errors": errors}


# ----------------------------------------------------------------- training
class TrainRequest(BaseModel):
    domain: str
    mode: Literal["sft", "dpo"] = "sft"


@router.get("/training/status")
async def training_status(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return await rt.lifecycle.status()


@router.post("/training/run", status_code=202)
async def training_run(req: TrainRequest, rt: Runtime = Depends(pipeline_on)) -> dict[str, Any]:
    _domain(rt, req.domain)
    try:
        cfg = rt.lifecycle.start_cycle(req.domain, req.mode)
    except LifecycleError as exc:
        raise HTTPException(409, str(exc)) from None
    return cfg


@router.get("/training/runs")
async def training_runs(limit: int = 50, rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.lifecycle.runs(limit)


@router.get("/training/runs/{run_id}/metrics")
async def training_metrics(run_id: str, rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.lifecycle.run_metrics(run_id)


class RollbackRequest(BaseModel):
    run_id: str | None = None  # None = base model


@router.post("/training/rollback")
async def training_rollback(req: RollbackRequest, rt: Runtime = Depends(pipeline_on)) -> dict[str, Any]:
    try:
        return await rt.lifecycle.rollback_to(req.run_id)
    except LifecycleError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/training/recover")
async def training_recover(rt: Runtime = Depends(pipeline_on)) -> dict[str, Any]:
    try:
        return await rt.lifecycle.recover()
    except LifecycleError as exc:
        raise HTTPException(409, str(exc)) from None


# ------------------------------------------------------------------- vision
class VisionRequest(BaseModel):
    screenshot_b64: str
    viewport: tuple[int, int] | None = None


@router.post("/vision/parse")
async def vision_parse(req: VisionRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    if rt.vision is None:
        raise HTTPException(503, "set S1_ORACLE_VISION_MODEL to enable screenshot parsing")
    elements = await rt.vision.parse(req.screenshot_b64, req.viewport)
    return {"elements": elements}
