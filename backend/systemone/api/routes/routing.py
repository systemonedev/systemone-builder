"""Phase 3 API: act (fast-slow routing), thresholds, escalations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from systemone.api.deps import get_rt
from systemone.extraction.pipeline import Observation
from systemone.routing.router import Decision
from systemone.runtime import Runtime

router = APIRouter(tags=["routing"])


class ActRequest(BaseModel):
    observation: Observation
    session_id: str = "default"
    wait_for_oracle: bool = False
    oracle_timeout_s: float = Field(120.0, gt=0, le=1800)


class ThresholdUpdate(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    triage_threshold: float | None = Field(None, ge=0.0, le=1.0)
    triage_enabled: bool | None = None


def _domain(rt: Runtime, domain_id: str):
    try:
        return rt.domains.get(domain_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/act/{domain_id}")
async def act(domain_id: str, req: ActRequest, rt: Runtime = Depends(get_rt)) -> Decision:
    _domain(rt, domain_id)
    try:
        return await rt.router.act(domain_id, req.observation, session_id=req.session_id,
                                   wait_for_oracle=req.wait_for_oracle, oracle_timeout_s=req.oracle_timeout_s)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from None


@router.delete("/sessions/{domain_id}/{session_id}", status_code=204)
async def reset_session(domain_id: str, session_id: str, rt: Runtime = Depends(get_rt)) -> None:
    await rt.router.reset_session(domain_id, session_id)


@router.get("/routing/thresholds")
async def thresholds(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return [
        {"domain": d.id, "name": d.name, "threshold": d.threshold, "triage_threshold": d.effective_triage_threshold,
         "triage_enabled": d.triage_enabled}
        for d in rt.domains.all()
    ]


@router.put("/routing/thresholds/{domain_id}")
async def set_threshold(domain_id: str, upd: ThresholdUpdate, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    _domain(rt, domain_id)
    changes = upd.model_dump(exclude_none=True)
    spec = await rt.domains.update(domain_id, **changes)
    rt.bus.publish("routing", "threshold_changed", domain=domain_id, threshold=spec.threshold,
                   triage_threshold=spec.effective_triage_threshold, triage_enabled=spec.triage_enabled)
    return {"domain": spec.id, "threshold": spec.threshold, "triage_threshold": spec.effective_triage_threshold,
            "triage_enabled": spec.triage_enabled}


@router.get("/routing/stats")
async def routing_stats(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {d.id: await rt.router.stats(d.id) for d in rt.domains.all()}


@router.get("/routing/escalations")
async def escalations(status: str | None = None, limit: int = 100, rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.escalations.list(status, limit)


@router.get("/routing/escalations/{job_id}")
async def escalation(job_id: str, wait_s: float = 0.0, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    job = await (rt.escalations.wait_for(job_id, wait_s) if wait_s > 0 else rt.escalations.get(job_id))
    if job is None:
        raise HTTPException(404, "unknown escalation")
    return job


@router.get("/models")
async def models(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role, adapter in (("student", rt.student), ("triage", rt.triage), ("oracle", rt.oracle)):
        info = adapter.describe()
        try:
            info["available"] = await adapter.list_models()
        except Exception as exc:
            info["error"] = repr(exc)
        out[role] = info
    out["student"]["served_model"] = await rt.student_model_name()
    return out
