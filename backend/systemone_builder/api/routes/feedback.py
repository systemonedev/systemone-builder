"""Phase 5 API: state-delta feedback, DPO corrections, evaluation sandbox."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from systemone_builder.api.deps import get_rt
from systemone_builder.contracts.replay import Outcome
from systemone_builder.dpo.loop import Correction, Feedback
from systemone_builder.evaluation.sandbox import EvalRequest
from systemone_builder.extraction.pipeline import Observation
from systemone_builder.factory.dataset import HeldOutSample
from systemone_builder.runtime import Runtime

router = APIRouter(tags=["dpo", "evaluation"])


def _domain(rt: Runtime, domain_id: str):
    try:
        return rt.domains.get(domain_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


# ----------------------------------------------------------------- feedback
@router.post("/feedback")
async def feedback(fb: Feedback, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        return await rt.dpo.feedback(fb)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/dpo/stats")
async def dpo_stats(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return await rt.dpo.stats()


@router.get("/dpo/candidates")
async def dpo_candidates(status: str | None = None, domain: str | None = None, limit: int = 100,
                         rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.dpo.list(status, domain, limit)


@router.get("/dpo/candidates/{cid}")
async def dpo_candidate(cid: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    cand = await rt.dpo.get(cid)
    if cand is None:
        raise HTTPException(404, "unknown candidate")
    return cand


@router.post("/dpo/candidates/{cid}/review")
async def dpo_review(cid: str, corr: Correction, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        return await rt.dpo.review(cid, corr)
    except KeyError:
        raise HTTPException(404, "unknown candidate") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.post("/dpo/candidates/{cid}/retry", status_code=202)
async def dpo_retry(cid: str, rt: Runtime = Depends(get_rt)) -> dict[str, str]:
    try:
        await rt.dpo.retry_teacher(cid)
    except KeyError:
        raise HTTPException(404, "unknown candidate") from None
    return {"status": "retried"}


# ------------------------------------------------------------- held-out set
class HeldOutImport(BaseModel):
    """Real samples: either an already-extracted ``state`` or a raw ``observation``."""

    state: dict[str, Any] | None = None
    observation: Observation | None = None
    expected: dict[str, Any]
    acceptable: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


@router.post("/eval/{domain_id}/heldout", status_code=201)
async def import_heldout(domain_id: str, items: list[HeldOutImport], rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    domain = _domain(rt, domain_id)
    ex = rt.extractor(domain_id)
    added, errors = 0, []
    for i, item in enumerate(items):
        if item.observation is not None:
            state = ex.extract(item.observation).state
        elif item.state is not None:
            state = ex.extract(Observation(kind="state", data=item.state)).state
        else:
            errors.append({"index": i, "error": "state or observation required"})
            continue
        v = domain.validate_action(item.expected, state)
        if not v.ok:
            errors.append({"index": i, "error": v.errors})
            continue
        if await rt.datasets.add_heldout(HeldOutSample(domain=domain_id, state=state, expected=v.action or item.expected,
                                                       acceptable=item.acceptable, tags=item.tags)):
            added += 1
    return {"added": added, "errors": errors, "total": await rt.datasets.count(domain_id, "heldout")}


class PromoteRequest(BaseModel):
    limit: int = Field(50, ge=1, le=10_000)
    tag: str = "replay"


@router.post("/eval/{domain_id}/heldout/from-replay", status_code=201)
async def promote_from_replay(domain_id: str, req: PromoteRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    """Promote verified real traffic (outcome=success) into the held-out set.

    Records already used as SFT training samples are skipped so the held-out
    set stays unseen by the student.
    """
    _domain(rt, domain_id)
    trained = {s.get("replay_seq") for s in rt.datasets.iter(domain_id, "sft")}
    added = 0
    for rec in await rt.replay.latest(rt.settings.replay_capacity):
        if added >= req.limit:
            break
        if rec.domain != domain_id or rec.outcome != Outcome.SUCCESS or rec.seq in trained or not rec.action:
            continue
        if await rt.datasets.add_heldout(HeldOutSample(domain=domain_id, state=rec.state, expected=rec.action, tags=[req.tag])):
            added += 1
    return {"added": added, "total": await rt.datasets.count(domain_id, "heldout")}


# --------------------------------------------------------------- evaluation
@router.post("/eval/run", status_code=202)
async def eval_run(req: EvalRequest, rt: Runtime = Depends(get_rt)) -> dict[str, str]:
    _domain(rt, req.domain)
    return {"report_id": rt.evaluation.launch(req)}


@router.get("/eval/reports")
async def eval_reports(domain: str | None = None, rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.evaluation.reports(domain)


@router.get("/eval/reports/{report_id}")
async def eval_report(report_id: str, include_rows: bool = False, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    rep = rt.evaluation.load_report(report_id)
    if rep is None:
        summary = await rt.store.hget(("eval", "reports"), report_id)
        if summary is None:
            raise HTTPException(404, "unknown report")
        return summary
    if not include_rows:
        rep = {k: v for k, v in rep.items() if k != "rows"}
    return rep
