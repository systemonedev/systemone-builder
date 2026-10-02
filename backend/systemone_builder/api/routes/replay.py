"""Replay buffer API - backs the 120GB RAM Replay Explorer."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from systemone_builder.api.deps import get_rt
from systemone_builder.contracts.replay import ReplayRecord
from systemone_builder.runtime import Runtime

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/stats")
async def stats(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    s = await rt.replay.stats()
    s["outcomes"] = await rt.replay.outcome_counts()
    return s


@router.post("", status_code=201)
async def append(record: ReplayRecord, rt: Runtime = Depends(get_rt)) -> ReplayRecord:
    rec = await rt.replay.append(record)
    rt.bus.publish("replay", "append", seq=rec.seq, domain=rec.domain)
    return rec


@router.get("")
async def list_records(
    start_seq: int | None = None,
    limit: int = Query(100, ge=1, le=1000),
    reverse: bool = True,
    rt: Runtime = Depends(get_rt),
) -> list[ReplayRecord]:
    return await rt.replay.range(start_seq, limit=limit, reverse=reverse)


@router.get("/scrub")
async def scrub(
    offset: int = Query(0, ge=0, description="0 = newest action, size-1 = oldest"),
    window: int = Query(1, ge=1, le=100),
    rt: Runtime = Depends(get_rt),
) -> dict[str, Any]:
    """Temporal scrubber: jump to an offset from the head of the buffer."""
    lo, hi = await rt.replay.bounds()
    if hi is None:
        return {"offset": offset, "records": [], "size": 0}
    seq = max(hi - offset, lo or 0)
    records = await rt.replay.range(seq, limit=window, reverse=True)
    return {"offset": hi - seq, "records": records, "size": await rt.replay.size(), "bounds": [lo, hi]}


@router.get("/{seq}")
async def get_record(seq: int, rt: Runtime = Depends(get_rt)) -> ReplayRecord:
    rec = await rt.replay.get(seq)
    if rec is None:
        raise HTTPException(404, f"seq {seq} not in replay window")
    return rec
