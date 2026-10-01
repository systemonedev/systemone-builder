"""System One API: the Jev contract served by the local engine.

``POST /api/v1/systemone`` takes and returns exactly what TypeSafe's
``POST /v1/systemone`` does, so a client can switch between Jev and the
local model by changing only the base URL (and key).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from systemone.api.deps import get_rt
from systemone.runtime import Runtime
from systemone.s1.contract import SystemOneRequest, SystemOneResponse
from systemone.s1.engines import EngineError, SystemOneEngine
from systemone.s1.factory import build_engine

router = APIRouter(tags=["system-one"])


def _engine(request: Request, rt: Runtime) -> SystemOneEngine:
    engine = getattr(request.app.state, "s1_local", None)
    if engine is None:
        engine = request.app.state.s1_local = build_engine("local", rt.settings)
    return engine


@router.post("/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def system_one(req: SystemOneRequest, request: Request, rt: Runtime = Depends(get_rt)) -> SystemOneResponse:
    try:
        return await _engine(request, rt).answer(req)
    except EngineError as exc:
        raise HTTPException(502, str(exc)) from exc
