"""System One API: the Jev contract served locally.

``POST /api/v1/systemone`` takes and returns exactly what TypeSafe's
``POST /v1/systemone`` does, so a client can switch between Jev and the
local model by changing only the base URL (and key). It is answered by the
Kenning, the local System One model (``S1_SYSTEM_ONE_BACKEND=kenning``, the default), or
by the label-token readout of a local LLM (``logprob``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from systemone_builder.api.deps import get_rt
from systemone_builder.runtime import Runtime
from systemone_builder.system_one.contract import SystemOneRequest, SystemOneResponse
from systemone_builder.system_one.engines import EngineError, SystemOneEngine
from systemone_builder.system_one.factory import api_engine

router = APIRouter(tags=["system-one"])


def _engine(request: Request, rt: Runtime) -> SystemOneEngine:
    engine = getattr(request.app.state, "system_one_engine", None)
    if engine is None:
        engine = request.app.state.system_one_engine = api_engine(rt.settings)
    return engine


@router.post("/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def system_one(req: SystemOneRequest, request: Request, rt: Runtime = Depends(get_rt)) -> SystemOneResponse:
    try:
        return await _engine(request, rt).answer(req)
    except EngineError as exc:
        raise HTTPException(502, str(exc)) from exc
