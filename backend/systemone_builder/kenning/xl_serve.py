"""HTTP server for Kenning-XL, speaking the System One wire format.

    POST /v1/systemone   {state, questions, model?}  -> {model, answers, usage}
    GET  /health         status, the served model, the escalation policy

Kenning-XL answers one-pass (constrained readout) by default, and **escalates to deliberate mode**
(generate a short reasoning trace, then read the answer) for any question whose one-pass confidence is
below a threshold -- so the extra latency is spent only where the model is unsure. Set the threshold with
``S1_XL_ESCALATE`` (0 = never escalate, 1 = always). A request may also force it with ``{"deliberate": true}``
(handled by clients that add the field); the default policy needs no client change.

Model: ``XL_MODEL`` (a model directory or a Hugging Face id, default ``Qwen/Qwen3-1.7B``);
``XL_MAX_LENGTH`` bounds context. Port on 127.0.0.1.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI

from systemone_builder.kenning.xl import DEFAULT_XL, KenningXL
from systemone_builder.system_one.contract import SystemOneRequest, SystemOneResponse

_model: KenningXL | None = None


def _confidence(ans: Any) -> float:
    """A 0..1 confidence for any answer type (noul: distance from 0.5)."""
    if ans.type == "noul":
        return abs(ans.noul - 0.5) * 2
    return ans.confidence


def escalate_threshold() -> float:
    try:
        return float(os.environ.get("S1_XL_ESCALATE", "0.75"))
    except ValueError:
        return 0.75


def model() -> KenningXL:
    assert _model is not None
    return _model


def answer(req: SystemOneRequest) -> SystemOneResponse:
    """One-pass, then re-answer the low-confidence questions in deliberate mode."""
    m = model()
    resp = m.system_one(req)
    thr = escalate_threshold()
    unsure = [q for q, a in resp.answers.items() if _confidence(a) < thr]
    if thr > 0 and unsure:
        deep = m.system_one(SystemOneRequest(state=req.state, questions={q: req.questions[q] for q in unsure}),
                            deliberate=True)
        resp.answers.update(deep.answers)
    return resp


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _model
    name = os.environ.get("XL_MODEL") or DEFAULT_XL
    _model = KenningXL(name, max_length=int(os.environ.get("XL_MAX_LENGTH", "4096")),
                       load_4bit=os.environ.get("XL_LOAD_4BIT", "0") == "1")
    yield
    _model = None


app = FastAPI(title="Kenning-XL (System One decoder)", version="1", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    m = model()
    return {"status": "ok", "model": m.name, "engine": "kenning-xl", "max_length": m.max_length,
            "escalate_below": escalate_threshold(), "temperature": m.temperature}


@app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def system_one(req: SystemOneRequest) -> SystemOneResponse:
    import anyio
    return await anyio.to_thread.run_sync(answer, req)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
