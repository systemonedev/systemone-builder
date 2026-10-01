"""HTTP server for the local System One model, speaking Jev's wire format.

    POST /v1/systemone   {state, model?, questions} -> {model, answers, usage}
    GET  /health

Because the endpoint and payloads match TypeSafe's, any Jev client (including
:class:`systemone.s1.engines.JevEngine`) can use this server by pointing its
base URL here.

Environment: ``S1_MODEL`` (a trained model directory, or a Hugging Face
cross-encoder id for zero-shot use; default the ModernBERT zero-shot base),
``S1_MODEL_MAX_LENGTH`` (tokens of state + hypothesis), ``PORT``.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException

from systemone.s1.contract import SystemOneRequest, SystemOneResponse
from systemone.s1model.model import DEFAULT_BASE, S1Model

app = FastAPI(title="systemone System One model", version="1")
_model: S1Model | None = None


def model() -> S1Model:
    global _model
    if _model is None:
        path = os.environ.get("S1_MODEL") or DEFAULT_BASE
        max_len = os.environ.get("S1_MODEL_MAX_LENGTH")
        _model = S1Model(path, max_length=int(max_len) if max_len else None)
    return _model


@app.on_event("startup")
async def _load() -> None:
    m = model()
    print(f"[s1-serve] loaded {m.path} on {m.device}; temperatures {m.temperature}", flush=True)


@app.get("/health")
async def health() -> dict[str, object]:
    m = model()
    return {"status": "ok", "model": m.name, "path": m.path, "device": m.device, "temperature": m.temperature}


@app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def system_one(req: SystemOneRequest) -> SystemOneResponse:
    try:
        # inference runs in a worker thread; S1Model serialises requests itself
        return await asyncio.to_thread(model().answer, req)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
