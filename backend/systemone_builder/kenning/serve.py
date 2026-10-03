"""HTTP server for Kenning, speaking the System One wire format.

    POST /v1/systemone   {state, model?, questions} -> {model, answers, usage}
    GET  /health         status, the served model and its calibration
    POST /admin/load     {"model": "<name>"} hot-swap to another registered model

Any client of that wire format (``systemone`` client library, the builder's
engines) can use this server by pointing its base URL here. The port is
published on 127.0.0.1 only; ``/admin/load`` only accepts model names from the
registry (``$KENNING_HOME/models``) or the configured default, never paths.

Which model is served: ``$KENNING_HOME/active.json`` (written when a model is
activated in the Models page) if it names an existing model, else
``KENNING_MODEL`` (a model directory or a Hugging Face cross-encoder id; default
the ModernBERT zero-shot base). ``KENNING_MAX_LENGTH`` bounds tokens per pair.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from systemone_builder.kenning.model import DEFAULT_BASE, Kenning
from systemone_builder.system_one.contract import SystemOneRequest, SystemOneResponse

_model: Kenning | None = None
_swap = asyncio.Lock()


def home() -> Path:
    return Path(os.environ.get("KENNING_HOME", "/workspace/kenning"))


def default_model() -> str:
    return os.environ.get("KENNING_MODEL") or os.environ.get("S1_MODEL") or DEFAULT_BASE


def resolve(name: str) -> str:
    """A registered model name -> its directory; the configured default as is."""
    if name == default_model():
        return name
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("pass a registered model name, not a path")
    path = home() / "models" / name
    if not path.is_dir():
        raise ValueError(f"no model named {name!r} in {home() / 'models'}")
    return str(path)


def startup_model() -> str:
    active = home() / "active.json"
    if active.exists():
        try:
            return resolve(json.loads(active.read_text())["model"])
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[kenning-serve] ignoring {active}: {exc}", flush=True)
    return default_model()


def load(path: str) -> Kenning:
    max_len = os.environ.get("KENNING_MAX_LENGTH") or os.environ.get("S1_MODEL_MAX_LENGTH")
    return Kenning(path, max_length=int(max_len) if max_len else None)


def model() -> Kenning:
    global _model
    if _model is None:
        _model = load(startup_model())
    return _model


def describe(m: Kenning) -> dict[str, object]:
    return {"model": m.name, "path": m.path, "device": m.device, "temperature": m.temperature,
            "base_model": m.config.get("base_model", m.path), "max_length": m.max_length}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    m = await asyncio.to_thread(model)
    print(f"[kenning-serve] serving {m.name} ({m.path}) on {m.device}; temperatures {m.temperature}", flush=True)
    yield


app = FastAPI(title="Kenning (System One model)", version="1", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", **describe(model())}


class LoadRequest(BaseModel):
    model: str


@app.post("/admin/load")
async def admin_load(req: LoadRequest) -> dict[str, object]:
    global _model
    try:
        path = resolve(req.model)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    async with _swap:
        new = await asyncio.to_thread(load, path)  # the old model keeps serving meanwhile
        old, _model = _model, new
        del old
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - CPU-only or no torch: nothing to free
            pass
    print(f"[kenning-serve] now serving {new.name} ({new.path})", flush=True)
    return {"status": "ok", **describe(new)}


@app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
async def system_one(req: SystemOneRequest) -> SystemOneResponse:
    try:
        # inference runs in a worker thread; Kenning serialises requests itself
        return await asyncio.to_thread(model().answer, req)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
