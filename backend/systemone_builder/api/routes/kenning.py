"""Kenning API: model registry, activation, export, and engine comparison.

    GET    /kenning/status                  the Kenning server's health and served model
    GET    /kenning/models                  trained models with their held-out metrics
    GET    /kenning/models/{name}           one model
    POST   /kenning/models/{name}/activate  serve it (hot swap, persisted in active.json)
    DELETE /kenning/models/{name}           delete (not the active one)
    POST   /kenning/models/{name}/export    build the export bundle, return its metadata
    GET    /kenning/models/{name}/export    download the bundle (built on demand)
    GET    /systemone/engines               which engines can answer here
    POST   /systemone/compare               one request on several engines, side by side
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from systemone_builder.api.deps import get_rt
from systemone_builder.kenning import registry
from systemone_builder.runtime import Runtime
from systemone_builder.system_one.contract import SystemOneRequest
from systemone_builder.system_one.engines import EngineError
from systemone_builder.system_one.factory import build_engine

router = APIRouter(tags=["kenning"])


def _home(rt: Runtime):  # noqa: ANN202
    return rt.settings.kenning_dir()


async def _kenning_health(rt: Runtime) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{rt.settings.kenning_url}/health")
        return {"online": r.status_code == 200, **(r.json() if r.status_code == 200 else {})}
    except httpx.HTTPError as exc:
        return {"online": False, "error": f"Kenning server unreachable at {rt.settings.kenning_url}: {exc!r}"}


def _wrap(fn, *a):  # noqa: ANN001, ANN002, ANN202
    try:
        return fn(*a)
    except registry.RegistryError as exc:
        raise HTTPException(404 if "no model" in str(exc) else 400, str(exc)) from exc


@router.get("/kenning/status")
async def status(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {"server": await _kenning_health(rt), "active": registry.active(_home(rt)),
            "home": str(_home(rt))}


@router.get("/kenning/models")
async def models(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(registry.list_models, _home(rt))


@router.get("/kenning/models/{name}")
async def model(name: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return _wrap(registry.summary, _home(rt), name)


@router.post("/kenning/models/{name}/activate")
async def activate(name: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    _wrap(registry.model_dir, _home(rt), name)
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{rt.settings.kenning_url}/admin/load", json={"model": name})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Kenning server unreachable: {exc!r}") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"Kenning server could not load {name}: {r.text[:300]}")
    _wrap(registry.set_active, _home(rt), name)  # persist only once it is actually serving
    rt.bus.publish("system", "kenning_activated", model=name)
    return {"active": name, "server": r.json()}


@router.delete("/kenning/models/{name}")
async def delete(name: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    await asyncio.to_thread(_wrap, registry.delete_model, _home(rt), name)
    return {"deleted": name}


@router.post("/kenning/models/{name}/export")
async def export(name: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    t0 = time.perf_counter()
    path = await asyncio.to_thread(_wrap, registry.export_bundle, _home(rt), name)
    return {"name": name, "file": path.name, "size_bytes": path.stat().st_size,
            "seconds": round(time.perf_counter() - t0, 1), "download": f"/api/v1/kenning/models/{name}/export"}


@router.get("/kenning/models/{name}/export")
async def download(name: str, rt: Runtime = Depends(get_rt)) -> FileResponse:
    path = await asyncio.to_thread(_wrap, registry.export_bundle, _home(rt), name)
    return FileResponse(path, media_type="application/zip", filename=path.name)


# --------------------------------------------------------------- engines
@router.get("/systemone/engines")
async def engines(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    k = await _kenning_health(rt)
    return [
        {"id": "kenning", "name": f"Kenning ({k.get('model', 'offline')})", "available": k.get("online", False),
         "note": k.get("error") or "local, deterministic"},
        {"id": "jev", "name": "TypeSafe Jev (opt-in)", "available": bool(rt.settings.typesafe_api_key),
         "note": "uses your TYPESAFE_API_KEY; each call is a paid TypeSafe request under your TypeSafe agreement"
         if rt.settings.typesafe_api_key else "set TYPESAFE_API_KEY in .env to enable"},
    ]


class CompareRequest(BaseModel):
    request: SystemOneRequest
    engines: list[str] = Field(default_factory=lambda: ["kenning"], min_length=1, max_length=2)


@router.post("/systemone/compare")
async def compare(body: CompareRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    allowed = {"kenning", "jev"}
    bad = set(body.engines) - allowed
    if bad:
        raise HTTPException(400, f"engines must be among {sorted(allowed)}")

    async def run(kind: str) -> dict[str, Any]:
        engine = build_engine(kind, rt.settings)
        t0 = time.perf_counter()
        try:
            resp = await engine.answer(body.request)
            return {"engine": kind, "response": resp.model_dump(exclude_none=True),
                    "latency_ms": resp.latency_ms or (time.perf_counter() - t0) * 1000}
        except EngineError as exc:
            return {"engine": kind, "error": str(exc), "latency_ms": (time.perf_counter() - t0) * 1000}
        finally:
            await engine.aclose()

    return {"results": await asyncio.gather(*(run(k) for k in body.engines))}
