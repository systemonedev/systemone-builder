from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from systemone import __version__
from systemone.api.deps import get_rt
from systemone.runtime import Runtime

router = APIRouter(tags=["system"])
public_router = APIRouter(tags=["system"])


@public_router.get("/health")
async def health(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {"status": "ok", "version": __version__, **await rt.health()}


@router.get("/health/deep")
async def deep_health(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {"status": "ok", "version": __version__, **await rt.deep_health()}


@router.get("/hardware")
async def hardware(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    snap = await rt.hardware.snapshot()
    snap["ram_datastore"] = await rt.store.memory_info()
    return snap


@router.get("/hardware/gpus")
async def gpus(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return [g.to_dict() for g in rt.gpu.snapshot()]


@router.get("/hardware/audit")
async def audit(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return await rt.hardware.audit()


@router.get("/hardware/containers/{name}/logs")
async def container_logs(name: str, tail: int = 200, rt: Runtime = Depends(get_rt)) -> dict[str, str]:
    return {"name": name, "logs": await rt.docker.logs(name, tail=tail)}
