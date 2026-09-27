"""Phase 7 API: BYOM endpoints, model validation, starter templates."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from systemone.adapters.byom import validate_hf_model
from systemone.adapters.factory import load_plugins
from systemone.api.deps import get_rt
from systemone.domains.spec import DomainSpec
from systemone.runtime import Runtime
from systemone.templates import list_templates, load_template

router = APIRouter(tags=["byom"])


class EndpointPatch(BaseModel):
    adapter: str | None = None
    url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    vision_model: str | None = None
    base_model: str | None = None


@router.get("/byom")
async def byom(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {
        "adapters": sorted(load_plugins()),
        "roles": {r: getattr(rt, r).describe() for r in ("student", "triage", "oracle")},
    }


@router.put("/byom/{role}")
async def swap(role: Literal["student", "triage", "oracle"], patch: EndpointPatch, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    if patch.base_model:
        v = await validate_hf_model(patch.base_model, os.environ.get("HF_TOKEN"))
        if not v["ok"]:
            raise HTTPException(422, v)
    try:
        cfg = await rt.swap_endpoint(role, patch.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    handle = getattr(rt, role)
    return {"role": role, "config": cfg.model_dump(exclude={"api_key_env"}), "health": await handle.health()}


@router.post("/byom/validate")
async def validate(model: str) -> dict[str, Any]:
    return await validate_hf_model(model, os.environ.get("HF_TOKEN"))


@router.get("/templates")
async def templates() -> list[dict[str, Any]]:
    return list_templates()


class InstallTemplate(BaseModel):
    id: str | None = None  # install under a different domain id
    bootstrap: bool = False


@router.post("/templates/{template_id}/install")
async def install(template_id: str, req: InstallTemplate, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        spec = load_template(template_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    if req.id:
        spec = DomainSpec.model_validate({**spec.model_dump(), "id": req.id})
    spec = spec.model_copy(update={"source": "template"})
    await rt.domains.put(spec)
    job = rt.factory.launch_synthesis(spec.id, None, None) if req.bootstrap else None
    return {"domain": spec.id, "bootstrap_job": job}
