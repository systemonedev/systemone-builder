"""Phase 6 API: live WebSocket telemetry, telemetry history, Prompt-to-Workflow."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from systemone.api.deps import get_rt, ws_authorized
from systemone.domains.spec import DomainSpec
from systemone.runtime import Runtime
from systemone.workflow.engine import generate_safe

log = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])
ws_router = APIRouter()


@ws_router.websocket("/ws")
async def events(ws: WebSocket) -> None:
    """Live event stream. Query ``channels=routing,telemetry`` to filter.

    On connect the server sends ``{"type": "hello", "history": ...}`` with
    the recent telemetry samples and events, then streams every bus event.
    """
    # Accept before rejecting: a close before accept becomes an HTTP 403 in the
    # handshake, which browsers only surface as a generic code 1006.
    await ws.accept()
    if not ws_authorized(ws):
        log.info("websocket rejected: missing or wrong API key")
        await ws.close(code=4401, reason="missing or wrong API key")
        return
    rt: Runtime = ws.app.state.rt
    channels = set(filter(None, (ws.query_params.get("channels") or "").split(",")))
    q = rt.bus.subscribe()
    try:
        await ws.send_text(json.dumps({"type": "hello", "telemetry": rt.telemetry.snapshot(),
                                       "events": rt.bus.recent(limit=200)}, default=str))

        async def reader() -> None:
            nonlocal channels
            while True:
                msg = json.loads(await ws.receive_text())
                if "channels" in msg:
                    channels = set(msg["channels"] or [])

        read_task = asyncio.create_task(reader())
        try:
            while True:
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, read_task}, return_when=asyncio.FIRST_COMPLETED)
                if read_task in done:
                    get.cancel()
                    read_task.result()  # re-raise disconnects
                ev = get.result()
                if not channels or ev["channel"] in channels:
                    await ws.send_text(json.dumps(ev, default=str))
        finally:
            read_task.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rt.bus.unsubscribe(q)


@router.get("/telemetry")
async def telemetry(n: int = 300, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return rt.telemetry.snapshot(n)


@router.get("/events")
async def recent_events(channel: str | None = None, limit: int = 100, rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return rt.bus.recent(channel, limit)


# ----------------------------------------------------------- workflows
class WorkflowPrompt(BaseModel):
    prompt: str = Field(min_length=10, max_length=8000)
    kind: Literal["computer_use", "secops", "custom"] | None = None


class ActivateRequest(BaseModel):
    draft_id: str | None = None
    spec: DomainSpec | None = None  # user-edited spec (overrides the draft)
    bootstrap: bool = True
    per_scenario: int | None = Field(None, ge=1, le=500)


@router.post("/workflows/generate")
async def generate(req: WorkflowPrompt, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    draft = await generate_safe(rt.workflows, req.prompt, req.kind)
    if draft.get("status") == "error":
        raise HTTPException(502, draft["error"])
    return draft


@router.get("/workflows/drafts")
async def drafts(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await rt.workflows.drafts()


@router.post("/workflows/activate")
async def activate(req: ActivateRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    spec = req.spec
    if spec is None:
        if not req.draft_id:
            raise HTTPException(422, "draft_id or spec required")
        draft = await rt.workflows.get(req.draft_id)
        if draft is None or not draft.get("spec"):
            raise HTTPException(404, "draft not found or invalid")
        spec = DomainSpec.model_validate(draft["spec"])
    if "ESCALATE" not in spec.supported_actions:
        raise HTTPException(422, "supported_actions must include ESCALATE")
    await rt.domains.put(spec)
    job_id = rt.factory.launch_synthesis(spec.id, None, req.per_scenario) if req.bootstrap else None
    rt.bus.publish("workflow", "activated", domain=spec.id, bootstrap_job=job_id)
    return {"domain": spec.id, "bootstrap_job": job_id}


@router.put("/domains/{domain_id}")
async def put_domain(domain_id: str, spec: DomainSpec, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    if spec.id != domain_id:
        raise HTTPException(422, "id mismatch")
    await rt.domains.put(spec.model_copy(update={"source": "user"}))
    return spec.model_dump(mode="json")


@router.delete("/domains/{domain_id}", status_code=204)
async def delete_domain(domain_id: str, rt: Runtime = Depends(get_rt)) -> None:
    try:
        await rt.domains.delete(domain_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
