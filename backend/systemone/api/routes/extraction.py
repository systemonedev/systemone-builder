"""Phase 2 API: domain contracts and state extraction / fuzzy scrubbing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from systemone.api.deps import get_rt
from systemone.extraction.dom import BROWSER_COLLECTOR_JS
from systemone.extraction.pipeline import Observation
from systemone.runtime import Runtime

router = APIRouter(tags=["extraction"])


def _domain_or_404(rt: Runtime, domain_id: str):
    try:
        return rt.domains.get(domain_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/domains")
async def list_domains(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return [
        {"id": d.id, "name": d.name, "kind": d.kind, "threshold": d.threshold, "source": d.source,
         "supported_actions": d.supported_actions, "description": d.description}
        for d in rt.domains.all()
    ]


@router.get("/domains/{domain_id}")
async def get_domain(domain_id: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return _domain_or_404(rt, domain_id).model_dump(mode="json")


@router.post("/extract/{domain_id}")
async def extract(domain_id: str, obs: Observation, session_id: str = "default", rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    _domain_or_404(rt, domain_id)
    ex = rt.extractor(domain_id)
    try:
        res = ex.extract(obs)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from None
    messages, _ = ex.prompts.messages(res.state)
    overlap = rt.prefix.observe(f"{domain_id}:{session_id}", "".join(m["content"] for m in messages))
    return {
        "state_input": res.state,
        "state_hash": res.state_hash,
        "canonical": res.canonical,
        "static_prefix_hash": ex.prompts.static_prefix_hash,
        "prefix_overlap": overlap,
        "element_index": res.index,
        "entities": res.vault.to_dict() if res.vault else {},
        "validation_errors": res.validation_errors,
    }


@router.get("/extract/collector.js", response_class=PlainTextResponse)
async def collector_js() -> str:
    """Browser-side element collector for Playwright (``page.evaluate``)."""
    return BROWSER_COLLECTOR_JS
