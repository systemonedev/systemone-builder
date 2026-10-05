"""Kenning API: model registry, activation, export, and engine comparison.

    GET    /kenning/status                  the Kenning server's health and served model
    GET    /kenning/models                  trained models with their held-out metrics
    GET    /kenning/models/{name}           one model
    POST   /kenning/models/{name}/activate  serve it (hot swap, persisted in active.json)
    DELETE /kenning/models/{name}           delete (not the active one)
    POST   /kenning/models/{name}/export    build the export bundle, return its metadata
    GET    /kenning/models/{name}/export    download the bundle (built on demand)
    GET    /kenning/bases                   base models for training, with their licences
    GET    /kenning/datasets                training sets (rows, sources, licences, teacher)
    GET    /kenning/clef                    whether the optional Clef teacher is running
    GET    /kenning/jobs                    data / label / train / bench jobs (newest first)
    POST   /kenning/jobs                    start one: {kind, params} (one at a time)
    GET    /kenning/jobs/{id}               one job with its log tail
    POST   /kenning/jobs/{id}/cancel        stop it (paused services are restarted)
    GET    /kenning/bench/results           benchmark runs (summaries)
    GET    /kenning/bench/results/{file}    one run with per-item answers
    GET    /systemone/engines               which engines can answer here
    POST   /systemone/compare               one request on several engines, side by side
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from systemone_builder.api.deps import get_rt
from systemone_builder.kenning import jobs as kjobs
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
            "home": str(_home(rt)), "pipeline": rt.settings.pipeline}


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


# ------------------------------------------------------- train & verify
@router.get("/kenning/bases")
async def bases() -> list[dict[str, Any]]:
    return [{"id": k, "license": v, "apache_release": k not in registry.NC_BASES,
             "recommended": k == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"}
            for k, v in registry.BASE_LICENSES.items()]


@router.get("/kenning/datasets")
async def datasets(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(kjobs.list_datasets, _home(rt))


MAX_IMPORT_BYTES = 50 * 2**20


class ImportRequest(BaseModel):
    name: str
    license: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=MAX_IMPORT_BYTES)        # the JSONL text
    questions: dict[str, Any] | None = None                    # for {state, labels} lines


@router.get("/kenning/problems")
async def problems(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(kjobs.list_problems, _home(rt))


@router.post("/kenning/problems")
async def save_problem(spec: dict[str, Any], rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    """Save a problem spec (validated) so data jobs can use it."""
    from systemone_builder.kenning.problems import builtin_problems, parse_problem

    try:
        prob = parse_problem(spec)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if prob.name in builtin_problems():
        raise HTTPException(409, f"{prob.name!r} is a built-in problem; pick another name")
    d = kjobs.problems_dir(_home(rt))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prob.name}.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return prob.summary()


@router.get("/kenning/imports")
async def imports(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    return await asyncio.to_thread(kjobs.list_imports, _home(rt))


@router.post("/kenning/imports")
async def upload_import(body: ImportRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    """Upload your own labelled JSONL; it is validated line by line before it's kept."""
    from systemone_builder.kenning import importer

    if not kjobs.SLUG.match(body.name):
        raise HTTPException(400, "name: lower-case letters, digits, '.', '_' or '-'")
    d = kjobs.imports_dir(_home(rt))
    if (d / f"{body.name}.jsonl").exists():
        raise HTTPException(409, f"an import named {body.name!r} already exists")
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{body.name}.uploading.jsonl"
    qtmp = d / f".{body.name}.uploading.questions.json"
    tmp.write_text(body.content, encoding="utf-8", newline="")  # stored exactly as uploaded
    if body.questions is not None:
        qtmp.write_text(json.dumps(body.questions), encoding="utf-8")
    try:
        rows, questions = await asyncio.to_thread(importer.load_rows, tmp, qtmp if body.questions is not None else None)
    except importer.InvalidData as exc:
        tmp.unlink(missing_ok=True)
        qtmp.unlink(missing_ok=True)
        raise HTTPException(400, str(exc).replace(tmp.name, f"{body.name}.jsonl")
                            .replace(qtmp.name, "questions")) from exc
    tmp.replace(d / f"{body.name}.jsonl")
    if body.questions is not None:
        qtmp.replace(d / f"{body.name}.questions.json")
    meta = {"rows": len(rows), "questions": sorted(questions), "license": body.license,
            "uploaded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / f"{body.name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"name": body.name, **meta}


@router.get("/kenning/clef")
async def clef(rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{rt.settings.clef_url}/health")
        return {"online": r.status_code == 200, **(r.json() if r.status_code == 200 else {})}
    except httpx.HTTPError:
        return {"online": False, "hint": "docker compose --profile clef up -d clef"}


class JobRequest(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/kenning/jobs")
async def list_jobs(limit: int = 30, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    return {"busy": rt.kenning_jobs.busy, "jobs": rt.kenning_jobs.list(min(max(limit, 1), 200))}


@router.post("/kenning/jobs", status_code=202)
async def start_job(body: JobRequest, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        return rt.kenning_jobs.start(body.kind, body.params)
    except kjobs.JobError as exc:
        raise HTTPException(409 if "running" in str(exc) or "holds the GPU" in str(exc) else 400, str(exc)) from exc


@router.get("/kenning/jobs/{job_id}")
async def get_job(job_id: str, tail: int = 200, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        return rt.kenning_jobs.get(job_id, min(max(tail, 1), 2000))
    except kjobs.JobError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError:
        raise HTTPException(404, "no such job") from None


@router.post("/kenning/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    try:
        return await rt.kenning_jobs.cancel(job_id)
    except kjobs.JobError as exc:
        raise HTTPException(409, str(exc)) from exc


RESULT_FILE = re.compile(r"^(s1-)?bench-[\w.-]+\.json$")


def _results_dir(rt: Runtime):  # noqa: ANN202
    return rt.settings.data_dir / "eval_results"


def _summaries(root) -> list[dict[str, Any]]:  # noqa: ANN001
    out = []
    for f in sorted(root.glob("*bench-*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:200]:
        if not RESULT_FILE.match(f.name):
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        suite = d.get("suite") or {}
        out.append({"file": f.name, "ts": d.get("ts") or f.stat().st_mtime, "suite": suite.get("name"),
                    "description": suite.get("description"), "gate": suite.get("gate"),
                    "items": len(suite.get("items") or []), "reports": d.get("reports") or []})
    return out


@router.get("/kenning/bench/results")
async def bench_results(rt: Runtime = Depends(get_rt)) -> list[dict[str, Any]]:
    root = _results_dir(rt)
    return await asyncio.to_thread(_summaries, root) if root.is_dir() else []


@router.get("/kenning/bench/results/{file}")
async def bench_result(file: str, rt: Runtime = Depends(get_rt)) -> dict[str, Any]:
    if not RESULT_FILE.match(file):
        raise HTTPException(400, "invalid result file name")
    f = _results_dir(rt) / file
    if not f.exists():
        raise HTTPException(404, "no such result")
    return json.loads(await asyncio.to_thread(f.read_text))


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
