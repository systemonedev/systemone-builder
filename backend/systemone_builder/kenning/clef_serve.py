"""Serve Cloudflare's Clef (Apache-2.0) on the System One wire format.

Clef ships its own loader and a ``systemone(model, processor, request)`` function
that answers a ``POST /v1/systemone`` body directly, so this server is a thin
wrapper. The builder uses it as a benchmark engine (``--engines clef``) and as a
teacher for distillation (``systemone label``).

    CLEF_MODEL   Hugging Face repo (default Cloudflare/clef-flash, ~19 GB bf16)
    CLEF_MAX_LENGTH  tokens per request (default 4096: keeps a 24 GB GPU enough)

The model code (``joint_schema_model.py``) comes from the model repository itself.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException

_state: dict[str, Any] = {}
_lock = threading.Lock()


def load() -> None:
    import torch
    from huggingface_hub import snapshot_download

    repo = os.environ.get("CLEF_MODEL", "Cloudflare/clef-flash")
    t0 = time.time()
    path = snapshot_download(repo)
    sys.path.insert(0, path)
    from joint_schema_model import load_release_model, systemone  # noqa: E402 - shipped with the weights

    torch.backends.cuda.matmul.allow_tf32 = False
    model, processor = load_release_model(path, device="cuda")
    _state.update(model=model, processor=processor, systemone=systemone, repo=repo, path=path)
    print(f"[clef-serve] loaded {repo} in {time.time() - t0:.0f}s; "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(load)
    yield


app = FastAPI(title="Clef (System One wire format)", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": _state.get("repo")}


def _answer(body: dict[str, Any]) -> dict[str, Any]:
    body = {**body, "model": body.get("model") or _state["repo"]}
    max_length = int(os.environ.get("CLEF_MAX_LENGTH", "4096"))
    t0 = time.perf_counter()
    with _lock:  # one request at a time: a 9B model on one consumer GPU
        out = _state["systemone"](_state["model"], _state["processor"], body, max_length=max_length)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def _clef_question(q: dict[str, Any]) -> dict[str, Any]:
    # Clef's noul criteria must be a dict ({"true": ..., "false": ...}); the wire
    # format also allows a plain string, which Clef cannot take: drop it.
    if q.get("type") == "noul" and not isinstance(q.get("criteria"), dict):
        return {k: v for k, v in q.items() if k != "criteria"}
    return q


def _answer_batch(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Many requests in one padded forward pass (much faster for labelling datasets)."""
    import torch
    from joint_schema_model import collate_records, encode_record, systemone_answer  # noqa: E402

    max_length = int(os.environ.get("CLEF_MAX_LENGTH", "4096"))
    model, processor = _state["model"], _state["processor"]
    reqs = [{**b, "model": b.get("model") or _state["repo"],
             "questions": {k: _clef_question(q) for k, q in b["questions"].items()}} for b in bodies]
    t0 = time.perf_counter()
    with _lock, torch.inference_mode():
        encoded = [encode_record(processor.tokenizer, r, max_length=max_length, processor=processor) for r in reqs]
        logits = model(collate_records(encoded, processor.tokenizer.pad_token_id, next(model.parameters()).device))
    out = []
    for r, enc, rec_logits in zip(reqs, encoded, logits):
        answers = {q.question_id: systemone_answer(r["questions"][q.question_id],
                                                   dict(zip(q.option_ids, ql.float().softmax(-1).tolist())))
                   for q, ql in zip(enc.questions, rec_logits)}
        out.append({"model": r["model"], "answers": answers,
                    "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}})
    per = (time.perf_counter() - t0) * 1000 / max(len(out), 1)
    for o in out:
        o["latency_ms"] = per
    return out


@app.post("/v1/systemone/batch")
async def system_one_batch(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"responses": await asyncio.to_thread(_answer_batch, body["requests"])}
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/systemone")
async def system_one(body: dict[str, Any]) -> dict[str, Any]:
    body = {**body, "questions": {k: _clef_question(q) for k, q in (body.get("questions") or {}).items()}}
    try:
        return await asyncio.to_thread(_answer, body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
