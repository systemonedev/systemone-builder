"""TypeSafe WorkflowEvals as a System One benchmark suite -- BENCHMARK ONLY, never trained on.

Four open eval datasets (Apache-2.0) from TypeSafe's WorkflowEvals, each a workflow of typed
decisions over a structured case:

    invoice           accounts-payable review of an invoice packet
    customer_service  support-ticket decisions (intent, action, sentiment)
    security          security-incident triage (severity, scope, confirmation)
    agent_trace       agent-trace observability (what went wrong, where, how bad)

We score **per-question agreement with the consensus reference** -- the mean of OpenAI and
Anthropic high-reasoning answers published with the datasets -- which is directly comparable
across engines on our own scorer. Two honest caveats, stated wherever the numbers are shown:

* the labels are **model-consensus "silver"**, not ground truth (a model that disagrees with
  GPT+Claude is not necessarily wrong), and
* this is NOT TypeSafe's case-level action-set metric (that needs their workflow policy engine);
  it is per-decision accuracy.

``question_json`` is case-specific, so each question-row is its own single-question item; metrics
aggregate by workflow (the four datasets are the "families"). Fetched from the HF datasets-server
at run time with a fixed seed and cached; nothing is redistributed.

    systemone bench --suite workflowevals --engines kenning,clef,glide -n 120   # n per workflow
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.system_one.contract import Question

WORKFLOWS = {
    "invoice": "typesafe/evalsafe-invoice-processing",
    "customer_service": "typesafe/evalsafe-customer-service",
    "security": "typesafe/evalsafe-security-incidents",
    "agent_trace": "typesafe/evalsafe-agent-trace-observability",
}


def _target(kind: str, consensus: dict[str, Any]) -> Any:
    """The hard reference label from the consensus distribution (argmax)."""
    probs = {p["option"]: float(p["probability"]) for p in (consensus.get("probabilities") or [])}
    if not probs:
        return None
    if kind == "noul":
        return 1 if probs.get("true", 0.0) >= probs.get("false", 0.0) else 0
    winner = max(probs, key=probs.get)
    if kind == "choice":
        return winner
    try:  # score: options are level strings "0".."k"
        return int(winner)
    except (TypeError, ValueError):
        return None


async def fetch_workflowevals(n_per_workflow: int, seed: int) -> list[dict[str, Any]]:
    from systemone_builder.system_one.bench import hf_get

    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=90) as client:
        for wf, ds in WORKFLOWS.items():
            meta = await hf_get(client, "/rows", {"dataset": ds, "config": "questions", "split": "test",
                                                  "offset": 0, "length": 1})
            total = meta["num_rows_total"]
            pages = list(range(0, max(1, total), 100))
            rng.shuffle(pages)
            got = 0
            for off in pages:
                if got >= n_per_workflow:
                    break
                rows = (await hf_get(client, "/rows", {"dataset": ds, "config": "questions", "split": "test",
                                                       "offset": off, "length": 100}))["rows"]
                for rr in rows:
                    if got >= n_per_workflow:
                        break
                    r = rr["row"]
                    cons = r.get("consensus") or {}
                    if cons.get("status") != "answered":
                        continue
                    try:
                        spec = json.loads(r["question_json"])
                        q = Question.model_validate(spec)
                        state = json.loads(r["state_json"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    tgt = _target(r["kind"], cons)
                    if tgt is None:
                        continue
                    if q.type == "choice" and tgt not in (q.criteria or {}):
                        continue
                    qid = f"{wf}::{r['question_instance_id']}"
                    items.append({"id": qid, "state": state, "labels": {qid: tgt}, "question": spec})
                    got += 1
    rng.shuffle(items)
    return items


async def workflowevals_suite(n_per_workflow: int, seed: int, cache_dir: Path | None):
    from systemone_builder.system_one.bench import Item, Suite

    cache = cache_dir / f"workflowevals-n{n_per_workflow}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = await fetch_workflowevals(n_per_workflow, seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
    questions = {x["id"]: Question.model_validate(x["question"]) for x in raw}
    items = [Item(id=x["id"], state=x["state"], labels=x["labels"]) for x in raw]
    wfs = sorted({qid.split("::")[0] for qid in questions})
    desc = (f"TypeSafe WorkflowEvals: per-question agreement with the OpenAI+Anthropic consensus "
            f"reference over {len(wfs)} workflows ({', '.join(wfs)}); {len(items)} items (seed {seed}). "
            f"Silver labels, benchmark-only; not TypeSafe's case-level action metric.")
    return Suite("workflowevals", desc, questions, items, gate=None)


def workflow_averages(report: dict[str, Any]) -> dict[str, float]:
    """Macro accuracy per workflow (score questions: exact level)."""
    acc: dict[str, list[float]] = {}
    for qid, qm in report.get("questions", {}).items():
        if not qm.get("n"):
            continue
        v = qm.get("accuracy", qm.get("exact_level"))
        if v is not None:
            acc.setdefault(qid.split("::")[0], []).append(v)
    return {w: sum(v) / len(v) for w, v in sorted(acc.items())}
