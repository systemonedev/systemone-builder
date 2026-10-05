"""Refit a Kenning model's temperatures on your own labelled data.

Training fits one temperature per question type on held-out training data, so the probabilities are
honest on data like the training data. On yours they can drift. This asks the served model the
questions in your labelled file, recovers each option's score from the returned probabilities
(``score = T * log p``, up to a constant the softmax ignores) and fits the temperatures to your data.

A temperature never changes which option wins, so accuracy stays the same; only how sure the
probabilities are changes. A few hundred labelled examples per question type are enough.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.kenning.importer import load_rows
from systemone_builder.kenning.model import CONFIG_FILE, read_config
from systemone_builder.kenning.train import fit_temperatures, metrics
from systemone_builder.system_one.contract import Question

Raw = list[tuple[str, list[float], list[float]]]


def _vector(q: Question, options: list[str], t: Any) -> list[float]:
    if q.type == "noul":
        return [float(t), 1.0 - float(t)]
    if isinstance(t, dict):
        return [float(t.get(o, 0.0)) for o in options]
    key = str(t)
    return [1.0 if o == key else 0.0 for o in options]


def scores_from(answer: dict[str, Any], temperature: float) -> tuple[list[str], list[float]]:
    """(options, scores) recovered from an answer's probabilities at the given temperature."""
    if answer["type"] == "noul":
        p = min(max(float(answer["noul"]), 1e-12), 1 - 1e-12)
        return ["yes", "no"], [temperature * math.log(p), temperature * math.log(1 - p)]
    probs = answer["probabilities"]
    options = list(probs)
    return options, [temperature * math.log(max(float(probs[o]), 1e-12)) for o in options]


async def collect(url: str, rows: list[dict[str, Any]], temps: dict[str, float],
                  concurrency: int = 4) -> tuple[Raw, set[str]]:
    """Raw (type, scores, target) per question, and the model names that answered."""
    sem = asyncio.Semaphore(concurrency)
    raw: Raw = []
    models: set[str] = set()

    async def one(client: httpx.AsyncClient, row: dict[str, Any]) -> None:
        async with sem:
            r = await client.post("/v1/systemone", json={"state": row["state"], "questions": row["questions"]})
        r.raise_for_status()
        body = r.json()
        models.add(body.get("model", ""))
        for qid, t in row["targets"].items():
            q = Question.model_validate(row["questions"][qid])
            options, scores = scores_from(body["answers"][qid], temps.get(q.type, 1.0))
            if q.type == "score":  # levels come back keyed "0", "1", ...
                options = [str(o) for o in options]
            raw.append((q.type, scores, _vector(q, options, t)))

    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=120) as client:
        await asyncio.gather(*(one(client, row) for row in rows))
    return raw, models


def calibrate(model_dir: Path, data: Path, url: str) -> dict[str, Any]:
    """Fit temperatures for the model in model_dir (which must be the one served at url) on data."""
    config = read_config(model_dir)
    if not config:
        raise ValueError(f"{model_dir} has no {CONFIG_FILE}: not a Kenning model")
    name = config.get("name") or model_dir.name
    old = {"noul": 1.0, "choice": 1.0, "score": 1.0, **config.get("temperature", {})}
    qpath = data.with_name(data.stem + ".questions.json")
    rows, _ = load_rows(data, qpath if qpath.exists() else None)
    raw, models = asyncio.run(collect(url, rows, old))
    if models != {name}:
        raise ValueError(f"the model served at {url} is {', '.join(sorted(models)) or 'unknown'}, not {name}: "
                         "activate it first (Models page) so its answers are the ones being calibrated")
    new = {**old, **fit_temperatures(raw)}
    return {"model": name, "data": data.name, "questions": len(raw), "temperature_before": old,
            "temperature_after": new, "before": metrics(raw, old), "after": metrics(raw, new)}


def save(model_dir: Path, report: dict[str, Any], name: str | None = None) -> None:
    """Write a calibrate() report's temperatures into model_dir's config, keeping the old ones in its history."""
    config = read_config(model_dir)
    history = config.get("calibration_history", [])
    history.append({"temperature": report["temperature_before"],
                    "replaced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    config.update({"temperature": report["temperature_after"], "calibration_history": history,
                   "calibration": {k: report[k] for k in ("data", "questions", "before", "after")}})
    if name:
        config["name"] = name
    (model_dir / CONFIG_FILE).write_text(json.dumps(config, indent=1))
