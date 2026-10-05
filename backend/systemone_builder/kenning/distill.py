"""Distillation: label training rows with a teacher System One model.

Sends rows (state + questions) to a teacher speaking the wire format - by default
the local Clef server's batch endpoint - and writes the rows back with soft
targets: the teacher's probability of "yes" for a noul, its distribution over the
options for a choice or score. A soft target carries the teacher's uncertainty,
which is what makes a student's probabilities calibrated, not just its answers.

``alpha`` blends teacher and existing labels: 1.0 replaces them, 0.5 averages. Sources whose labels
are exact by construction (rule-generated, ``EXACT_SOURCES``) keep them; the teacher is still asked, so
its agreement with the rules is reported.
Rows whose questions the teacher could not answer keep their labels. The
original labels are kept under ``label_targets``, and agreement between teacher
and labels is reported per source (a cheap check of label quality).

Only distil from teachers whose terms allow it (Clef: Apache-2.0). TypeSafe's
terms forbid training on Jev outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


def soft_target(question: dict[str, Any], answer: dict[str, Any]) -> Any:
    if question["type"] == "noul":
        return float(answer["noul"])
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def as_distribution(question: dict[str, Any], target: Any) -> dict[str, float]:
    """Any target (hard or soft) as {option: p} (noul: {"yes", "no"})."""
    if question["type"] == "noul":
        p = float(target)
        return {"yes": p, "no": 1 - p}
    if isinstance(target, dict):
        s = sum(target.values()) or 1.0
        return {str(k): v / s for k, v in target.items()}
    if question["type"] == "choice":
        return {str(o): float(str(o) == str(target)) for o in question["criteria"]}
    return {str(i): float(i == int(target)) for i in range(len(question["criteria"]))}


def blend(question: dict[str, Any], label: Any, teacher: Any, alpha: float) -> Any:
    if alpha >= 1.0 or label is None:
        return teacher
    a, b = as_distribution(question, label), as_distribution(question, teacher)
    mixed = {k: (1 - alpha) * a.get(k, 0.0) + alpha * b.get(k, 0.0) for k in set(a) | set(b)}
    return mixed["yes"] if question["type"] == "noul" else mixed


def _top(d: dict[str, float]) -> str:
    return max(d, key=d.__getitem__)


EXACT_SOURCES = ("structured:",)  # labels computed by rules: the teacher is asked (agreement) but not blended in


def label_rows(rows: list[dict[str, Any]], teacher_url: str, alpha: float = 1.0, batch: int = 16,
               timeout_s: float = 600.0, progress: Any = None,
               exact: tuple[str, ...] = EXACT_SOURCES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"questions": 0, "agree": 0, "failed": 0})
    out: list[dict[str, Any]] = []
    with httpx.Client(base_url=teacher_url.rstrip("/"), timeout=timeout_s) as client:
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            reqs = [{"state": r["state"], "questions": r["questions"]} for r in chunk]
            try:
                r = client.post("/v1/systemone/batch", json={"requests": reqs})
                r.raise_for_status()
                answers = [x["answers"] for x in r.json()["responses"]]
            except (httpx.HTTPError, KeyError, ValueError):
                answers = [None] * len(chunk)
            for row, ans in zip(chunk, answers):
                src = row.get("source", "?")
                new = dict(row)
                new["label_targets"] = row.get("targets", {})
                targets = dict(row.get("targets", {}))
                for qid, q in row["questions"].items():
                    stats[src]["questions"] += 1
                    if not ans or qid not in ans:
                        stats[src]["failed"] += 1
                        continue
                    t = soft_target(q, ans[qid])
                    label = row.get("targets", {}).get(qid)
                    if label is not None and _top(as_distribution(q, label)) == _top(as_distribution(q, t)):
                        stats[src]["agree"] += 1
                    if not (label is not None and src.startswith(exact)):
                        targets[qid] = blend(q, label, t, alpha)
                new["targets"] = targets
                out.append(new)
            if progress:
                progress(min(start + batch, len(rows)), len(rows))
    report = {src: {**v, "agreement": round(v["agree"] / max(v["questions"] - v["failed"], 1), 4)}
              for src, v in stats.items()}
    return out, report


def label_file(src: Path, dst: Path, teacher_url: str, teacher_name: str, alpha: float = 1.0, batch: int = 16,
               limit: int | None = None, exact: tuple[str, ...] = EXACT_SOURCES,
               skip_exact: bool = True) -> dict[str, Any]:
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    if limit:
        rows = rows[:limit]
    # exact-label rows (rule-generated) don't need the teacher at all: pass them through untouched and
    # keep the teacher's GPU time for the rows whose calibration it actually improves. These are also
    # the long (log/table) rows that dominate labelling time.
    passthrough: list[dict[str, Any]] = []
    if skip_exact:
        to_label, passthrough = [], []
        for r in rows:
            (passthrough if str(r.get("source", "")).startswith(exact) else to_label).append(r)
        rows = to_label
        print(f"[kenning-label] {len(passthrough)} exact-label rows pass through; {len(rows)} to the teacher",
              flush=True)

    def progress(done: int, total: int) -> None:
        if done == total or done % (batch * 50) == 0:
            print(f"[kenning-label] {done}/{total} rows", flush=True)

    labelled, report = label_rows(rows, teacher_url, alpha, batch, progress=progress, exact=exact)
    labelled += passthrough
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r) + "\n" for r in labelled))
    manifest_src = src.with_suffix(".manifest.json")
    manifest = json.loads(manifest_src.read_text()) if manifest_src.exists() else {"sources": {}}
    manifest["teacher"] = {"model": teacher_name, "url": teacher_url, "alpha": alpha, "agreement": report,
                           "labels_kept_for": list(exact)}
    dst.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    return report
