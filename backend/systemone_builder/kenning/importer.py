"""Import your own labelled data as Kenning training rows.

Two line formats are accepted (JSONL, one example per line):

* training rows: ``{"state": ..., "questions": {qid: question}, "targets": {qid: target}}``, the format
  `systemone data` writes. Targets are hard (noul 0/1, a choice option, a score level index) or soft
  (noul P(yes), {option: p}, {"0": p, "1": p, ...}).
* labelled items: ``{"state": ..., "labels": {qid: label}}`` with the questions in a separate JSON file
  (``{qid: question}``), the format `systemone bench --suite <file>` reads.

Every line is checked against the System One contract before anything is written; the first errors
are reported with their line numbers. A share of the examples can be held out, with one fixed
question per id, to benchmark the trained model on this data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from systemone_builder.system_one.contract import Question

MAX_ERRORS = 10


class InvalidData(ValueError):
    """The file doesn't follow the System One contract (the message lists the first errors)."""


def _check_target(q: Question, t: Any) -> str | None:
    """None when t is a valid target for q, else what's wrong."""
    if q.type == "noul":
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1:
            return "a noul target is 0, 1 or a probability of yes"
        return None
    options = list(q.criteria) if q.type == "choice" else [str(i) for i in range(len(q.criteria or []))]
    if isinstance(t, dict):
        if not t or any(k not in options for k in t) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in t.values()):
            return f"a soft target maps options ({', '.join(options)}) to probabilities"
        if not math.isclose(sum(t.values()), 1.0, abs_tol=0.02):
            return "soft target probabilities must sum to 1"
        return None
    if q.type == "choice":
        return None if isinstance(t, str) and t in options else f"a choice target is one of: {', '.join(options)}"
    ok = isinstance(t, int) and not isinstance(t, bool) and 0 <= t < len(options)
    return None if ok else f"a score target is a level index 0-{len(options) - 1}"


def hard_label(q: Question, t: Any) -> Any:
    """The hard label of a target (argmax of a soft one), as the benchmark expects it."""
    if q.type == "noul":
        return int(float(t) >= 0.5)
    if isinstance(t, dict):
        best = max(t, key=t.get)
        return int(best) if q.type == "score" else best
    return t


def load_rows(path: Path, questions_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validated training rows from a JSONL file, and the questions by id (first definition wins).

    Raises InvalidData listing the first errors when any line is invalid.
    """
    shared: dict[str, Question] = {}
    if questions_path:
        try:
            raw = json.loads(questions_path.read_text(encoding="utf-8"))
            shared = {k: Question.model_validate(v) for k, v in raw.items()}
        except (OSError, ValueError, AttributeError, ValidationError) as exc:
            raise InvalidData(f"{questions_path.name}: not a valid questions file ({exc})") from exc
    rows: list[dict[str, Any]] = []
    questions: dict[str, dict[str, Any]] = {k: q.model_dump(exclude_none=True) for k, q in shared.items()}
    errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            if len(errors) >= MAX_ERRORS:
                break
            try:
                d = json.loads(line)
            except ValueError:
                errors.append(f"line {n}: not JSON")
                continue
            if not isinstance(d, dict) or not isinstance(d.get("state"), (str, dict)) or not d["state"]:
                errors.append(f"line {n}: needs a non-empty 'state' (text or object)")
                continue
            if "targets" in d:
                try:
                    qs = {k: Question.model_validate(v) for k, v in (d.get("questions") or {}).items()}
                except (ValidationError, AttributeError) as exc:
                    errors.append(f"line {n}: invalid question ({str(exc).splitlines()[0]})")
                    continue
                targets = d["targets"]
            elif "labels" in d:
                if not shared:
                    errors.append(f"line {n}: 'labels' rows need a questions file")
                    continue
                qs, targets = shared, d["labels"]
            else:
                errors.append(f"line {n}: needs 'questions' + 'targets', or 'labels' with a questions file")
                continue
            if not isinstance(targets, dict) or not targets:
                errors.append(f"line {n}: no targets")
                continue
            bad = next(((k, msg) for k, t in targets.items()
                        for msg in [f"unknown question {k!r}" if k not in qs else _check_target(qs[k], t)] if msg), None)
            if bad:
                errors.append(f"line {n}: {bad[0]}: {bad[1]}")
                continue
            used = {k: qs[k] for k in targets}
            for k, q in used.items():
                questions.setdefault(k, q.model_dump(exclude_none=True))
            rows.append({"state": d["state"], "questions": {k: q.model_dump(exclude_none=True) for k, q in used.items()},
                         "targets": targets})
    if errors:
        raise InvalidData(f"{path.name}: " + "; ".join(errors))
    if not rows:
        raise InvalidData(f"{path.name}: no examples")
    return rows, questions


def holdout_items(rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Benchmark items for held-out rows, asked with the fixed question per id."""
    fixed = {k: Question.model_validate(v) for k, v in questions.items()}
    items = []
    for i, r in enumerate(rows):
        # only ids asked exactly as the fixed question: a reworded question is a different test
        labels = {k: hard_label(fixed[k], t) for k, t in r["targets"].items() if r["questions"][k] == questions[k]}
        if labels:
            items.append({"id": f"{key}-{i}", "state": r["state"], "labels": labels})
    return items
