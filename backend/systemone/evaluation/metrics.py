"""Metric primitives for the evaluation sandbox (Module G)."""

from __future__ import annotations

import math
from typing import Any

from systemone.domains.spec import DomainSpec

LATENCY_BUCKETS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300, 500, 1000]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    hist = []
    prev = -math.inf
    for edge in LATENCY_BUCKETS_MS + [math.inf]:
        hist.append({"le": None if edge == math.inf else edge, "count": sum(1 for v in values if prev < v <= edge)})
        prev = edge
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "under_100ms": sum(1 for v in values if v <= 100) / len(values),
        "histogram": hist,
    }


def calibration(confidences: list[float], correct: list[bool], bins: int = 10) -> dict[str, Any]:
    if not confidences:
        return {"ece": None, "brier": None, "bins": []}
    table = []
    ece = 0.0
    n = len(confidences)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, c in enumerate(confidences) if (lo < c <= hi) or (i == 0 and c == 0)]
        if not idx:
            table.append({"lo": lo, "hi": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        conf = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        ece += len(idx) / n * abs(conf - acc)
        table.append({"lo": lo, "hi": hi, "n": len(idx), "confidence": conf, "accuracy": acc})
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct)) / n
    return {"ece": ece, "brier": brier, "bins": table}


def _coords_match(pred: dict[str, Any] | None, exp: dict[str, Any] | None, tol_px: int) -> bool:
    if not pred:
        return False
    if exp and abs(pred.get("x", -1e9) - exp.get("x", 0)) <= tol_px and abs(pred.get("y", -1e9) - exp.get("y", 0)) <= tol_px:
        return True
    return False


def target_matches(domain: DomainSpec, pred: dict[str, Any], exp: dict[str, Any], state: dict[str, Any], tol_px: int = 24) -> bool:
    if domain.action_label(exp) == "ESCALATE":
        return True  # escalating is the whole answer; no target to compare
    if domain.kind == "computer_use":
        if exp.get("target_id") is not None:
            if pred.get("target_id") == exp["target_id"]:
                return True
            # A CLICK_XY landing inside the expected element's bbox is correct.
            if pred.get("coordinates"):
                node = next((n for n in state.get("viewport_tree", []) if n.get("id") == exp["target_id"]), None)
                if node and node.get("bbox"):
                    x, y, w, h = node["bbox"]
                    c = pred["coordinates"]
                    return x <= c.get("x", -1) <= x + w and y <= c.get("y", -1) <= y + h
            return False
        if exp.get("coordinates"):
            if _coords_match(pred.get("coordinates"), exp["coordinates"], tol_px):
                return True
            tid = pred.get("target_id")
            node = next((n for n in state.get("viewport_tree", []) if n.get("id") == tid), None) if tid is not None else None
            if node and node.get("bbox"):
                x, y, w, h = node["bbox"]
                c = exp["coordinates"]
                return x <= c["x"] <= x + w and y <= c["y"] <= y + h
            return False
        if exp.get("text") is not None and pred.get("text") != exp.get("text"):
            return False
        return True
    if domain.kind == "secops":
        return pred.get("verdict") == exp.get("verdict") and set(pred.get("target_ioc") or []) == set(exp.get("target_ioc") or [])
    keys = [k for k in exp if k not in ("confidence_score", "supported_actions", domain.action_field)]
    return all(pred.get(k) == exp.get(k) for k in keys)


def text_matches(domain: DomainSpec, pred: dict[str, Any], exp: dict[str, Any]) -> bool:
    if domain.kind == "computer_use" and exp.get("action") == "TYPE":
        return (pred.get("text") or "").strip() == (exp.get("text") or "").strip()
    return True


def full_match(domain: DomainSpec, pred: dict[str, Any] | None, expected: list[dict[str, Any]], state: dict[str, Any]) -> tuple[bool, bool]:
    """(action label matches, full action matches) against any acceptable answer."""
    if not pred:
        return False, False
    label_ok = full_ok = False
    for exp in expected:
        if domain.action_label(pred) == domain.action_label(exp):
            label_ok = True
            if target_matches(domain, pred, exp, state) and text_matches(domain, pred, exp):
                full_ok = True
                break
    return label_ok, full_ok
