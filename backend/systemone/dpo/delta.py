"""State-delta computation (Module E).

The *delta* is the immediate consequence of an action: what appeared,
disappeared or changed between the state the student acted on and the state
observed right after. It is what turns a bare "the action failed" into a
teachable signal ("clicking Save produced 'Error: email is required'").
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

ERROR_RE = re.compile(
    r"\b(error|failed|failure|invalid|incorrect|denied|forbidden|unauthori[sz]ed|required|not found|"
    r"expired|try again|unable|exception|refused|blocked|timed? ?out|problem)\b",
    re.IGNORECASE,
)
ERROR_ROLES = {"alert", "alertdialog"}


def _node_key(n: dict[str, Any]) -> tuple[str, str]:
    return (str(n.get("role", "")), str(n.get("name", "")))


def dom_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    b_nodes = before.get("viewport_tree", []) or []
    a_nodes = after.get("viewport_tree", []) or []
    b_count = Counter(_node_key(n) for n in b_nodes)
    a_count = Counter(_node_key(n) for n in a_nodes)
    added = [{"role": r, "name": n} for (r, n), c in (a_count - b_count).items() for _ in range(c)]
    removed = [{"role": r, "name": n} for (r, n), c in (b_count - a_count).items() for _ in range(c)]

    b_vals = {_node_key(n): n.get("value") for n in b_nodes}
    changed = [
        {"role": k[0], "name": k[1], "before": b_vals[k], "after": n.get("value")}
        for n in a_nodes
        if (k := _node_key(n)) in b_vals and b_vals[k] != n.get("value")
    ]
    b_states = {_node_key(n): tuple(n.get("states") or ()) for n in b_nodes}
    state_changes = [
        {"role": k[0], "name": k[1], "before": list(b_states[k]), "after": list(n.get("states") or [])}
        for n in a_nodes
        if (k := _node_key(n)) in b_states and b_states[k] != tuple(n.get("states") or ())
    ]
    errors = [
        f"{x['role']}: {x['name']}"
        for x in added
        if x["role"] in ERROR_ROLES or ERROR_RE.search(x["name"] or "")
    ]
    delta: dict[str, Any] = {
        "url_changed": before.get("url") != after.get("url"),
        "added": added,
        "removed": removed,
        "value_changes": changed,
        "state_changes": state_changes,
        "error_signals": errors,
    }
    if delta["url_changed"]:
        delta["url_before"], delta["url_after"] = before.get("url"), after.get("url")
    delta["no_effect"] = not (delta["url_changed"] or added or removed or changed or state_changes)
    return delta


def log_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """For SecOps the 'after' observation is the next event(s) from the flow."""
    changed = {k: {"before": before.get(k), "after": v} for k, v in after.items() if before.get(k) != v}
    blob = " ".join(str(v) for v in after.values())
    errors: list[str] = []
    alert = after.get("alert") or {}
    if alert:
        errors.append(f"alert: {alert.get('signature')} (severity {alert.get('severity')})")
    http = after.get("http") or {}
    if isinstance(http.get("status"), int) and http["status"] == 200 and before.get("src_ip") == after.get("src_ip"):
        # attacker request that *succeeded* after an ALLOW decision
        errors.append("request from same source succeeded (HTTP 200)")
    if ERROR_RE.search(blob):
        errors.append("error keywords in follow-up event")
    return {"changed": changed, "error_signals": errors, "same_source": before.get("src_ip") == after.get("src_ip")}


def generic_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    added = {k: v for k, v in after.items() if k not in before}
    removed = [k for k in before if k not in after]
    changed = {k: {"before": before[k], "after": v} for k, v in after.items() if k in before and before[k] != v}
    blob = " ".join(str(v) for v in list(added.values()) + [c["after"] for c in changed.values()])
    return {"added": added, "removed": removed, "changed": changed,
            "error_signals": ["error keywords in new state"] if ERROR_RE.search(blob) else []}


def compute_delta(kind: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if kind == "computer_use":
        return dom_delta(before, after)
    if kind == "secops":
        return log_delta(before, after)
    return generic_delta(before, after)


def infer_failure(kind: str, delta: dict[str, Any]) -> bool:
    """Heuristic failure detection when the client does not report an outcome."""
    if delta.get("error_signals"):
        return True
    if kind == "computer_use" and delta.get("no_effect"):
        return True  # the click/type did nothing observable
    return False
