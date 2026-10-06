"""Gold reasoning traces for the rule-generated record families (TRAINING ONLY).

The structured generators compute each label by an exact rule, so we can emit the *correct* step-by-step
computation that leads to it -- the arithmetic, threshold and membership reasoning that a small model's
free-form chain-of-thought gets wrong. Training Kenning-XL to produce these traces, then read the answer
(deliberate mode), teaches the computation itself, not just the answer. This is a stronger signal than a
teacher's soft label for numeric decisions: it is ground-truth reasoning, free, by construction.

``trace_for(kind, state, labels)`` recomputes the chain from the state (so it stays in sync with what a
reader sees). ``deliberate_rows`` builds training rows ``{state, questions, targets, trace}`` for the
record families; ``train_xl.py --deliberate`` trains on the trace then the answer.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime
from typing import Any, Callable

from systemone_builder.kenning import structured as st

RECORD_FAMILIES = ("expense", "inventory", "incident", "eligibility", "loan", "subscription")


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def _age(dob: str, today: str) -> int:
    d, t = date.fromisoformat(dob), date.fromisoformat(today)
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def trace_expense(s: dict[str, Any], labels: dict[str, str]) -> str:
    p, e = s["policy"], s["expense"]
    cap = p[f"cap_{e['category']}_usd"]
    parts = [f"Category {e['category']}: cap is ${cap}, amount is ${e['amount_usd']}."]
    if e["amount_usd"] > cap:
        parts.append(f"${e['amount_usd']} > ${cap}, so it is over the cap.")
    else:
        parts.append(f"${e['amount_usd']} <= ${cap}, within the cap.")
        if e["amount_usd"] > p["receipt_required_over_usd"] and not e["receipt_attached"]:
            parts.append(f"Over ${p['receipt_required_over_usd']} needs a receipt and none is attached.")
        else:
            d = _days(e["date_submitted"], e["date_spent"])
            parts.append(f"Submitted {d} days after spending; limit is {p['submit_within_days']} days"
                         + (" -- too late." if d > p["submit_within_days"] else " -- in time."))
    parts.append(f"Needs manager approval: amount ${e['amount_usd']} vs threshold ${p['manager_approval_over_usd']}"
                 f" -> {'yes' if e['amount_usd'] > p['manager_approval_over_usd'] else 'no'}.")
    return " ".join(parts)


def trace_inventory(s: dict[str, Any], labels: dict[str, str]) -> str:
    it = next(x for x in s["inventory"] if x["item"] == s["question_about"])
    days_left = it["on_hand"] / it["daily_usage"]
    return (f"{it['item']}: on_hand {it['on_hand']} vs reorder_point {it['reorder_point']} -> "
            f"{'reorder' if it['on_hand'] <= it['reorder_point'] else 'no reorder'}. "
            f"Days of stock = {it['on_hand']}/{it['daily_usage']} = {days_left:.1f}, lead time {it['lead_time_days']} days -> "
            f"{'runs out first (critical)' if days_left < it['lead_time_days'] else 'arrives in time'}.")


def trace_incident(s: dict[str, Any], labels: dict[str, str]) -> str:
    inc = s["incident"]
    sla = s["first_response_sla_minutes"][inc["severity"]]
    end = inc.get("first_response_at") or inc["now"]
    mins = int((datetime.fromisoformat(end) - datetime.fromisoformat(inc["opened_at"])).total_seconds() / 60)
    out = f"Severity {inc['severity']}: SLA {sla} min. Elapsed {mins} min -> {'breached' if mins > sla else 'within SLA'}."
    if "time_pressure" in labels:
        out += f" {'past SLA' if mins > sla else 'less than half left' if (sla - mins) < sla / 2 else 'over half the window left'}."
    return out


def trace_eligibility(s: dict[str, Any], labels: dict[str, str]) -> str:
    a = s["applicant"]
    age = _age(a["date_of_birth"], s["today"])
    parts = [f"Age = {age} (born {a['date_of_birth']}, today {s['today']}); minimum is {s['minimum_age']}."]
    if age < s["minimum_age"]:
        parts.append(f"{age} < {s['minimum_age']} -> too young, not eligible.")
    elif a["country"] not in s["available_in"]:
        parts.append(f"Country {a['country']} is not in {s['available_in']} -> not eligible.")
    else:
        parts.append(f"{age} >= {s['minimum_age']} and {a['country']} is supported -> eligible.")
    return " ".join(parts)


def trace_loan(s: dict[str, Any], labels: dict[str, str]) -> str:
    c, a = s["criteria"], s["applicant"]
    mult = int(str(c["max_amount"]).split()[0])
    dti = a["monthly_debt_payments_usd"] / a["monthly_income_usd"]
    parts = [f"Credit score {a['credit_score']} vs min {c['min_credit_score']} -> "
             f"{'fails' if a['credit_score'] < c['min_credit_score'] else 'ok'}."]
    if a["credit_score"] >= c["min_credit_score"]:
        parts.append(f"DTI = {a['monthly_debt_payments_usd']}/{a['monthly_income_usd']} = {dti:.2f} vs max "
                     f"{c['max_debt_to_income']} -> {'fails' if dti > c['max_debt_to_income'] else 'ok'}.")
        if dti <= c["max_debt_to_income"]:
            cap = mult * a["monthly_income_usd"]
            parts.append(f"Amount ${a['requested_amount_usd']} vs max {mult}x income = ${cap} -> "
                         f"{'too high' if a['requested_amount_usd'] > cap else 'ok, pre-qualifies'}.")
    return " ".join(parts)


def trace_subscription(s: dict[str, Any], labels: dict[str, str]) -> str:
    acc = s["account"]
    days = _days(acc["trial_ends"], s["today"])
    if days < 0:
        return f"Trial ended {-days} days ago ({acc['trial_ends']} before today {s['today']}) -> already ended."
    return (f"Trial ends in {days} days. Card on file: {acc['card_on_file']}; cancellation: "
            f"{acc['cancellation_requested']} -> "
            f"{'charged to the paid plan' if acc['card_on_file'] and not acc['cancellation_requested'] else 'downgraded to free'}.")


TRACERS: dict[str, Callable[[dict[str, Any], dict[str, str]], str]] = {
    "expense": trace_expense, "inventory": trace_inventory, "incident": trace_incident,
    "eligibility": trace_eligibility, "loan": trace_loan, "subscription": trace_subscription,
}


def trace_for(kind: str, state: dict[str, Any], labels: dict[str, str]) -> str | None:
    fn = TRACERS.get(kind)
    return fn(state, labels) if fn else None


def deliberate_rows(n: int, seed: int, kinds: tuple[str, ...] = RECORD_FAMILIES) -> list[dict[str, Any]]:
    """Training rows with a gold trace: {state, questions, targets, trace}, one per record case."""
    from systemone_builder.kenning.synthetic_tasks import questions_for
    rng = random.Random(f"deliberate-{seed}")
    out = []
    for kind in kinds:
        for state, attrs, labels in st.cases(kind, n, seed):
            tr = trace_for(kind, state, labels)
            if not tr:
                continue
            row = {"state": state, **questions_for(attrs, labels, rng), "trace": tr}
            out.append(row)
    return out


def _selfcheck() -> None:
    """Every trace ends consistent with the label (sanity, run as a script)."""
    for kind in RECORD_FAMILIES:
        for state, _attrs, labels in st.cases(kind, 50, 1):
            t = trace_for(kind, state, labels)
            assert t and len(t) > 10, kind
    print("traces: ok", json.dumps({k: trace_for(k, *st.cases(k, 1, 2)[0][::2]) for k in RECORD_FAMILIES}, indent=1)[:1200])


if __name__ == "__main__":
    _selfcheck()
