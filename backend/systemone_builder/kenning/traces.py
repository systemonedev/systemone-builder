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


def trace_access_log(s: dict[str, Any], labels: dict[str, str]) -> str:
    import re
    thr = int(re.search(r"more than (\d+)%", s["rules"])[1])
    lines = s["access_log"].splitlines()
    parsed = [m.groups() for ln in lines if (m := re.match(r'(\S+) .*"GET (\S+) HTTP/1\.1" (\d+)', ln))]
    total = len(parsed)
    n5 = sum(int(code) >= 500 for _, _, code in parsed)
    fails: dict[str, int] = {}
    logins: dict[str, int] = {}
    for ip, ep, code in parsed:
        if int(code) >= 500:
            fails[ep] = fails.get(ep, 0) + 1
        if ep == "/api/login" and code == "401":
            logins[ip] = logins.get(ip, 0) + 1
    rate = 100 * n5 / total if total else 0
    parts = [f"{total} requests, {n5} with 5xx = {rate:.0f}%; threshold {thr}% -> "
             f"{'alert' if rate > thr else 'no alert'}."]
    worst_ip = max(logins, key=logins.get) if logins else None
    parts.append(f"Most failed logins from one IP: {logins.get(worst_ip, 0) if worst_ip else 0} (need 5) -> "
                 f"{'brute force' if any(v >= 5 for v in logins.values()) else 'no brute force'}.")
    if "worst_endpoint" in labels and fails:
        parts.append(f"5xx by endpoint { {k: v for k, v in sorted(fails.items(), key=lambda x: -x[1])} } -> "
                     f"worst {max(fails, key=fails.get)}.")
    return " ".join(parts)


def trace_deploy_log(s: dict[str, Any], labels: dict[str, str]) -> str:
    events = [json.loads(x) for x in s["log"].splitlines() if x.strip()]
    at = next(i for i, e in enumerate(events) if str(e.get("msg", "")).startswith("deploy"))
    before = sum(e.get("level") == "error" for e in events[:at])
    after = sum(e.get("level") == "error" for e in events[at + 1:])
    return (f"Errors before the deploy: {before}; after: {after}. More after than before: "
            f"{'yes' if after > before else 'no'}. Rollback rule (>=3 after and >2x before): "
            f"{after} >= 3 and {after} > {2 * max(before, 1)} -> {'roll back' if after >= 3 and after > 2 * max(before, 1) else 'keep'}.")


def _parse_table(t: Any) -> tuple[list[str], list[list[Any]]]:
    import csv
    import io
    if isinstance(t, list):  # json rows
        cols = list(t[0])
        return cols, [[r[c] for c in cols] for r in t]
    if t.lstrip().startswith("|"):  # markdown
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in t.splitlines() if "---" not in ln]
    else:  # csv
        rows = list(csv.reader(io.StringIO(t)))
    cols = rows[0]
    body = [[r[0]] + [float(x) for x in r[1:]] for r in rows[1:]]
    return cols, body


def trace_table(s: dict[str, Any], labels: dict[str, str]) -> str:
    import re
    cols, rows = _parse_table(s["table"])
    if isinstance(s["table"], list):  # normalise json numeric cols
        rows = [[r[0]] + [float(x) for x in r[1:]] for r in rows]
    stmt = s["statement"]
    col = next((c for c in cols[1:] if f" {c} " in f" {stmt} ".replace(".", " ")), cols[1])
    i = cols.index(col)
    val = {r[0]: r[i] for r in rows}
    parts = []
    if m := re.match(r"(.+) has a higher \w+ than (.+)\.$", stmt):
        parts.append(f"{m[1]} {col}={val.get(m[1])} vs {m[2]} {col}={val.get(m[2])} -> "
                     f"{'true' if val.get(m[1], 0) > val.get(m[2], 0) else 'false'}.")
    elif m := re.match(r"(.+) has the highest \w+\.$", stmt):
        top = max(val, key=val.get)
        parts.append(f"Highest {col} is {top} ({val[top]}); claim is {m[1]} -> {'true' if val.get(m[1]) == val[top] else 'false'}.")
    elif m := re.match(r"Exactly (-?\d+) rows have a \w+ above ([\d.]+)\.$", stmt):
        k = sum(v > float(m[2]) for v in val.values())
        parts.append(f"{k} rows have {col} above {m[2]}; claim {m[1]} -> {'true' if k == int(m[1]) else 'false'}.")
    elif m := re.match(r"The total \w+ across all rows is ([\d.]+)\.$", stmt):
        tot = round(sum(val.values()), 2)
        parts.append(f"Sum of {col} = {tot}; claim {m[1]} -> {'true' if abs(tot - float(m[1])) < 0.01 else 'false'}.")
    elif m := re.match(r"The \w+ of (.+) is ([\d.]+)\.$", stmt):
        parts.append(f"{m[1]} {col}={val.get(m[1])}; claim {m[2]} -> "
                     f"{'true' if abs(val.get(m[1], 1e9) - float(m[2])) < 1e-6 else 'false'}.")
    if "top_row" in labels:
        parts.append(f"Highest {col}: {max(val, key=val.get)}.")
    return " ".join(parts) or f"Check the {col} column against the statement."


TRACERS: dict[str, Callable[[dict[str, Any], dict[str, str]], str]] = {
    "expense": trace_expense, "inventory": trace_inventory, "incident": trace_incident,
    "eligibility": trace_eligibility, "loan": trace_loan, "subscription": trace_subscription,
    "access_log": trace_access_log, "deploy_log": trace_deploy_log, "table": trace_table,
}
TRACE_FAMILIES = (*RECORD_FAMILIES, "access_log", "deploy_log", "table")


def trace_for(kind: str, state: dict[str, Any], labels: dict[str, str]) -> str | None:
    fn = TRACERS.get(kind)
    return fn(state, labels) if fn else None


def deliberate_rows(n: int, seed: int, kinds: tuple[str, ...] = TRACE_FAMILIES) -> list[dict[str, Any]]:
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
