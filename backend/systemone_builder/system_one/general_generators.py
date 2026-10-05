"""Generated benchmark cases with exact, rule-derived labels (BENCHMARK ONLY).

Structured records, long logs and agent steps have no good permissive public datasets, so these
cases are generated. Every label is computed from the case by an explicit rule, not by a model, so
the benchmark has no bias toward any teacher (Clef, Qwen, Jev).

Never reuse these generators, their templates or their seeds for training data: that would
contaminate the benchmark. Training generators live in systemone_builder.kenning.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any

Case = dict[str, Any]  # {"state": ..., "labels": {qid: label}}

# ------------------------------------------------------------------ refunds
REFUND_REASONS = ("eligible", "outside_return_window", "excluded_category", "item_used")
CATEGORIES = ("electronics", "clothing", "books", "digital_download", "groceries", "toys", "furniture")
EXCLUDED = ("digital_download", "groceries")


def refund_case(rng: random.Random) -> Case:
    window = rng.choice((14, 30, 60))
    today = date(2026, 9, 1) + timedelta(days=rng.randrange(0, 60))
    reason = rng.choice(REFUND_REASONS)
    category = rng.choice(EXCLUDED) if reason == "excluded_category" else rng.choice(
        [c for c in CATEGORIES if c not in EXCLUDED])
    days_since = rng.randrange(window + 1, window + 40) if reason == "outside_return_window" else rng.randrange(0, window + 1)
    delivered = today - timedelta(days=days_since)
    used = reason == "item_used"
    state = {
        "policy": {"return_window_days": window, "excluded_categories": list(EXCLUDED),
                   "condition": "items must be unused"},
        "order": {"id": f"ORD-{rng.randrange(10**5, 10**6)}", "category": category,
                  "price_usd": round(rng.uniform(5, 900), 2), "delivered_on": delivered.isoformat(),
                  "item_condition": "opened and used" if used else rng.choice(("unopened", "opened, unused"))},
        "request": {"type": "refund", "date": today.isoformat()},
    }
    return {"state": state, "labels": {"refund_eligible": int(reason == "eligible"), "refund_reason": reason}}


# ------------------------------------------------------------ transactions
COUNTRIES = ("US", "GB", "DE", "FR", "BR", "JP", "MX", "CA")


def transactions_case(rng: random.Random) -> Case:
    home = rng.choice(COUNTRIES)
    limit = rng.choice((500, 1000, 2500, 5000))
    over = rng.random() < 0.5
    foreign = rng.random() < 0.5
    n = rng.randrange(3, 9)
    target = limit * rng.uniform(1.05, 1.6) if over else limit * rng.uniform(0.3, 0.95)
    amounts = [rng.random() + 0.2 for _ in range(n)]
    s = sum(amounts)
    amounts = [round(a / s * target, 2) for a in amounts]
    countries = [home] * n
    if foreign:
        for i in rng.sample(range(n), rng.randrange(1, min(3, n) + 1)):
            countries[i] = rng.choice([c for c in COUNTRIES if c != home])
    t0 = datetime(2026, 9, 14, 7, 0)
    txs = [{"time": (t0 + timedelta(minutes=37 * i + rng.randrange(30))).strftime("%H:%M"), "amount_usd": a,
            "merchant_country": c} for i, (a, c) in enumerate(zip(amounts, countries))]
    total = round(sum(amounts), 2)
    state = {"customer": {"home_country": home, "daily_spending_limit_usd": limit},
             "transactions_today": txs}
    return {"state": state, "labels": {"over_daily_limit": int(total > limit),
                                       "foreign_transaction": int(any(c != home for c in countries))}}


# ------------------------------------------------------------------ access
ROLES = ("intern", "engineer", "manager", "director", "contractor")
LEVEL = {"intern": 0, "contractor": 0, "engineer": 1, "manager": 2, "director": 3}
SENSITIVITY = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def access_case(rng: random.Random) -> Case:
    role = rng.choice(ROLES)
    sens = rng.choice(list(SENSITIVITY))
    own_dept = rng.choice(("finance", "engineering", "sales", "legal"))
    user_dept = own_dept if rng.random() < 0.6 else rng.choice(("finance", "engineering", "sales", "legal"))
    # Rule: public -> everyone; internal -> any non-contractor; confidential -> same department and
    # level >= 1; restricted -> same department and level >= 2.
    s = SENSITIVITY[sens]
    allowed = (s == 0 or (s == 1 and role != "contractor")
               or (s == 2 and user_dept == own_dept and LEVEL[role] >= 1)
               or (s == 3 and user_dept == own_dept and LEVEL[role] >= 2))
    state = {
        "policy": ["public: anyone may read",
                   "internal: any employee except contractors may read",
                   "confidential: only the owning department, engineer level or above",
                   "restricted: only the owning department, manager level or above"],
        "levels": "intern = contractor < engineer < manager < director",
        "user": {"role": role, "department": user_dept},
        "resource": {"name": rng.choice(("q3-forecast.xlsx", "payroll-2026.csv", "roadmap.md", "contract-acme.pdf",
                                         "incident-1142.md")), "sensitivity": sens, "owning_department": own_dept},
    }
    return {"state": state, "labels": {"access_allowed": int(allowed)}}


# ---------------------------------------------------------------- tickets
TIERS = ("free", "pro", "enterprise")


def ticket_case(rng: random.Random) -> Case:
    tier = rng.choice(TIERS)
    sla = {"free": 72, "pro": 24, "enterprise": 4}[tier]
    age = rng.choice((rng.uniform(0.1, sla * 0.5), rng.uniform(sla * 0.5, sla), rng.uniform(sla * 1.05, sla * 3)))
    outage = rng.random() < 0.25
    overdue = age > sla
    # Rule: high = production outage, or overdue enterprise; medium = any other overdue ticket, or an
    # enterprise ticket past half its SLA; low otherwise.
    level = 2 if outage or (overdue and tier == "enterprise") else 1 if overdue or (
        tier == "enterprise" and age > sla / 2) else 0
    created = datetime(2026, 9, 20, 9, 0)
    state = {"ticket": {"customer_tier": tier, "sla_first_response_hours": sla,
                        "created_at": created.isoformat(timespec="minutes"),
                        "now": (created + timedelta(hours=age)).isoformat(timespec="minutes"),
                        "status": "awaiting first response",
                        "summary": "Production API is down for all users" if outage else
                        rng.choice(("Question about invoice line items", "Feature request: CSV export",
                                    "Login works but dashboard is slow", "How do I rotate an API key?"))},
             "priority_rules": ["high: a production outage, or an overdue enterprise ticket",
                                "medium: any other overdue ticket, or an enterprise ticket past half its SLA",
                                "low: everything else"]}
    return {"state": state, "labels": {"ticket_priority": level}}


# -------------------------------------------------------------------- logs
SERVICES = ("auth", "payments", "search", "inventory", "notifications")
OK_MSG = {"auth": "token issued user={u}", "payments": "charge captured amount={a}", "search": "query ok hits={h}",
          "inventory": "stock read sku={s}", "notifications": "email queued to={u}"}
ERR_MSG = {"auth": "token validation failed: upstream timeout", "payments": "charge declined: gateway 502",
           "search": "index shard unavailable", "inventory": "db connection refused",
           "notifications": "smtp relay timeout"}


def logs_case(rng: random.Random) -> Case:
    # 60-140 lines (at most ~2.5k tokens) so the whole log fits a 4k context: a truncated log would
    # hide the last minutes, where the answer is
    n = rng.randrange(60, 140)
    failing = rng.choice(SERVICES) if rng.random() < 0.5 else None
    start = datetime(2026, 9, 21, 14, 0, 0)
    span = 10 * 60
    lines, recent = [], {s: [0, 0] for s in SERVICES}  # last 5 minutes: [total, errors]
    for i in range(n):
        t = start + timedelta(seconds=int(span * i / n))
        svc = rng.choice(SERVICES)
        in_window = (t - start).total_seconds() >= span - 300
        # background errors stay well under 10%; the failing service errors heavily in the last 5 minutes
        p_err = 0.3 if (failing == svc and in_window) else 0.01
        err = rng.random() < p_err
        if in_window:
            recent[svc][0] += 1
            recent[svc][1] += int(err)
        msg = ERR_MSG[svc] if err else OK_MSG[svc].format(u=f"u{rng.randrange(999)}", a=rng.randrange(5, 500),
                                                          h=rng.randrange(0, 90), s=f"SKU{rng.randrange(9999)}")
        lines.append(f"{t.strftime('%H:%M:%S')} {'ERROR' if err else 'INFO '} [{svc}] {msg}")
    # ground truth from the generated log itself, not from the intent
    rates = {s: (e / t if t else 0.0) for s, (t, e) in recent.items()}
    bad = [s for s, r in rates.items() if r > 0.10 and recent[s][0] >= 5]
    labels: dict[str, Any] = {"service_failing": int(bool(bad))}
    labels["failing_service"] = max(bad, key=rates.get) if bad else "none"
    state = {"window": "last 10 minutes, newest last", "now": (start + timedelta(seconds=span)).strftime("%H:%M:%S"),
             "rule": "a service is failing when more than 10% of its requests in the last 5 minutes are errors",
             "log": "\n".join(lines)}
    return {"state": state, "labels": labels}


# ------------------------------------------------------------- agent steps
def agent_steps_case(rng: random.Random) -> Case:
    people, hour = rng.randrange(2, 9), rng.randrange(17, 22)
    restaurant = rng.choice(("Luigi's", "Sakura", "The Oak Room", "Casa Verde"))
    goal = f"Book a table for {people} at {restaurant} tonight at {hour}:00."
    outcome = rng.choice(("done", "wrong_time", "wrong_size", "failed", "not_finished"))
    steps = [{"tool": "search_restaurants", "args": {"name": restaurant}, "result": {"found": True, "id": "r-17"}},
             {"tool": "check_availability", "args": {"id": "r-17", "people": people, "time": f"{hour}:00"},
              "result": {"available": True}}]
    booked_people, booked_hour = people, hour
    if outcome == "wrong_time":
        booked_hour = hour + rng.choice((-1, 1))
    if outcome == "wrong_size":
        booked_people = people + rng.choice((-1, 1, 2))
    if outcome == "failed":
        steps.append({"tool": "create_booking", "args": {"id": "r-17", "people": people, "time": f"{hour}:00"},
                      "result": {"error": "payment method required"}})
    elif outcome != "not_finished":
        steps.append({"tool": "create_booking", "args": {"id": "r-17", "people": booked_people,
                                                         "time": f"{booked_hour}:00"},
                      "result": {"confirmed": True, "booking_ref": f"BK{rng.randrange(10**5, 10**6)}"}})
    state = {"user_goal": goal, "steps_taken": steps}
    return {"state": state, "labels": {"goal_completed": int(outcome == "done")}}


GENERATORS = {
    "refund": refund_case, "transactions": transactions_case, "access": access_case,
    "ticket": ticket_case, "logs": logs_case, "agent_steps": agent_steps_case,
}


def generate(kind: str, n: int, seed: int) -> list[Case]:
    rng = random.Random(f"general-benchmark-{kind}-{seed}")  # benchmark-only seed space
    return [GENERATORS[kind](rng) for _ in range(n)]
