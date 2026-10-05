"""Training cases over structured state, with exact rule-derived labels (TRAINING ONLY).

Kenning v0.4 saw almost only short text, and on the general benchmark it is near chance on records,
tables and agent steps even when the whole state fits in its window. These generators teach the
skill those families need: read the rules written in the state, apply them to the values (numbers,
dates, lists, tool results) and answer.

Every label is computed by an explicit rule from the generated case. The domains and templates are
deliberately different from the benchmark's (system_one/general_generators.py: refunds, spending
limits, access rules, ticket priority, restaurant bookings, its log format), and the seeds live in
their own space, so training on these never leaks benchmark items.

Each generator returns (state, attrs, labels); `rows()` turns them into training rows with varied
question forms (noul, choice, score, "is the answer X?").
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from typing import Any, Callable

from systemone_builder.kenning.synthetic_tasks import Attr, questions_for

Case = tuple[Any, list[Attr], dict[str, str]]
YN = {"yes": "", "no": ""}


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _d(x: date) -> str:
    return x.isoformat()


def _pick(rng: random.Random, *options: str) -> str:
    return rng.choice(options)


# ======================================================================= records
def expense_case(rng: random.Random) -> Case:
    caps = {"meals": rng.choice((50, 75, 100)), "lodging": rng.choice((200, 250, 300)),
            "software": rng.choice((100, 300, 500)), "travel": rng.choice((800, 1500)), "office": 150}
    receipt_over = rng.choice((25, 50, 75))
    manager_over = rng.choice((500, 1000))
    days_to_submit = rng.choice((30, 60, 90))
    issue = rng.choice(("none", "none", "over_cap", "missing_receipt", "late_submission"))
    cat = rng.choice(list(caps))
    amount = round(rng.uniform(caps[cat] * 1.05, caps[cat] * 2), 2) if issue == "over_cap" else round(
        rng.uniform(5, caps[cat]), 2)
    if issue == "missing_receipt":
        amount = round(rng.uniform(max(receipt_over + 1, 5), max(caps[cat], receipt_over + 2)), 2)
    spent = date(2026, 3, 1) + timedelta(days=rng.randrange(0, 200))
    late = issue == "late_submission"
    submitted = spent + timedelta(days=rng.randrange(days_to_submit + 1, days_to_submit + 40) if late
                                  else rng.randrange(0, days_to_submit + 1))
    receipt = issue != "missing_receipt" if amount > receipt_over else rng.random() < 0.5
    # labels from the rules, in the order a reviewer checks them
    found = ("over_cap" if amount > caps[cat] else "missing_receipt" if amount > receipt_over and not receipt
             else "late_submission" if (submitted - spent).days > days_to_submit else "none")
    state = {
        "policy": {f"cap_{k}_usd": v for k, v in caps.items()} | {
            "receipt_required_over_usd": receipt_over, "manager_approval_over_usd": manager_over,
            "submit_within_days": days_to_submit},
        "expense": {"category": cat, "amount_usd": amount, "date_spent": _d(spent), "date_submitted": _d(submitted),
                    "receipt_attached": receipt},
    }
    attrs = [
        Attr("within_policy", ["Does this expense comply with the policy?", "Can this expense be reimbursed as submitted?"], YN),
        Attr("violation", ["Which rule, if any, does this expense break first?", "What is wrong with this expense claim?"],
             {"none": "complies", "over_cap": "amount above the category cap", "missing_receipt": "no receipt where one is required",
              "late_submission": "submitted after the deadline"}),
        Attr("manager_approval", ["Does this expense need a manager's approval?"], YN),
    ]
    return state, attrs, {"within_policy": _yn(found == "none"), "violation": found,
                          "manager_approval": _yn(amount > manager_over)}


def shipping_case(rng: random.Random) -> Case:
    services = {"standard": rng.choice((20, 30)), "express": rng.choice((5, 10)), "freight": 500}
    restricted = {"lithium batteries": ["air"], "liquids over 1L": ["air"], "perishables": ["standard"]}
    service = rng.choice(list(services))
    mode = "air" if service == "express" else "ground"
    cutoff = rng.choice((12, 14, 16))
    problem = rng.choice(("ok", "ok", "overweight", "restricted", "no_address"))
    weight = round(rng.uniform(services[service] * 1.05, services[service] * 1.8), 1) if problem == "overweight" else round(
        rng.uniform(0.2, services[service]), 1)
    contents = rng.choice(["books", "clothing", "shoes", "kitchenware", "toys"])
    if problem == "restricted":
        bad = [(c, m) for c, ms in restricted.items() for m in ms if m == mode or (m == "standard" and service == "standard")]
        contents = rng.choice(bad)[0] if bad else contents
    order_time = datetime(2026, 5, 4, rng.randrange(7, 20), rng.choice((0, 15, 30, 45)))
    address_ok = problem != "no_address"
    is_restricted = any(contents == c and (m == mode or (m == "standard" and service == "standard"))
                        for c, ms in restricted.items() for m in ms)
    issue = ("overweight" if weight > services[service] else "restricted_contents" if is_restricted
             else "incomplete_address" if not address_ok else "none")
    can_ship = issue == "none"
    state = {
        "rules": [f"{s}: up to {w} kg" for s, w in services.items()]
                 + [f"express travels by air; no {c} by air" for c, ms in restricted.items() if "air" in ms]
                 + ["perishables can't go by standard service", f"orders placed before {cutoff}:00 ship the same day"],
        "order": {"service": service, "weight_kg": weight, "contents": contents,
                  "placed_at": order_time.strftime("%H:%M"),
                  "address": "12 Harbour St, Leeds LS1 4AP" if address_ok else "12 Harbour St"},
    }
    attrs = [
        Attr("can_ship", ["Can this order ship with the chosen service?", "Is this shipment allowed?"], YN),
        Attr("blocking_issue", ["What stops this order from shipping?"],
             {"none": "nothing", "overweight": "too heavy for the service", "restricted_contents": "contents not allowed",
              "incomplete_address": "address is incomplete"}),
        Attr("ships_today", ["Will this order ship today?"], YN),
    ]
    return state, attrs, {"can_ship": _yn(can_ship), "blocking_issue": issue,
                          "ships_today": _yn(can_ship and order_time.hour < cutoff)}


def leave_case(rng: random.Random) -> Case:
    balance = rng.randrange(0, 25)
    days = rng.randrange(1, 15)
    notice_per_day = rng.choice((1, 2))
    today = date(2026, 6, 1) + timedelta(days=rng.randrange(0, 90))
    start = today + timedelta(days=rng.randrange(1, 40))
    blackout = (date(2026, 12, 15), date(2026, 12, 31)) if rng.random() < 0.5 else (start + timedelta(days=rng.randrange(-5, 10)),
                                                                                    start + timedelta(days=rng.randrange(10, 20)))
    end = start + timedelta(days=days - 1)
    notice = (start - today).days
    reason = ("insufficient_balance" if days > balance else "short_notice" if notice < days * notice_per_day
              else "blackout_period" if start <= blackout[1] and end >= blackout[0] else "approved")
    state = {"policy": {"notice_days_per_leave_day": notice_per_day, "blackout": f"{_d(blackout[0])} to {_d(blackout[1])}"},
             "employee": {"leave_balance_days": balance},
             "request": {"submitted": _d(today), "first_day": _d(start), "last_day": _d(end), "working_days": days}}
    attrs = [Attr("approve", ["Should this leave request be approved?", "Does the request meet the leave policy?"], YN),
             Attr("decision", ["What is the outcome of this leave request?"],
                  {"approved": "", "insufficient_balance": "not enough days left", "short_notice": "not enough notice",
                   "blackout_period": "overlaps a blackout period"})]
    return state, attrs, {"approve": _yn(reason == "approved"), "decision": reason}


def inventory_case(rng: random.Random) -> Case:
    names = rng.sample(["bolts M6", "filters", "gloves", "labels", "cartons", "tape", "valves", "cables", "pallets", "sensors"], 5)
    items = []
    for n in names:
        usage = rng.randrange(2, 40)
        lead = rng.randrange(2, 15)
        point = usage * lead + rng.randrange(0, usage * 3)
        stock = rng.randrange(0, point * 2)
        items.append({"item": n, "on_hand": stock, "reorder_point": point, "daily_usage": usage, "lead_time_days": lead})
    target = rng.choice(items)
    days_left = target["on_hand"] / target["daily_usage"]
    reorder = target["on_hand"] <= target["reorder_point"]
    stockout = days_left < target["lead_time_days"]
    urgency = "critical" if stockout else "soon" if reorder else "none"
    state = {"rule": "reorder when on_hand is at or below reorder_point; an item runs out before a new order "
                     "arrives when on_hand / daily_usage is less than lead_time_days",
             "inventory": items, "question_about": target["item"]}
    attrs = [Attr("needs_reorder", [f"Does {target['item']} need to be reordered?"], YN),
             Attr("stockout_risk", [f"Will {target['item']} run out before a new order arrives?"], YN),
             Attr("urgency", [f"How urgent is restocking {target['item']}?"],
                  {"none": "above the reorder point", "soon": "at or below the reorder point",
                   "critical": "runs out before a new order can arrive"}, ordinal=True)]
    return state, attrs, {"needs_reorder": _yn(reorder), "stockout_risk": _yn(stockout), "urgency": urgency}


def incident_case(rng: random.Random) -> Case:
    sla = {"sev1": 15, "sev2": 60, "sev3": 240, "sev4": 1440}
    sev = rng.choice(list(sla))
    opened = datetime(2026, 7, 9, rng.randrange(0, 23), rng.randrange(0, 60))
    # whole minutes, as displayed, so the label is what a reader of the state would compute
    elapsed = round(rng.choice((rng.uniform(0.05, 0.5), rng.uniform(0.5, 0.95), rng.uniform(1.05, 3))) * sla[sev])
    responded = rng.random() < 0.4
    at = opened + timedelta(minutes=elapsed)
    breached = elapsed > sla[sev]
    left = sla[sev] - elapsed
    state = {"first_response_sla_minutes": sla,
             "incident": {"severity": sev, "opened_at": opened.strftime("%Y-%m-%d %H:%M")} | (
                 {"first_response_at": at.strftime("%Y-%m-%d %H:%M")} if responded else
                 {"status": "no response yet", "now": at.strftime("%Y-%m-%d %H:%M")})}
    attrs = [Attr("sla_breached", ["Was the first-response SLA missed?" if responded else "Has the first-response SLA been breached?"], YN)]
    labels = {"sla_breached": _yn(breached)}
    if not responded:
        attrs.append(Attr("time_pressure", ["How close is this incident to breaching its SLA?"],
                          {"comfortable": "more than half the SLA left", "tight": "less than half left",
                           "breached": "past the SLA"}, ordinal=True))
        labels["time_pressure"] = "breached" if breached else "tight" if left < sla[sev] / 2 else "comfortable"
    return state, attrs, labels


def eligibility_case(rng: random.Random) -> Case:
    products = {"standard account": 16, "credit card": 18, "car rental": 21, "senior plan": 65}
    product = rng.choice(list(products))
    allowed = rng.sample(["US", "CA", "GB", "IE", "DE", "FR", "AU", "NZ", "MX", "BR"], 6)
    today = date(2026, 8, 20)
    age_ok = rng.random() < 0.6
    min_age = products[product]
    years = rng.randrange(min_age, min_age + 30) if age_ok else rng.randrange(max(10, min_age - 8), min_age)
    if rng.random() < 0.25:  # birthday edge: within a few days of the cutoff
        dob = date(today.year - min_age, today.month, today.day) + timedelta(days=rng.choice((-2, -1, 1, 2)))
    else:
        dob = date(today.year - years, rng.randrange(1, 13), rng.randrange(1, 29))
    country = rng.choice(allowed) if rng.random() < 0.7 else rng.choice(["JP", "IN", "ZA", "AR", "KR"])
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    reason = "too_young" if age < min_age else "country_not_supported" if country not in allowed else "eligible"
    state = {"product": product, "minimum_age": min_age, "available_in": allowed, "today": _d(today),
             "applicant": {"date_of_birth": _d(dob), "country": country}}
    attrs = [Attr("eligible", [f"Is the applicant eligible for the {product}?"], YN),
             Attr("reason", ["Why is the applicant eligible or not?"],
                  {"eligible": "", "too_young": "below the minimum age", "country_not_supported": "country not supported"})]
    return state, attrs, {"eligible": _yn(reason == "eligible"), "reason": reason}


def loan_case(rng: random.Random) -> Case:
    max_dti = rng.choice((0.36, 0.40, 0.43))
    min_score = rng.choice((620, 650, 680))
    mult = rng.choice((4, 5))
    income = rng.randrange(2500, 12000, 50)
    debt = round(income * rng.uniform(0.05, 0.6))
    score = rng.randrange(540, 820)
    amount = rng.randrange(1000, income * mult * 2, 500)
    dti = debt / income
    reason = ("credit_score" if score < min_score else "debt_to_income" if dti > max_dti
              else "amount_too_high" if amount > income * mult else "prequalified")
    state = {"criteria": {"max_debt_to_income": max_dti, "min_credit_score": min_score,
                          "max_amount": f"{mult} x monthly income"},
             "applicant": {"monthly_income_usd": income, "monthly_debt_payments_usd": debt, "credit_score": score,
                           "requested_amount_usd": amount}}
    attrs = [Attr("prequalified", ["Does the applicant pre-qualify?", "Does this application meet every criterion?"], YN),
             Attr("first_failed_check", ["Which criterion fails first, in the order listed?"],
                  {"prequalified": "none fails", "credit_score": "", "debt_to_income": "", "amount_too_high": ""})]
    return state, attrs, {"prequalified": _yn(reason == "prequalified"), "first_failed_check": reason}


def subscription_case(rng: random.Random) -> Case:
    today = date(2026, 9, 10)
    trial_end = today + timedelta(days=rng.randrange(-10, 30))
    card = rng.random() < 0.6
    cancelled = rng.random() < 0.25
    days = (trial_end - today).days
    charged = card and not cancelled and days >= 0
    state = {"today": _d(today), "account": {"plan": rng.choice(["Pro", "Team", "Plus"]), "trial_ends": _d(trial_end),
                                              "card_on_file": card, "cancellation_requested": cancelled},
             "billing_rule": "at the end of the trial, accounts with a card on file and no cancellation are charged; "
                             "others are downgraded to free"}
    attrs = [Attr("trial_ending_soon", ["Does the trial end within the next 7 days?"], YN),
             Attr("outcome", ["What happens when the trial ends?"],
                  {"charged": "moves to the paid plan", "downgraded": "moves to the free plan",
                   "already_ended": "the trial is already over"})]
    return state, attrs, {"trial_ending_soon": _yn(0 <= days <= 7),
                          "outcome": "already_ended" if days < 0 else "charged" if charged else "downgraded"}


# ======================================================================= tables
TABLES = {
    "league": (["team", "played", "wins", "losses", "points"],
               lambda r: [r.choice(["Rovers", "United", "Athletic", "City", "Wanderers", "Rangers", "Albion", "Town",
                                    "Harriers", "Olympic", "Dynamo", "Celtic"])]),
    "catalog": (["product", "price_usd", "stock", "rating"],
                lambda r: [r.choice(["Desk lamp", "Kettle", "Headphones", "Backpack", "Monitor", "Keyboard", "Blender",
                                     "Chair", "Router", "Speaker", "Mouse", "Webcam"])]),
    "cities": (["city", "population_k", "area_km2", "founded"],
               lambda r: [r.choice(["Arden", "Belmont", "Corvale", "Dunmore", "Elstow", "Fairhaven", "Glenrock",
                                    "Holloway", "Ivydale", "Juniper", "Kestrel", "Larkspur"])]),
}


def _table_rows(kind: str, rng: random.Random) -> tuple[list[str], list[list[Any]]]:
    cols = TABLES[kind][0]
    n = rng.randrange(4, 11)
    names: list[str] = []
    while len(names) < n:
        x = TABLES[kind][1](rng)[0]
        if x not in names:
            names.append(x)
    rows = []
    for name in names:
        if kind == "league":
            p = rng.randrange(10, 20)
            w = rng.randrange(0, p + 1)
            losses = rng.randrange(0, p - w + 1)
            rows.append([name, p, w, losses, 3 * w + (p - w - losses)])
        elif kind == "catalog":
            rows.append([name, round(rng.uniform(8, 400), 2), rng.randrange(0, 120), round(rng.uniform(2.5, 5), 1)])
        else:
            rows.append([name, rng.randrange(20, 900), rng.randrange(30, 600), rng.randrange(1650, 1990)])
    return cols, rows


def _render_table(cols: list[str], rows: list[list[Any]], rng: random.Random) -> Any:
    style = rng.choice(("markdown", "csv", "json"))
    if style == "json":
        return [dict(zip(cols, r)) for r in rows]
    if style == "csv":
        return "\n".join([",".join(cols)] + [",".join(str(v) for v in r) for r in rows])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def table_case(rng: random.Random) -> Case:
    kind = rng.choice(list(TABLES))
    cols, rows = _table_rows(kind, rng)
    key, nums = cols[0], cols[1:]
    col = rng.choice(nums)
    idx = cols.index(col)
    kind_q = rng.choice(("compare", "superlative", "count", "sum", "value"))
    a, b = rng.sample(rows, 2)
    if kind_q == "compare":
        truth = a[idx] > b[idx]
        stmt = f"{a[0]} has a higher {col} than {b[0]}."
    elif kind_q == "superlative":
        best = max(rows, key=lambda r: r[idx])
        cand = best if rng.random() < 0.5 else rng.choice(rows)
        truth = cand[idx] == best[idx]
        stmt = f"{cand[0]} has the highest {col}."
    elif kind_q == "count":
        thr = sorted(r[idx] for r in rows)[len(rows) // 2]
        k = sum(r[idx] > thr for r in rows)
        claim = k if rng.random() < 0.5 else k + (1 if k == 0 else rng.choice((-1, 1)))
        truth = claim == k
        stmt = f"Exactly {claim} rows have a {col} above {thr}."
    elif kind_q == "sum":
        total = round(sum(r[idx] for r in rows), 2)
        claim = total if rng.random() < 0.5 else round(total * rng.choice((0.8, 0.9, 1.1, 1.25)), 2)
        truth = abs(claim - total) < 1e-6
        stmt = f"The total {col} across all rows is {claim}."
    else:
        claim = a[idx] if rng.random() < 0.5 else b[idx] if b[idx] != a[idx] else a[idx] + 1
        truth = claim == a[idx]
        stmt = f"The {col} of {a[0]} is {claim}."
    best = max(rows, key=lambda r: r[idx])
    ties = sum(r[idx] == best[idx] for r in rows)
    state = {"table": _render_table(cols, rows, rng), "statement": stmt}
    attrs = [Attr("statement_true", ["Is the statement true according to the table?", "Does the table support the statement?"], YN)]
    labels = {"statement_true": _yn(truth)}
    if ties == 1 and len(rows) <= 8:
        attrs.append(Attr("top_row", [f"Which {key} has the highest {col}?"], {r[0]: "" for r in rows}))
        labels["top_row"] = best[0]
    return state, attrs, labels


# ======================================================================= agents
FUNCTIONS = {
    "send_email": {"to": "address", "subject": "text", "body": "text"},
    "create_event": {"title": "text", "date": "YYYY-MM-DD", "time": "HH:MM", "attendees": "list"},
    "set_reminder": {"text": "text", "at": "YYYY-MM-DD HH:MM"},
    "get_weather": {"city": "text", "date": "YYYY-MM-DD"},
    "convert_currency": {"amount": "number", "from": "code", "to": "code"},
    "search_flights": {"origin": "airport", "destination": "airport", "date": "YYYY-MM-DD"},
    "move_file": {"path": "path", "destination": "folder"},
    "add_to_cart": {"product_id": "id", "quantity": "number"},
    "translate_text": {"text": "text", "target_language": "code"},
    "create_ticket": {"title": "text", "priority": "low|medium|high"},
}


def _request(rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    city = rng.choice(["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo"])
    day = _d(date(2026, 11, 1) + timedelta(days=rng.randrange(0, 50)))
    hour = f"{rng.randrange(8, 18):02d}:{rng.choice(('00', '30'))}"
    person = rng.choice(["maria@acme.io", "dev-team@acme.io", "li.wei@acme.io"])
    amount = rng.randrange(10, 5000)
    options = [
        (f"What's the weather in {city} on {day}?", "get_weather", {"city": city, "date": day}),
        (f"Convert {amount} EUR to USD.", "convert_currency", {"amount": amount, "from": "EUR", "to": "USD"}),
        (f"Remind me to renew the domain on {day} at {hour}.", "set_reminder", {"text": "renew the domain", "at": f"{day} {hour}"}),
        (f"Book a meeting called 'Q4 review' on {day} at {hour} with {person}.", "create_event",
         {"title": "Q4 review", "date": day, "time": hour, "attendees": [person]}),
        (f"Find flights from LHR to JFK on {day}.", "search_flights", {"origin": "LHR", "destination": "JFK", "date": day}),
        ("Move report.pdf into the Archive folder.", "move_file", {"path": "report.pdf", "destination": "Archive"}),
        (f"Add {amount % 5 + 1} of product SKU-{amount} to my cart.", "add_to_cart",
         {"product_id": f"SKU-{amount}", "quantity": amount % 5 + 1}),
        (f"Email {person} that the deploy is done, subject 'Deploy'.", "send_email",
         {"to": person, "subject": "Deploy", "body": "The deploy is done."}),
    ]
    return rng.choice(options)


def tool_call_case(rng: random.Random) -> Case:
    req, fn, args = _request(rng)
    avail = sorted({fn, *rng.sample(list(FUNCTIONS), 3)})
    problem = rng.choice(("none", "none", "wrong_function", "wrong_argument", "missing_argument"))
    call_fn, call_args = fn, dict(args)
    if problem == "wrong_function":
        call_fn = rng.choice([f for f in avail if f != fn] or [f for f in FUNCTIONS if f != fn])
    elif problem == "wrong_argument":
        k = rng.choice(list(call_args))
        v = call_args[k]
        call_args[k] = (v + rng.choice((1, 10, 100)) if isinstance(v, int) else ["someone@else.io"] if isinstance(v, list)
                        else v[:-1] + ("1" if v[-1] != "1" else "2") if any(ch.isdigit() for ch in str(v)) else "Madrid")
    elif problem == "missing_argument":
        call_args.pop(rng.choice(list(call_args)))
    state = {"available_functions": {f: FUNCTIONS[f] for f in avail}, "user_request": req,
             "proposed_call": {"name": call_fn, "arguments": call_args}}
    attrs = [Attr("call_correct", ["Does the proposed call do exactly what the user asked?",
                                   "Is this function call correct for the request?"], YN),
             Attr("call_problem", ["What is wrong with the proposed call?"],
                  {"none": "it is correct", "wrong_function": "calls the wrong function",
                   "wrong_argument": "an argument has the wrong value", "missing_argument": "a needed argument is missing"})]
    return state, attrs, {"call_correct": _yn(problem == "none"), "call_problem": problem}


def agent_trace_case(rng: random.Random) -> Case:
    req, fn, args = _request(rng)
    outcome = rng.choice(("completed", "completed", "wrong_details", "error_ignored", "unfinished", "extra_action"))
    steps: list[dict[str, Any]] = [{"thought": "find what the user needs", "action": "read_request"}]
    call_args = dict(args)
    if outcome == "wrong_details":
        k = rng.choice(list(call_args))
        call_args[k] = "2026-01-01" if "date" in k or k == "at" else 999 if isinstance(call_args[k], int) else "unknown"
    if outcome == "unfinished":
        steps.append({"action": "list_functions", "result": sorted(FUNCTIONS)[:4]})
    else:
        result = {"error": "service unavailable"} if outcome == "error_ignored" else {"ok": True}
        steps.append({"action": fn, "arguments": call_args, "result": result})
        if outcome == "extra_action":
            steps.append({"action": "send_email", "arguments": {"to": "all-staff@acme.io", "subject": "FYI", "body": req},
                          "result": {"ok": True}})
    final = ("Done!" if outcome != "unfinished" else "Let me look into that.")
    state = {"user_request": req, "steps": steps, "agent_final_message": final}
    attrs = [Attr("task_completed", ["Did the agent complete what the user asked?"], YN),
             Attr("issue", ["What went wrong in this run, if anything?"],
                  {"none": "done correctly", "wrong_details": "acted with the wrong details",
                   "ignored_error": "claimed success after an error", "stopped_early": "never did the task",
                   "unrequested_action": "did something the user didn't ask for"}),
             Attr("needs_user_attention", ["Should the user be told something went wrong?"], YN)]
    issue = {"completed": "none", "wrong_details": "wrong_details", "error_ignored": "ignored_error",
             "unfinished": "stopped_early", "extra_action": "unrequested_action"}[outcome]
    return state, attrs, {"task_completed": _yn(outcome in ("completed", "extra_action")), "issue": issue,
                          "needs_user_attention": _yn(issue != "none")}


# ======================================================================= logs
def access_log_case(rng: random.Random) -> Case:
    """nginx-style access log: is the 5xx rate over the threshold, which endpoint fails most, brute force?"""
    endpoints = rng.sample(["/api/orders", "/api/login", "/api/cart", "/api/search", "/static/app.js", "/api/profile"], 4)
    bad_ep = rng.choice(endpoints) if rng.random() < 0.5 else None
    brute_ip = f"203.0.113.{rng.randrange(2, 250)}" if rng.random() < 0.4 else None
    thr = rng.choice((5, 10))
    lines, fails = [], {e: 0 for e in endpoints}
    total = rng.randrange(12, 22)  # stays under ~1k tokens
    t = datetime(2026, 9, 30, 14, 0, 0)
    logins_failed: dict[str, int] = {}
    for _ in range(total):
        t += timedelta(seconds=rng.randrange(1, 9))
        ep = rng.choice(endpoints)
        ip = f"198.51.100.{rng.randrange(2, 60)}"
        status = 200
        if bad_ep and ep == bad_ep and rng.random() < 0.6:
            status = rng.choice((500, 502, 503))
        elif rng.random() < 0.03:
            status = 500
        if brute_ip and rng.random() < 0.3:
            ip, ep, status = brute_ip, "/api/login", 401
        if status >= 500:
            fails[ep] = fails.get(ep, 0) + 1
        if ep == "/api/login" and status == 401:
            logins_failed[ip] = logins_failed.get(ip, 0) + 1
        lines.append(f'{ip} - - [{t.strftime("%d/%b/%Y:%H:%M:%S")}] "GET {ep} HTTP/1.1" {status} {rng.randrange(200, 9000)}')
    n5 = sum(fails.values())
    rate = 100 * n5 / total
    worst: str | None = max(fails, key=fails.get) if n5 else "none"
    if n5 and list(fails.values()).count(fails[worst]) > 1:
        worst = None  # tie: the choice question is skipped
    brute = any(v >= 5 for v in logins_failed.values())
    state = {"rules": f"alert when more than {thr}% of requests return 5xx; brute force is 5 or more failed "
                      "logins (401 on /api/login) from one IP", "access_log": "\n".join(lines)}
    attrs = [Attr("error_alert", ["Should the 5xx alert fire for this window?"], YN),
             Attr("brute_force", ["Is there a brute-force login attempt in this log?"], YN)]
    labels = {"error_alert": _yn(rate > thr), "brute_force": _yn(brute)}
    if worst:
        attrs.append(Attr("worst_endpoint", ["Which endpoint returned the most 5xx errors?"],
                          {**{e: "" for e in endpoints}, **({"none": "no 5xx at all"} if "none" not in endpoints else {})}))
        labels["worst_endpoint"] = worst
    return state, attrs, labels


def deploy_log_case(rng: random.Random) -> Case:
    """JSON lines around a deploy: did errors start after it, should it be rolled back?"""
    regress = rng.random() < 0.5
    t = datetime(2026, 10, 2, 9, 0, 0)
    out, before, after = [], 0, 0
    n = rng.randrange(14, 24)
    at = rng.randrange(4, n - 4)
    for i in range(n):
        t += timedelta(seconds=rng.randrange(5, 40))
        if i == at:
            out.append(json.dumps({"ts": t.strftime("%H:%M:%S"), "level": "info", "msg": f"deploy v2.{rng.randrange(1, 40)} complete"}))
            continue
        p = 0.45 if (regress and i > at) else 0.04
        err = rng.random() < p
        before += int(err and i < at)
        after += int(err and i > at)
        out.append(json.dumps({"ts": t.strftime("%H:%M:%S"), "level": "error" if err else "info",
                               "msg": rng.choice(["db timeout", "null pointer in checkout", "upstream 502"]) if err
                               else rng.choice(["request ok", "cache hit", "job finished"])}))
    rollback = after >= 3 and after > 2 * max(before, 1)
    state = {"rule": "roll back when there are at least 3 errors after the deploy and more than twice as many as before it",
             "log": "\n".join(out)}
    attrs = [Attr("rollback", ["Should this deploy be rolled back?"], YN),
             Attr("errors_after_deploy", ["Are there more errors after the deploy than before it?"], YN)]
    return state, attrs, {"rollback": _yn(rollback), "errors_after_deploy": _yn(after > before)}


GENERATORS: dict[str, Callable[[random.Random], Case]] = {
    "expense": expense_case, "shipping": shipping_case, "leave": leave_case, "inventory": inventory_case,
    "incident": incident_case, "eligibility": eligibility_case, "loan": loan_case, "subscription": subscription_case,
    "table": table_case, "tool_call": tool_call_case, "agent_trace": agent_trace_case,
    "access_log": access_log_case, "deploy_log": deploy_log_case,
}
FAMILY = {"expense": "records", "shipping": "records", "leave": "records", "inventory": "records", "incident": "records",
          "eligibility": "records", "loan": "records", "subscription": "records", "table": "table",
          "tool_call": "agent", "agent_trace": "agent", "access_log": "logs", "deploy_log": "logs"}


def cases(kind: str, n: int, seed: int) -> list[Case]:
    """n cases, alternating yes and no on the first (yes/no) question so neither answer dominates."""
    rng = random.Random(f"train-structured-{kind}-{seed}")  # training-only seed space
    out = []
    for i in range(n):
        want = "yes" if i % 2 == 0 else "no"
        for _ in range(50):
            case = GENERATORS[kind](rng)
            if case[2][case[1][0].name] == want:
                break
        out.append(case)
    return out


def rows(kind: str, n: int, seed: int) -> list[dict[str, Any]]:
    """n training rows for one generator, each asking all of its questions in varied forms."""
    rng = random.Random(f"train-structured-rows-{kind}-{seed}")
    return [{"state": state, **questions_for(attrs, labels, rng)} for state, attrs, labels in cases(kind, n, seed)]
