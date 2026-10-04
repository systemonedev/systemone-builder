"""Every generated label must follow from the state alone (recomputed here independently)."""
from __future__ import annotations

import collections
import re
from datetime import date, datetime

from systemone_builder.system_one.general_generators import generate

N = 300


def test_refund_labels_follow_the_policy():
    for c in generate("refund", N, 1):
        s, lab = c["state"], c["labels"]
        days = (date.fromisoformat(s["request"]["date"]) - date.fromisoformat(s["order"]["delivered_on"])).days
        if s["order"]["category"] in s["policy"]["excluded_categories"]:
            reason = "excluded_category"
        elif days > s["policy"]["return_window_days"]:
            reason = "outside_return_window"
        elif "used" in s["order"]["item_condition"] and "unused" not in s["order"]["item_condition"]:
            reason = "item_used"
        else:
            reason = "eligible"
        assert lab["refund_reason"] == reason and lab["refund_eligible"] == int(reason == "eligible")


def test_transaction_labels_follow_the_numbers():
    for c in generate("transactions", N, 1):
        s, lab = c["state"], c["labels"]
        total = sum(t["amount_usd"] for t in s["transactions_today"])
        assert lab["over_daily_limit"] == int(total > s["customer"]["daily_spending_limit_usd"])
        assert lab["foreign_transaction"] == int(any(t["merchant_country"] != s["customer"]["home_country"]
                                                     for t in s["transactions_today"]))


def test_access_labels_follow_the_written_policy():
    level = {"intern": 0, "contractor": 0, "engineer": 1, "manager": 2, "director": 3}
    for c in generate("access", N, 1):
        u, r, lab = c["state"]["user"], c["state"]["resource"], c["labels"]
        same = u["department"] == r["owning_department"]
        expected = {"public": True, "internal": u["role"] != "contractor",
                    "confidential": same and level[u["role"]] >= 1,
                    "restricted": same and level[u["role"]] >= 2}[r["sensitivity"]]
        assert lab["access_allowed"] == int(expected)


def test_ticket_priority_follows_the_rules():
    for c in generate("ticket", N, 1):
        t = c["state"]["ticket"]
        age = (datetime.fromisoformat(t["now"]) - datetime.fromisoformat(t["created_at"])).total_seconds() / 3600
        sla, tier = t["sla_first_response_hours"], t["customer_tier"]
        outage = "down" in t["summary"]
        overdue = age > sla
        expected = 2 if outage or (overdue and tier == "enterprise") else \
            1 if overdue or (tier == "enterprise" and age > sla / 2) else 0
        assert c["labels"]["ticket_priority"] == expected


def test_log_labels_follow_the_log_lines():
    for c in generate("logs", 120, 1):
        s, lab = c["state"], c["labels"]
        now = datetime.strptime(s["now"], "%H:%M:%S")
        recent = collections.defaultdict(lambda: [0, 0])
        for line in s["log"].splitlines():
            m = re.match(r"(\d\d:\d\d:\d\d) (ERROR|INFO ) \[(\w+)\]", line)
            t = datetime.strptime(m.group(1), "%H:%M:%S")
            if (now - t).total_seconds() <= 300:
                recent[m.group(3)][0] += 1
                recent[m.group(3)][1] += m.group(2) == "ERROR"
        bad = {svc: e / n for svc, (n, e) in recent.items() if n >= 5 and e / n > 0.10}
        assert lab["service_failing"] == int(bool(bad))
        assert lab["failing_service"] == (max(bad, key=bad.get) if bad else "none")


def test_agent_goal_completion_follows_the_steps():
    for c in generate("agent_steps", N, 1):
        goal = c["state"]["user_goal"]
        people, hour = map(int, re.search(r"for (\d+) .* at (\d+):00", goal).groups())
        last = c["state"]["steps_taken"][-1]
        done = (last["tool"] == "create_booking" and last["result"].get("confirmed")
                and last["args"]["people"] == people and last["args"]["time"] == f"{hour}:00")
        assert c["labels"]["goal_completed"] == int(bool(done))


def test_every_question_has_both_classes_and_generation_is_reproducible():
    for kind in ("refund", "transactions", "access", "ticket", "logs", "agent_steps"):
        cases = generate(kind, 200, 7)
        for qid in cases[0]["labels"]:
            assert len({c["labels"][qid] for c in cases}) >= 2, (kind, qid)
        assert cases == generate(kind, 200, 7)
