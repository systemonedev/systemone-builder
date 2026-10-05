"""Training generators over structured state: labels recomputed independently from each state."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime

from systemone_builder.kenning import structured as st
from systemone_builder.system_one import general_generators
from systemone_builder.system_one.contract import Question

N = 300


def _cases(kind):
    return st.cases(kind, N, 3)


def test_every_row_is_valid_and_every_generator_is_balanced():
    for kind in st.GENERATORS:
        for row in st.rows(kind, 60, 1):
            assert row["targets"] and set(row["targets"]) == set(row["questions"])
            for qid, q in row["questions"].items():
                q = Question.model_validate(q)
                t = row["targets"][qid]
                if q.type == "noul":
                    assert t in (0, 1)
                elif q.type == "choice":
                    assert t in q.criteria
                else:
                    assert 0 <= t < len(q.criteria)
        first = [labels[attrs[0].name] for _, attrs, labels in st.cases(kind, 100, 2)]
        assert first.count("yes") == first.count("no") == 50, kind


def test_reproducible_and_separate_from_the_benchmark():
    assert st.rows("loan", 20, 5) == st.rows("loan", 20, 5)
    assert st.rows("loan", 20, 5) != st.rows("loan", 20, 6)
    assert not set(st.GENERATORS) & set(general_generators.GENERATORS)  # no shared generator names / domains
    bench = {json.dumps(c["state"], sort_keys=True) for k in general_generators.GENERATORS
             for c in general_generators.generate(k, 50, 42)}
    train = {json.dumps(s, sort_keys=True) for k in st.GENERATORS for s, _, _ in st.cases(k, 50, 42)}
    assert not bench & train


def test_expense_labels():
    for state, _, labels in _cases("expense"):
        pol, e = state["policy"], state["expense"]
        cap = pol[f"cap_{e['category']}_usd"]
        late = (date.fromisoformat(e["date_submitted"]) - date.fromisoformat(e["date_spent"])).days > pol["submit_within_days"]
        want = ("over_cap" if e["amount_usd"] > cap else
                "missing_receipt" if e["amount_usd"] > pol["receipt_required_over_usd"] and not e["receipt_attached"] else
                "late_submission" if late else "none")
        assert labels["violation"] == want
        assert labels["within_policy"] == ("yes" if want == "none" else "no")
        assert labels["manager_approval"] == ("yes" if e["amount_usd"] > pol["manager_approval_over_usd"] else "no")


def test_loan_and_eligibility_labels():
    for state, _, labels in _cases("loan"):
        c, a = state["criteria"], state["applicant"]
        mult = int(c["max_amount"].split()[0])
        want = ("credit_score" if a["credit_score"] < c["min_credit_score"] else
                "debt_to_income" if a["monthly_debt_payments_usd"] / a["monthly_income_usd"] > c["max_debt_to_income"] else
                "amount_too_high" if a["requested_amount_usd"] > mult * a["monthly_income_usd"] else "prequalified")
        assert labels["first_failed_check"] == want
    for state, _, labels in _cases("eligibility"):
        dob, today = date.fromisoformat(state["applicant"]["date_of_birth"]), date.fromisoformat(state["today"])
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        ok = age >= state["minimum_age"] and state["applicant"]["country"] in state["available_in"]
        assert labels["eligible"] == ("yes" if ok else "no")


def test_inventory_and_incident_labels():
    for state, _, labels in _cases("inventory"):
        it = next(x for x in state["inventory"] if x["item"] == state["question_about"])
        assert labels["needs_reorder"] == ("yes" if it["on_hand"] <= it["reorder_point"] else "no")
        assert labels["stockout_risk"] == ("yes" if it["on_hand"] / it["daily_usage"] < it["lead_time_days"] else "no")
    for state, _, labels in _cases("incident"):
        inc = state["incident"]
        sla = state["first_response_sla_minutes"][inc["severity"]]
        end = inc.get("first_response_at") or inc["now"]
        mins = (datetime.fromisoformat(end) - datetime.fromisoformat(inc["opened_at"])).total_seconds() / 60
        assert labels["sla_breached"] == ("yes" if mins > sla else "no")
        if "time_pressure" in labels:
            want = "breached" if mins > sla else "tight" if sla - mins < sla / 2 else "comfortable"
            assert labels["time_pressure"] == want


def _table(state):
    t = state["table"]
    if isinstance(t, list):
        return [list(r.values()) for r in t], list(t[0])
    if t.startswith("|"):
        lines = [ln.strip("|").split("|") for ln in t.splitlines() if not ln.startswith("|-")]
        rows = [[c.strip() for c in ln] for ln in lines]
    else:
        rows = list(csv.reader(io.StringIO(t)))
    cols, body = rows[0], rows[1:]
    return [[r[0]] + [float(v) for v in r[1:]] for r in body], cols


def test_table_statements_are_true_exactly_when_the_table_says_so():
    for state, _, labels in _cases("table"):
        rows, cols = _table(state)
        rows = [[r[0]] + [float(v) for v in r[1:]] for r in rows]
        s = state["statement"]
        col = next(c for c in cols[1:] if f" {c} " in f" {s} ".replace(".", " "))
        i = cols.index(col)
        val = {r[0]: r[i] for r in rows}
        if m := re.match(r"(.+) has a higher \w+ than (.+)\.$", s):
            truth = val[m[1]] > val[m[2]]
        elif m := re.match(r"(.+) has the highest \w+\.$", s):
            truth = val[m[1]] == max(val.values())
        elif m := re.match(r"Exactly (-?\d+) rows have a \w+ above ([\d.]+)\.$", s):
            truth = int(m[1]) == sum(v > float(m[2]) for v in val.values())
        elif m := re.match(r"The total \w+ across all rows is ([\d.]+)\.$", s):
            truth = abs(float(m[1]) - sum(val.values())) < 0.01
        else:
            m = re.match(r"The \w+ of (.+) is ([\d.]+)\.$", s)
            truth = abs(val[m[1]] - float(m[2])) < 1e-6
        assert labels["statement_true"] == ("yes" if truth else "no"), s
        if "top_row" in labels:
            assert val[labels["top_row"]] == max(val.values())


def test_tool_calls_and_traces():
    for state, _, labels in _cases("tool_call"):
        call = state["proposed_call"]
        assert labels["call_correct"] == ("yes" if labels["call_problem"] == "none" else "no")
        if labels["call_problem"] == "missing_argument":
            assert len(call["arguments"]) < len(state["available_functions"].get(call["name"], {}))
        if labels["call_problem"] == "none":
            assert set(call["arguments"]) == set(state["available_functions"][call["name"]])
    for state, _, labels in _cases("agent_trace"):
        results = [s.get("result") for s in state["steps"]]
        if any(isinstance(r, dict) and "error" in r for r in results):
            assert labels["task_completed"] == "no" and labels["issue"] == "ignored_error"
        assert labels["needs_user_attention"] == ("no" if labels["issue"] == "none" else "yes")


def test_log_labels_from_the_log_itself():
    for state, _, labels in _cases("access_log"):
        thr = int(re.search(r"more than (\d+)%", state["rules"])[1])
        lines = state["access_log"].splitlines()
        parsed = [re.match(r'(\S+) .*"GET (\S+) HTTP/1.1" (\d+)', ln).groups() for ln in lines]
        n5 = sum(int(code) >= 500 for _, _, code in parsed)
        assert labels["error_alert"] == ("yes" if 100 * n5 / len(lines) > thr else "no")
        fails: dict[str, int] = {}
        for ip, ep, code in parsed:
            if ep == "/api/login" and code == "401":
                fails[ip] = fails.get(ip, 0) + 1
        assert labels["brute_force"] == ("yes" if any(v >= 5 for v in fails.values()) else "no")
    for state, _, labels in _cases("deploy_log"):
        events = [json.loads(x) for x in state["log"].splitlines()]
        at = next(i for i, e in enumerate(events) if e["msg"].startswith("deploy"))
        before = sum(e["level"] == "error" for e in events[:at])
        after = sum(e["level"] == "error" for e in events[at + 1:])
        assert labels["rollback"] == ("yes" if after >= 3 and after > 2 * max(before, 1) else "no")
        assert labels["errors_after_deploy"] == ("yes" if after > before else "no")
