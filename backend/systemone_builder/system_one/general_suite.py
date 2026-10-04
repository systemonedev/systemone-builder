"""The `general` benchmark: the headline measure of a general-purpose System One model.

A superset of the `multi` suite (14 text tasks) adding the kinds of state and question a general
decision model has to handle:

    table         is a statement true according to a table (TabFact)
    conversation  which service a multi-turn dialogue is about (Schema-Guided Dialogue)
    agent         should the assistant call a function; does a proposed call match the request
                  (Glaive function calling); is the user's goal completed by the steps taken (generated)
    quality       how helpful / whether correct an assistant answer is (HelpSteer2 human ratings)
    records       census income from a record (Adult, CC0); refund eligibility and reason, daily limit
                  and foreign transaction, access policy, ticket priority (generated, exact rule labels)
    logs          is a service failing and which one, in 150-400 line logs (generated, exact labels)

Several generated states are asked more than one question at once (multi-question requests).
Public data: held-out or test splits under permissive licences, fetched at run time with a fixed
seed and cached; nothing is redistributed. Generated cases: system_one.general_generators
(benchmark-only, never used for training). Reports add macro accuracy overall and per family.

    systemone bench --suite general --engines kenning,clef -n 50      # n = items per task
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.system_one.contract import Question
from systemone_builder.system_one.general_generators import generate
from systemone_builder.system_one.multitask_suite import TASKS as TEXT_TASKS
from systemone_builder.system_one.multitask_suite import Task, fetch_task


def _noul(text: str) -> Question:
    return Question(type="noul", instructions=text)


def _choice(text: str, options: dict[str, str | None]) -> Question:
    return Question(type="choice", instructions=text, criteria=options)


SGD_SERVICES = {"Hotels": "Hotel search or booking", "Trains": "Train tickets", "Flights": "Flights",
                "Events": "Concerts, sports or other event tickets", "Movies": "Movies and cinema tickets",
                "RideSharing": "A taxi or ride-share"}

QUESTIONS: dict[str, Question] = {
    # table
    "table_statement_true": _noul("Is the statement true according to the table?"),
    # conversation
    "dialogue_service": _choice("Which service is the user dealing with in this conversation?", SGD_SERVICES),
    # agent
    "should_call_function": _noul("Should the assistant call one of the available functions to handle the user's request?"),
    "tool_call_matches_request": _noul("Does the proposed function call do what the user asked for?"),
    "goal_completed": _noul("Has the user's goal been fully completed by the steps taken?"),
    # quality
    "answer_helpfulness": Question(type="score", instructions="How helpful is the assistant's answer to the user's request?",
                                   criteria=["not helpful", "partly helpful", "very helpful"]),
    "answer_correct": _noul("Is the assistant's answer factually correct?"),
    # records
    "income_over_50k": _noul("Does this person earn more than $50,000 a year?"),
    "refund_eligible": _noul("Is this refund request eligible under the policy?"),
    "refund_reason": _choice("What decides this refund request?", {
        "eligible": "It meets every condition", "outside_return_window": "It is past the return window",
        "excluded_category": "The category can't be returned", "item_used": "The item has been used"}),
    "over_daily_limit": _noul("Has today's total spending gone over the customer's daily limit?"),
    "foreign_transaction": _noul("Was any of today's transactions made outside the customer's home country?"),
    "access_allowed": _noul("Is this user allowed to read this resource under the policy?"),
    "ticket_priority": Question(type="score", instructions="What priority does this ticket have under the rules?",
                                criteria=["low", "medium", "high"]),
    # logs
    "service_failing": _noul("Is any service failing according to the rule?"),
    "failing_service": _choice("Which service is failing?", {
        "auth": None, "payments": None, "search": None, "inventory": None, "notifications": None,
        "none": "No service is failing"}),
}

FAMILY = {**{t.qid: "text" for t in TEXT_TASKS},
          "table_statement_true": "table", "dialogue_service": "conversation",
          "should_call_function": "agent", "tool_call_matches_request": "agent", "goal_completed": "agent",
          "answer_helpfulness": "quality", "answer_correct": "quality",
          "income_over_50k": "records", "refund_eligible": "records", "refund_reason": "records",
          "over_daily_limit": "records", "foreign_transaction": "records", "access_allowed": "records",
          "ticket_priority": "records", "service_failing": "logs", "failing_service": "logs"}


# ------------------------------------------------------------ public data
def _sgd_row(r: dict[str, Any]):
    svc = (r.get("service") or "").split("_")[0]
    ctx = r.get("context") or []
    if svc not in SGD_SERVICES or len(ctx) < 4:
        return None
    turns = [{("user" if i % 2 == 0 else "assistant"): str(t)[:300]} for i, t in enumerate(ctx[-10:])]
    return {"conversation": turns, "latest_user_message": str(r.get("prompt", ""))[:300]}, svc


def _helpsteer_helpful(r: dict[str, Any]):
    level = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}.get(r.get("helpfulness"))
    return ({"user_request": str(r["prompt"])[:1200], "assistant_answer": str(r["response"])[:1500]}, level)


def _helpsteer_correct(r: dict[str, Any]):
    c = r.get("correctness")
    label = 1 if c == 4 else 0 if c in (0, 1) else None
    return ({"user_request": str(r["prompt"])[:1200], "assistant_answer": str(r["response"])[:1500]}, label)


def _adult(r: dict[str, Any]):
    rec = {k.replace(".", "_"): v for k, v in r.items() if k not in ("fnlwgt", "income") and v != "?"}
    return {"person": rec}, int(str(r.get("income", "")).startswith(">50K"))


DATA_TASKS = [
    Task("answer_helpfulness", "Answer helpfulness", "nvidia/HelpSteer2 (validation)", "CC-BY-4.0",
         QUESTIONS["answer_helpfulness"], "nvidia/HelpSteer2", "default", "validation", _helpsteer_helpful, (0, 1, 2)),
    Task("answer_correct", "Answer correctness", "nvidia/HelpSteer2 (validation)", "CC-BY-4.0",
         QUESTIONS["answer_correct"], "nvidia/HelpSteer2", "default", "validation", _helpsteer_correct, (0, 1)),
    Task("income_over_50k", "Census income", "scikit-learn/adult-census-income", "CC0-1.0",
         QUESTIONS["income_over_50k"], "scikit-learn/adult-census-income", "default", "train", _adult, (0, 1)),
]

GLAIVE = "glaiveai/glaive-function-calling-v2"
_FN_RE = re.compile(r'"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"')


def _glaive_parse(r: dict[str, Any]):
    fns = [{"name": n, "description": d} for n, d in _FN_RE.findall(r.get("system", ""))]
    chat = r.get("chat", "")
    m = re.match(r"\s*USER:\s*(.*?)\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|$)", chat, re.S)
    if not fns or not m:
        return None
    user, reply = m.group(1).strip()[:600], m.group(2).strip()
    call = None
    cm = re.match(r"<functioncall>\s*(\{.*\})", reply, re.S)
    if cm:
        nm = re.search(r'"name":\s*"([^"]+)"', cm.group(1))
        am = re.search(r'"arguments":\s*\'(.*?)\'\s*\}', cm.group(1), re.S)
        if not nm:
            return None
        try:
            args = json.loads(am.group(1)) if am else {}
        except json.JSONDecodeError:
            return None
        call = {"name": nm.group(1), "arguments": args}
    return {"functions": fns, "user": user, "call": call}


async def _glaive_tasks(client: httpx.AsyncClient, n: int, rng: random.Random) -> list[dict[str, Any]]:
    from systemone_builder.system_one.bench import hf_get

    parsed: list[dict[str, Any]] = []
    total = (await hf_get(client, "/rows", {"dataset": GLAIVE, "config": "default", "split": "train",
                                            "offset": 0, "length": 1}))["num_rows_total"]
    offsets = rng.sample(range(0, total - 100, 100), 40)
    for off in offsets:
        try:
            rows = (await hf_get(client, "/rows", {"dataset": GLAIVE, "config": "default", "split": "train",
                                                    "offset": off, "length": 100}))["rows"]
        except RuntimeError:
            continue
        parsed += [p for p in (_glaive_parse(x["row"]) for x in rows) if p]
        if len([p for p in parsed if p["call"]]) >= 2 * n and len([p for p in parsed if not p["call"]]) >= n:
            break
    calls = [p for p in parsed if p["call"]]
    declines = [p for p in parsed if not p["call"]]
    rng.shuffle(calls)
    rng.shuffle(declines)
    half = n // 2
    items = []
    for p in calls[:half] + declines[:half]:
        items.append({"qid": "should_call_function", "label": int(p["call"] is not None),
                      "state": {"available_functions": p["functions"], "user_request": p["user"]}})
    pool = calls[half:]
    for i, p in enumerate(pool[: 2 * half]):
        if i % 2 == 0:
            proposed, fns, label = p["call"], p["functions"], 1
        else:
            other = next((q for q in pool if q["call"]["name"] != p["call"]["name"]), None)
            if other is None:
                continue
            proposed, label = other["call"], 0
            fns = p["functions"] + [f for f in other["functions"] if f["name"] == other["call"]["name"]]
        items.append({"qid": "tool_call_matches_request", "label": label,
                      "state": {"available_functions": fns, "user_request": p["user"], "proposed_call": proposed}})
    return items


SGD = "GEM/schema_guided_dialog"


# Keywords per SGD service, including services outside SGD_SERVICES, used to keep only samples whose
# visible conversation is about the labelled service and nothing else.
SGD_KEYWORDS = {
    "Hotels": r"\bhotels?\b|\bstay\b|\bcheck.?in\b", "Trains": r"\btrains?\b",
    "Flights": r"\bflights?\b|\bairlines?\b|\bfly\b", "Events": r"\bconcerts?\b|\bevents?\b|\bgames?\b|\bmatch\b",
    "Movies": r"\bmovies?\b|\bfilms?\b|\btheaters?\b|\btheatres?\b|\bshowtimes?\b",
    "RideSharing": r"\bcab\b|\btaxi\b|\bride\b|\buber\b|\blyft\b",
    "_other": r"\bbus(es)?\b|\brestaurants?\b|\btable for\b|\bdoctor\b|\bdentist\b|\bsalon\b|\bhaircut\b"
              r"|\bapartments?\b|\brent(al)? (a )?car\b|\bcar rental\b|\bweather\b|\bsongs?\b|\bmusic\b"
              r"|\bbank\b|\btransfer\b|\bpayment\b|\balarm\b|\bhouse\b|\btravel\b|\battractions?\b",
}


def _sgd_clean(state: dict[str, Any], label: str) -> bool:
    text = " ".join([*(v for t in state["conversation"] for v in t.values()), state["latest_user_message"]]).lower()
    hits = {k for k, rx in SGD_KEYWORDS.items() if re.search(rx, text)}
    return hits == {label}


async def _sgd_tasks(client: httpx.AsyncClient, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Dialogue service, keeping only samples whose conversation is visibly about one service.

    The dataset's service field does not always match the visible conversation (a dialogue can book a
    bus, then search hotels, and be labelled Hotels throughout), so a sample is kept only when its
    labelled service's keywords appear and no other service's keywords do.
    """
    from systemone_builder.system_one.bench import hf_get

    total = (await hf_get(client, "/rows", {"dataset": SGD, "config": "default", "split": "test",
                                            "offset": 0, "length": 1}))["num_rows_total"]
    per = max(1, n // len(SGD_SERVICES))
    got: dict[str, list[dict[str, Any]]] = {svc: [] for svc in SGD_SERVICES}
    seen: set[str] = set()
    for off in rng.sample(range(0, total - 100, 100), min(150, total // 100 - 1)):
        if all(len(v) >= per for v in got.values()):
            break
        try:
            rows = (await hf_get(client, "/rows", {"dataset": SGD, "config": "default", "split": "test",
                                                    "offset": off, "length": 100}))["rows"]
        except RuntimeError:
            continue
        for x in rows:
            r = x["row"]
            out = _sgd_row(r)
            if not out or r["dialog_id"] in seen or len(got[out[1]]) >= per or not _sgd_clean(*out):
                continue
            seen.add(r["dialog_id"])
            got[out[1]].append({"qid": "dialogue_service", "label": out[1], "state": out[0]})
    return [x for v in got.values() for x in v]


TABFACT = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/master"


async def _tabfact(client: httpx.AsyncClient, n: int, rng: random.Random) -> list[dict[str, Any]]:
    r = await client.get(f"{TABFACT}/tokenized_data/test_examples.json")
    r.raise_for_status()
    tables = list(r.json().items())
    rng.shuffle(tables)
    got: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for table_id, (statements, labels, caption) in tables:
        if all(len(v) >= n // 2 for v in got.values()):
            break
        resp = await client.get(f"{TABFACT}/data/all_csv/{table_id}")
        if resp.status_code != 200:
            continue
        rows = list(csv.DictReader(io.StringIO(resp.text), delimiter="#"))
        if not rows or len(rows) > 12:
            continue
        k = rng.randrange(len(statements))
        lab = int(labels[k])
        if len(got[lab]) < n // 2:
            got[lab].append({"qid": "table_statement_true", "label": lab,
                             "state": {"table_caption": caption, "rows": rows, "statement": statements[k]}})
    return got[0] + got[1]


# --------------------------------------------------------------- generated
GENERATED = {  # kind: (primary question used for class balance, questions asked)
    "refund": ("refund_reason", ("refund_eligible", "refund_reason")),
    "transactions": ("over_daily_limit", ("over_daily_limit", "foreign_transaction")),
    "access": ("access_allowed", ("access_allowed",)),
    "ticket": ("ticket_priority", ("ticket_priority",)),
    "logs": ("service_failing", ("service_failing", "failing_service")),
    "agent_steps": ("goal_completed", ("goal_completed",)),
}


def _generated(n: int, seed: int) -> list[dict[str, Any]]:
    items = []
    for kind, (primary, qids) in GENERATED.items():
        cases = generate(kind, n * 8, seed)
        classes = sorted({c["labels"][primary] for c in cases}, key=str)
        per = max(1, n // len(classes))
        taken: dict[Any, int] = {c: 0 for c in classes}
        for c in cases:
            lab = c["labels"][primary]
            if taken[lab] < per:
                taken[lab] += 1
                items.append({"id": f"{kind}-{len(items)}", "state": c["state"],
                              "labels": {q: c["labels"][q] for q in qids}})
    return items


async def fetch_general(n_per_task: int, seed: int) -> list[dict[str, Any]]:
    from systemone_builder.system_one.multitask_suite import fetch_multitask

    rng = random.Random(seed)
    items = await fetch_multitask(n_per_task, seed)                       # the 14 text tasks
    async with httpx.AsyncClient(timeout=60) as client:
        for task in DATA_TASKS:
            for k, x in enumerate(await fetch_task(client, task, n_per_task, rng)):
                items.append({"id": f"{task.qid}-{k}", "state": x["state"], "labels": {task.qid: x["label"]}})
        extra = (await _sgd_tasks(client, n_per_task, rng) + await _glaive_tasks(client, n_per_task, rng)
                 + await _tabfact(client, n_per_task, rng))
        for k, x in enumerate(extra):
            items.append({"id": f"{x['qid']}-{k}", "state": x["state"], "labels": {x["qid"]: x["label"]}})
    items += _generated(n_per_task, seed)
    rng.shuffle(items)
    return items


async def general_suite(n_per_task: int, seed: int, cache_dir: Path | None):
    from systemone_builder.system_one.bench import Item, Suite

    cache = cache_dir / f"general-n{n_per_task}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = await fetch_general(n_per_task, seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
    questions = {**{t.qid: t.question for t in TEXT_TASKS}, **QUESTIONS}
    items = [Item(id=x["id"], state=x["state"], labels=x["labels"]) for x in raw]
    families = sorted(set(FAMILY.values()))
    desc = (f"general-purpose decisions: {len(questions)} questions in {len(families)} families "
            f"({', '.join(families)}), ~{n_per_task} items per task (seed {seed})")
    return Suite("general", desc, questions, items, gate=None)


def family_averages(report: dict[str, Any]) -> dict[str, float]:
    """Macro accuracy per family (score questions: exact level)."""
    acc: dict[str, list[float]] = {}
    for qid, q in report.get("questions", {}).items():
        v = q.get("accuracy", q.get("exact_level")) if q.get("n") else None
        if v is not None and qid in FAMILY:
            acc.setdefault(FAMILY[qid], []).append(v)
    return {f: sum(v) / len(v) for f, v in sorted(acc.items())}
