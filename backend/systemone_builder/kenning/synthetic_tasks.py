"""Label-conditioned synthetic decision tasks, written by a local teacher model.

Clean-licence training needs breadth without CC-BY-SA corpora (DBpedia, BoolQ...).
Here the teacher is asked to write a text *for* given labels ("a support ticket
about billing, urgency: high"), so every label is known by construction, never
guessed by the teacher. Each text then yields choice, noul and score questions.

Tasks are deliberately unlike the out-of-domain benchmark (SMS spam, emotion,
news topics) so that benchmark keeps measuring generality.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Attr:
    name: str
    question: list[str]          # instruction variants
    labels: dict[str, str]       # label -> description
    ordinal: bool = False        # ask as a score (labels in order, lowest first)


@dataclass
class Task:
    name: str
    state_key: str
    writes: str                  # what the teacher writes
    attrs: list[Attr] = field(default_factory=list)


TASKS = [
    Task("support_ticket", "ticket", "a customer support ticket for a software company", [
        Attr("department", ["Which team should handle this ticket?", "Route this ticket to a team."],
             {"billing": "charges, refunds, invoices, plans", "technical": "bugs, errors, outages, integrations",
              "account": "login, password, profile, cancellation", "sales": "pricing questions, upgrades, quotes",
              "feedback": "suggestions or praise, no problem to fix"}),
        Attr("urgency", ["How urgent is this ticket?", "What priority should this ticket get?"],
             {"low": "no time pressure", "medium": "should be handled this week",
              "high": "blocking the customer's work today", "critical": "outage or money loss right now"}, ordinal=True),
    ]),
    Task("app_review", "review", "an app store review", [
        Attr("type", ["What kind of feedback is this review?", "Classify this review."],
             {"bug report": "something is broken", "feature request": "asks for something new",
              "praise": "positive, nothing to fix", "complaint": "unhappy about price, ads or policy"}),
        Attr("rating", ["How many stars does this review probably give?"],
             {"1 star": "very negative", "2 stars": "negative", "3 stars": "mixed", "4 stars": "positive",
              "5 stars": "very positive"}, ordinal=True),
    ]),
    Task("workplace_message", "message", "a short workplace chat message", [
        Attr("intent", ["What does the sender want?", "What is the purpose of this message?"],
             {"request": "asks someone to do something", "question": "asks for information",
              "update": "shares status or news", "scheduling": "arranges a meeting or time",
              "social": "small talk or thanks"}),
        Attr("needs_reply", ["Does this message need a reply?"],
             {"yes": "the sender is waiting for an answer", "no": "informational, no reply expected"}),
    ]),
    Task("contract_clause", "clause", "a single clause from a business contract", [
        Attr("clause_type", ["What type of clause is this?"],
             {"payment": "fees, invoicing, payment terms", "termination": "ending the agreement",
              "confidentiality": "protecting non-public information", "liability": "limits of damages",
              "intellectual property": "ownership and licences of work", "governing law": "jurisdiction and disputes"}),
    ]),
    Task("code_review", "comment", "a code review comment on a pull request", [
        Attr("severity", ["How serious is the issue raised?", "How much does this comment block merging?"],
             {"nit": "style or naming, optional", "minor": "small improvement, should fix",
              "major": "a real bug or design problem", "blocker": "security issue or data loss, must fix"}, ordinal=True),
        Attr("category", ["What is this review comment about?"],
             {"style": "formatting, naming", "correctness": "logic errors, edge cases", "performance": "speed, memory",
              "security": "vulnerabilities, secrets, injection", "tests": "missing or weak tests"}),
    ]),
    Task("log_event", "event", "a single application or server log line with a short message", [
        Attr("level", ["How severe is this log event?"],
             {"debug": "diagnostic detail", "info": "normal operation", "warning": "unexpected but handled",
              "error": "an operation failed", "critical": "service down or data at risk"}, ordinal=True),
        Attr("security_relevant", ["Is this event relevant to security?"],
             {"yes": "authentication, access, attacks, permissions", "no": "ordinary operations"}),
    ]),
    Task("restaurant_review", "review", "a restaurant review", [
        Attr("aspect", ["What is the review mainly about?"],
             {"food": "taste, dishes", "service": "staff, waiting", "price": "value for money",
              "ambience": "atmosphere, noise, decor", "hygiene": "cleanliness"}),
        Attr("sentiment", ["How does the reviewer feel overall?"],
             {"very negative": "", "negative": "", "neutral": "", "positive": "", "very positive": ""}, ordinal=True),
    ]),
    Task("job_posting", "posting", "a short job posting", [
        Attr("seniority", ["What seniority level is this role?"],
             {"intern": "", "junior": "", "mid-level": "", "senior": "", "lead or principal": ""}, ordinal=True),
        Attr("remote", ["Is this role fully remote?"], {"yes": "work from anywhere", "no": "on-site or hybrid"}),
    ]),
    Task("customer_message", "message", "a message from a customer to a subscription business", [
        Attr("churn_risk", ["Is the customer likely to cancel?", "Is this customer at risk of leaving?"],
             {"yes": "frustrated, comparing competitors, asking how to cancel", "no": "satisfied or neutral"}),
        Attr("tone", ["How heated is the message?"],
             {"calm": "", "annoyed": "", "angry": "", "abusive": ""}, ordinal=True),
    ]),
    Task("medical_admin", "request", "an administrative request to a clinic's front desk (no diagnosis)", [
        Attr("request_type", ["What is the patient asking for?"],
             {"appointment": "book, move or cancel a visit", "prescription refill": "repeat medication",
              "records": "copies of medical records", "billing": "insurance or invoice questions",
              "other": "anything else"}),
    ]),
]

STYLES = ["very short", "short", "detailed", "formal", "casual", "with a typo or two", "terse"]


def _prompt(task: Task, labels: dict[str, str], style: str) -> str:
    want = "; ".join(f"{a}: {v}" for a, v in labels.items())
    return (f"Write {task.writes}. It must clearly have these properties - {want}. Style: {style}. "
            "Do not name the properties or labels in the text itself. Return JSON with the text in 'text'.")


def rows_for(task: Task, text: str, labels: dict[str, str], rng: random.Random) -> dict[str, Any]:
    """Questions for one generated text, with targets from the requested labels."""
    qs: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    for attr in task.attrs:
        truth = labels[attr.name]
        names = list(attr.labels)
        instr = rng.choice(attr.question)
        if set(names) == {"yes", "no"}:
            qs[attr.name] = {"type": "noul", "instructions": instr}
            targets[attr.name] = int(truth == "yes")
        elif attr.ordinal and rng.random() < 0.6:
            qs[attr.name] = {"type": "score", "instructions": instr,
                             "criteria": [f"{n}: {d}" if d else n for n, d in attr.labels.items()]}
            targets[attr.name] = names.index(truth)
        elif rng.random() < 0.7:
            items = list(attr.labels.items())
            rng.shuffle(items)
            qs[attr.name] = {"type": "choice", "instructions": instr,
                             "criteria": {n: (d or None) if rng.random() < 0.7 else None for n, d in items}}
            targets[attr.name] = truth
        else:  # "is it X?" with the true or a wrong label, half and half
            ask = truth if rng.random() < 0.5 else rng.choice([n for n in names if n != truth])
            qs[attr.name] = {"type": "noul", "instructions": f"{instr.rstrip('?')} - is the answer \"{ask}\"?"}
            targets[attr.name] = int(ask == truth)
    return {"state": {task.state_key: text}, "questions": qs, "targets": targets}


async def generate(base_url: str, model: str, per_task: int, seed: int, concurrency: int = 32,
                   api_key: str | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    jobs = []
    for task in TASKS:
        for _ in range(per_task):
            labels = {a.name: rng.choice(list(a.labels)) for a in task.attrs}
            jobs.append((task, labels, rng.choice(STYLES)))
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def one(client: httpx.AsyncClient, i: int, task: Task, labels: dict[str, str], style: str) -> dict[str, Any] | None:
        body = {"model": model, "temperature": 0.9, "top_p": 0.95, "seed": seed * 100_003 + i, "max_tokens": 300,
                "messages": [{"role": "user", "content": _prompt(task, labels, style)}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "text", "schema": schema}}}
        async with sem:
            try:
                r = await client.post("/chat/completions", json=body)
                text = json.loads(r.json()["choices"][0]["message"]["content"])["text"].strip()
            except (httpx.HTTPError, KeyError, ValueError, TypeError, AttributeError):
                return None
        if len(text) < 15:
            return None
        return {"task": task.name, "text": text[:1500], "labels": labels}

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=180, headers=headers) as client:
        out = await asyncio.gather(*(one(client, i, *job) for i, job in enumerate(jobs)))
    seen: set[str] = set()
    result = []
    for x in out:
        if x and x["text"] not in seen:
            seen.add(x["text"])
            result.append(x)
    return result


def to_training_rows(texts: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_name = {t.name: t for t in TASKS}
    rng = random.Random(seed)
    return [rows_for(by_name[x["task"]], x["text"], x["labels"], rng) for x in texts if x["task"] in by_name]
