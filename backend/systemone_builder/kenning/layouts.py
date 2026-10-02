"""State-layout variation: the same content, laid out the way real callers send it.

Every phishing row Kenning v0.1 saw looked like ``{"email": {"recipient", "body"}}``.
Sent ``{"email": {"from", "subject", "body"}}`` instead, it called an obvious
credential-phishing email "Safe": the model had learned the layout along with the
task. Real callers send plain text, flat or nested JSON, headers, IDs and
timestamps, so training rows are re-rendered in varied layouts:

* wrapper and field names from synonyms (``email``/``message``/``mail``,
  ``body``/``text``/``content``...), key order shuffled;
* flat or nested, with or without a wrapper;
* plain text with ``Field: value`` headers instead of JSON;
* unrelated metadata (ids, timestamps, folders) to learn to ignore.

Synthetic header values (sender, subject, timestamps) are drawn independently of
the label, so they cannot leak the answer.
"""

from __future__ import annotations

import random
import re
from typing import Any

SYNONYMS = {
    "email": ["email", "message", "mail", "inbound_email", "msg"],
    "review": ["review", "customer_review", "feedback", "product_review"],
    "article": ["article", "document", "entry", "page"],
    "comment": ["comment", "post", "reply", "user_comment"],
    "passage": ["passage", "context", "document", "reference"],
    "text": ["text", "content", "body", "passage"],
    "body": ["body", "text", "content", "message_body", "email_body"],
    "title": ["title", "headline", "name", "subject"],
    "subject": ["subject", "title", "topic"],
    "from": ["from", "sender", "from_address", "reply_to"],
    "recipient": ["recipient", "to", "mailbox"],
}

SENDERS = ["alex.morgan", "billing", "no-reply", "support", "j.chen", "team", "notifications", "hr", "it-desk",
           "sam.rivera", "accounts", "info", "newsletter", "admin", "service"]
DOMAINS = ["gmail.com", "outlook.com", "company.com", "mail-service.net", "example.org", "corp-mail.io",
           "yahoo.com", "partners.biz", "secure-notify.com", "hotmail.com", "office-online.net", "inbox.org"]
FOLDERS = ["inbox", "INBOX", "priority", "updates", "external", "triage-queue"]


def _syn(rng: random.Random, key: str) -> str:
    return rng.choice(SYNONYMS.get(key, [key]))


def subject_from(body: str) -> str:
    """A subject line made from the body's first words (no label information)."""
    words = re.sub(r"\s+", " ", body.strip()).split(" ")
    return " ".join(words[: min(len(words), 8)]).strip(" ,.:;-")[:80] or "(no subject)"


def enrich_email(email: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Add a sender, subject and date that are independent of whether it is phishing."""
    out = dict(email)
    out.setdefault("from", f"{rng.choice(SENDERS)}@{rng.choice(DOMAINS)}")
    out.setdefault("subject", subject_from(out.get("body", "")))
    if rng.random() < 0.5:
        out["date"] = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z"
    return out


def _metadata(rng: random.Random) -> dict[str, Any]:
    pool = {
        "id": f"{rng.choice(['msg', 'evt', 'rec', 'tkt'])}_{rng.randint(10_000, 99_999)}",
        "received_at": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z",
        "folder": rng.choice(FOLDERS),
        "size_bytes": rng.randint(400, 90_000),
        "source": rng.choice(["api", "webhook", "imap", "export"]),
    }
    keys = rng.sample(list(pool), rng.randint(1, 3))
    return {k: pool[k] for k in keys}


def _rename(d: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    items = [(_syn(rng, k), v) for k, v in d.items()]
    rng.shuffle(items)
    out: dict[str, Any] = {}
    for k, v in items:
        while k in out:
            k = f"{k}_"
        out[k] = v
    return out


def as_text(fields: dict[str, Any], rng: random.Random) -> str:
    """Render fields as plain text: short ones as 'Key: value' headers, the long one as the body."""
    long_key = max(fields, key=lambda k: len(str(fields[k])))
    headers = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in fields.items() if k != long_key]
    rng.shuffle(headers)
    return ("\n".join(headers) + "\n\n" if headers else "") + str(fields[long_key])


def vary(state: Any, rng: random.Random) -> Any:
    """One random re-layout of a state (plain text states get metadata or headers around them)."""
    if isinstance(state, str):
        r = rng.random()
        if r < 0.4:
            return state
        if r < 0.7:
            return {_syn(rng, "text"): state, **_metadata(rng)}
        return {_syn(rng, "text"): state}
    if not isinstance(state, dict) or len(state) != 1:
        return state
    (wrapper, inner), = state.items()
    if not isinstance(inner, dict):
        inner = {"text": inner}
    if wrapper == "email":
        inner = enrich_email(inner, rng)
    style = rng.choice(["json_renamed", "json_renamed", "flat", "text", "nested_meta", "unwrapped"])
    fields = _rename(inner, rng)
    if style == "text":
        return as_text(fields, rng)
    if style == "flat":
        return {f"{_syn(rng, wrapper)}_{k}": v for k, v in fields.items()}
    if style == "unwrapped":
        return fields
    if style == "nested_meta":
        long_key = max(fields, key=lambda k: len(str(fields[k])))
        header = {k: v for k, v in fields.items() if k != long_key}
        return {_syn(rng, wrapper): {**_metadata(rng), "headers": header, "content": {long_key: fields[long_key]}}}
    return {_syn(rng, wrapper): {**fields, **(_metadata(rng) if rng.random() < 0.4 else {})}}


# Fixed layouts for measuring layout sensitivity on the same emails.
def email_layouts(body: str, rng: random.Random) -> dict[str, Any]:
    e = enrich_email({"body": body}, rng)
    return {
        "original": {"email": {"recipient": "employee@internal-corp.com", "body": body}},
        "headers_json": {"email": {"from": e["from"], "subject": e["subject"], "body": body}},
        "plain_text": f"From: {e['from']}\nSubject: {e['subject']}\n\n{body}",
        "nested_metadata": {"message": {"id": f"msg_{rng.randint(10_000, 99_999)}", "received_at": "2026-09-30T08:15:00Z",
                                        "headers": {"from": e["from"], "subject": e["subject"]},
                                        "content": {"text": body}}},
    }
