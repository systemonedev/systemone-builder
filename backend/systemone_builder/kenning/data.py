"""Training data for the System One model.

``phishing_training_rows`` samples labelled emails from the public
zefang-liu/phishing-email-dataset - excluding every email in the benchmark
suite's cache, so the model is never scored on what it trained on - and asks
about each one with varied questions: different wordings, option names and
order, and inverted questions ("is this safe?") with inverted targets. The
variation is what makes the model read the question instead of memorising a
single task.

Rows use :mod:`systemone_builder.kenning.train`'s JSONL format.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.system_one.bench import PHISHING_DATASET, _hf_rows

MALICIOUS_QUESTIONS = [
    "Is this email a phishing attempt, scam or other security threat?",
    "Is this email malicious?",
    "Is this a phishing or scam email?",
    "Does this email try to deceive the recipient (phishing, fraud or a scam)?",
    "Should this email be treated as a security threat?",
    "Is the sender trying to steal credentials, money or personal data?",
]
SAFE_QUESTIONS = [
    "Is this a legitimate, safe email?",
    "Is this email harmless?",
    "Can this email be trusted as a normal personal or business message?",
]
CHOICE_SETS = [
    ("What kind of email is this?", {"Safe": "Legitimate personal or business email",
                                     "Phishing": "Phishing, scam, fraud, credential theft or other malicious email"}),
    ("Classify this email.", {"Legitimate": None, "Malicious": None}),
    ("How should this email be triaged?", {"Deliver": "A normal, harmless email",
                                           "Quarantine": "A phishing, scam or fraudulent email"}),
    ("Which category fits this message best?", {"Benign": "Ordinary correspondence, newsletters, notices",
                                                "Phishing/Scam": "Tries to trick the reader into acting against their interest"}),
]
SAFE_OPTION = {"Safe", "Legitimate", "Deliver", "Benign"}


def body_hash(text: str) -> str:
    return hashlib.sha256(text.strip()[:1500].encode()).hexdigest()


def excluded_hashes(cache_dir: Path) -> set[str]:
    """Bodies of every cached benchmark item: never used for training."""
    out: set[str] = set()
    for f in cache_dir.glob("phishing-*.json"):
        for it in json.loads(f.read_text()):
            out.add(body_hash(it["state"]["email"]["body"]))
    return out


def questions_for(rng: random.Random, malicious: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    qs: dict[str, Any] = {}
    targets: dict[str, Any] = {}
    if rng.random() < 0.75:
        qs["q_threat"] = {"type": "noul", "instructions": rng.choice(MALICIOUS_QUESTIONS)}
        targets["q_threat"] = int(malicious)
    else:
        qs["q_safe"] = {"type": "noul", "instructions": rng.choice(SAFE_QUESTIONS)}
        targets["q_safe"] = int(not malicious)
    instr, crit = rng.choice(CHOICE_SETS)
    items = list(crit.items())
    rng.shuffle(items)
    qs["q_kind"] = {"type": "choice", "instructions": instr, "criteria": dict(items)}
    targets["q_kind"] = next(o for o, _ in items if (o in SAFE_OPTION) != malicious)
    return qs, targets


async def phishing_training_rows(n: int, seed: int, cache_dir: Path) -> list[dict[str, Any]]:
    skip = excluded_hashes(cache_dir)
    want = {"Phishing Email": n // 2, "Safe Email": n - n // 2}
    got: dict[str, list[str]] = {k: [] for k in want}
    seen: set[str] = set()
    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=30) as client:
        _, total = await _hf_rows(client, 0, 1)
        pages = list(range(0, max(total - 100, 1), 100))
        rng.shuffle(pages)
        for offset in pages:
            if all(len(got[k]) >= want[k] for k in want):
                break
            rows, _ = await _hf_rows(client, offset)
            for row in rows:
                kind, text = row.get("Email Type"), (row.get("Email Text") or "").strip()
                h = body_hash(text) if text else ""
                if kind in got and len(got[kind]) < want[kind] and len(text) >= 20 and h not in skip and h not in seen:
                    seen.add(h)
                    got[kind].append(text[:1500])
    out = []
    for kind, bodies in got.items():
        for body in bodies:
            qs, targets = questions_for(rng, kind == "Phishing Email")
            out.append({"state": {"email": {"recipient": "employee@internal-corp.com", "body": body}},
                        "questions": qs, "targets": targets})
    rng.shuffle(out)
    print(f"[kenning-data] {len(out)} rows from {PHISHING_DATASET} ({len(skip)} benchmark emails excluded)")
    return out
