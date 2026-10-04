"""Multi-task benchmark: many kinds of decisions, none of them trained on by Kenning.

Each task is a held-out split of a permissively licensed public dataset. Items are fetched
from the Hugging Face datasets-server (banking77 from its GitHub release) with a fixed seed,
balanced by class, and cached locally. Nothing is redistributed. The suite reports
per-task accuracy and the macro average across tasks (the headline number when comparing
engines).

    systemone bench --suite multi --engines kenning,clef -n 50      # n = items per task

To add a task: append a Task to TASKS (dataset, split, row -> (state, label), question).
Only use data whose licence allows it, and never a dataset Kenning trains on.
"""

from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from systemone_builder.system_one.contract import Question

RowFn = Callable[[dict[str, Any]], "tuple[Any, Any] | None"]


@dataclass(frozen=True)
class Task:
    qid: str
    title: str
    source: str
    licence: str
    question: Question
    dataset: str
    config: str
    split: str
    row: RowFn                 # row -> (state, label), or None to skip the row
    classes: tuple[Any, ...]   # labels to balance over


def _clip(text: Any, n: int = 1200) -> str:
    return str(text or "").strip()[:n]


def _noul(instructions: str) -> Question:
    return Question(type="noul", instructions=instructions)


def _choice(instructions: str, criteria: dict[str, str | None]) -> Question:
    return Question(type="choice", instructions=instructions, criteria=criteria)


FIN = {0: "bearish", 1: "bullish", 2: "neutral"}
EMOTIONS = ("gratitude", "anger", "fear", "sadness", "amusement", "curiosity")
GO_NAMES = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"]
DBPEDIA = {0: "Company", 2: "Artist", 3: "Athlete", 6: "Building", 9: "Animal", 12: "Film"}
BANKING = {
    "card_arrival": "Waiting for a new card to arrive",
    "lost_or_stolen_card": "Their card is lost or stolen",
    "exchange_rate": "Exchange rates or currency conversion",
    "cash_withdrawal_charge": "A fee charged for withdrawing cash",
    "pending_card_payment": "A card payment still shown as pending",
    "top_up_failed": "Adding money to the account didn't work",
    "change_pin": "Changing their PIN",
    "refund_not_showing_up": "A refund they expected hasn't appeared",
}


def _toxicity_level(row: dict[str, Any]) -> tuple[Any, Any] | None:
    prompt = row.get("prompt") or {}
    tox = prompt.get("toxicity")
    if tox is None:
        return None
    level = 0 if tox < 0.1 else 1 if 0.45 <= tox <= 0.65 else 2 if tox > 0.85 else None
    return ({"text": _clip(prompt.get("text"), 600)}, level) if level is not None else None


def _go_emotion(row: dict[str, Any]) -> tuple[Any, Any] | None:
    labels = [GO_NAMES[i] for i in row.get("labels") or []]
    if len(labels) == 1 and labels[0] in EMOTIONS:
        return {"post": _clip(row["text"])}, labels[0]
    return None


TASKS: list[Task] = [
    Task("answerable", "Question answering (yes/no)", "google/boolq", "CC-BY-SA-3.0",
         _noul("Based only on the passage, is the answer to the question yes?"),
         "google/boolq", "default", "validation",
         lambda r: ({"passage": _clip(r["passage"], 1500), "question": r["question"]}, int(bool(r["answer"]))), (0, 1)),
    Task("claim_supported", "Claim supported by evidence", "allenai/scitail", "Apache-2.0",
         _noul("Does the premise support the claim?"),
         "allenai/scitail", "snli_format", "test",
         lambda r: ({"premise": _clip(r["sentence1"]), "claim": _clip(r["sentence2"])},
                    {"entailment": 1, "neutral": 0}.get(r.get("gold_label"))), (0, 1)),
    Task("financial_sentiment", "Financial sentiment", "zeroshot/twitter-financial-news-sentiment", "MIT",
         _choice("What outlook does this post express for the stock or market?",
                 {"bearish": "Expects prices to fall", "bullish": "Expects prices to rise",
                  "neutral": "No clear direction"}),
         "zeroshot/twitter-financial-news-sentiment", "default", "validation",
         lambda r: ({"post": _clip(r["text"])}, FIN.get(r["label"])), tuple(FIN.values())),
    Task("toxic_comment", "Toxic comment", "OxAISH-AL-LLM/wiki_toxic", "CC0-1.0",
         _noul("Is this comment toxic: insulting, threatening, obscene or hateful?"),
         "OxAISH-AL-LLM/wiki_toxic", "default", "test",
         lambda r: ({"comment": _clip(r["comment_text"])}, int(r["label"])), (0, 1)),
    Task("counterfactual", "Counterfactual statement", "mteb/amazon_counterfactual (en)", "CC-BY-4.0",
         _noul("Does this review describe something that did not happen, like a wish or what would have been?"),
         "mteb/amazon_counterfactual", "en", "test",
         lambda r: ({"review": _clip(r["text"])}, int(r["label"])), (0, 1)),
    Task("prompt_injection", "Prompt injection", "deepset/prompt-injections", "Apache-2.0",
         _noul("Does this text try to override or manipulate an AI assistant's instructions?"),
         "deepset/prompt-injections", "default", "test",
         lambda r: ({"user_input": _clip(r["text"])}, int(r["label"])), (0, 1)),
    Task("jailbreak", "Jailbreak attempt", "jackhhao/jailbreak-classification", "Apache-2.0",
         _noul("Is this prompt trying to make an AI ignore its safety rules (a jailbreak)?"),
         "jackhhao/jailbreak-classification", "default", "test",
         lambda r: ({"prompt": _clip(r["prompt"])}, {"jailbreak": 1, "benign": 0}.get(r.get("type"))), (0, 1)),
    Task("fine_emotion", "Fine-grained emotion", "google-research-datasets/go_emotions", "Apache-2.0",
         _choice("Which emotion does the writer express?", {e: None for e in EMOTIONS}),
         "google-research-datasets/go_emotions", "simplified", "test", _go_emotion, EMOTIONS),
    Task("toxicity_level", "Toxicity level (Perspective-scored)", "allenai/real-toxicity-prompts", "Apache-2.0",
         Question(type="score", instructions="How toxic is this text?",
                  criteria=["not toxic", "somewhat toxic", "very toxic"]),
         "allenai/real-toxicity-prompts", "default", "train", _toxicity_level, (0, 1, 2)),
    Task("article_topic", "Encyclopedia topic", "fancyzhx/dbpedia_14 (test)", "CC-BY-SA-3.0",
         _choice("What is this encyclopedia article about?", {v: None for v in DBPEDIA.values()}),
         "fancyzhx/dbpedia_14", "dbpedia_14", "test",
         lambda r: ({"title": r["title"], "article": _clip(r["content"])}, DBPEDIA.get(r["label"])),
         tuple(DBPEDIA.values())),
    Task("banking_intent", "Banking intent routing", "PolyAI banking77 (test)", "CC-BY-4.0",
         _choice("What does the customer need?", BANKING),
         "github:PolyAI-LDN/task-specific-datasets/banking_data/test.csv", "", "test",
         # banking77 spells one intent "Refund_not_showing_up"
         lambda r: ({"message": _clip(r["text"])},
                    r["category"].lower() if r["category"].lower() in BANKING else None),
         tuple(BANKING)),
    # tasks shared with the `ood` suite
    Task("spam", "SMS spam", "ucirvine/sms_spam", "CC-BY-4.0", _noul("Is this message spam?"),
         "ucirvine/sms_spam", "plain_text", "train",
         lambda r: ({"message": _clip(r["sms"])}, int(r["label"])), (0, 1)),
    Task("emotion", "Emotion", "dair-ai/emotion", "see dataset card",
         _choice("Which emotion does the writer express?",
                 {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}),
         "dair-ai/emotion", "split", "test",
         lambda r: ({"post": _clip(r["text"])},
                    ["sadness", "joy", "love", "anger", "fear", "surprise"][r["label"]]),
         ("sadness", "joy", "love", "anger", "fear", "surprise")),
    Task("news_topic", "News topic", "fancyzhx/ag_news", "see dataset card",
         _choice("What is this news story about?",
                 {"World": "International news, politics, conflicts", "Sports": None,
                  "Business": "Companies, markets, the economy", "Sci/Tech": "Science and technology"}),
         "fancyzhx/ag_news", "default", "test",
         lambda r: ({"story": _clip(r["text"])}, ["World", "Sports", "Business", "Sci/Tech"][r["label"]]),
         ("World", "Sports", "Business", "Sci/Tech")),
]

HF_ROWS = "https://datasets-server.huggingface.co"


async def _hf_pages(client: httpx.AsyncClient, task: Task, rng: random.Random):
    from systemone_builder.system_one.bench import hf_get

    first = await hf_get(client, "/rows", {"dataset": task.dataset, "config": task.config, "split": task.split,
                                           "offset": 0, "length": 1})
    pages = list(range(0, max(int(first["num_rows_total"]) - 100, 1), 100))
    rng.shuffle(pages)
    failures = 0
    for offset in pages:
        try:
            rows = (await hf_get(client, "/rows", {"dataset": task.dataset, "config": task.config,
                                                    "split": task.split, "offset": offset, "length": 100})).get("rows", [])
        except RuntimeError:  # a page the server keeps refusing: sample from the others
            failures += 1
            if failures > 5:
                raise
            continue
        yield [x["row"] for x in rows]


async def _github_csv(client: httpx.AsyncClient, task: Task, rng: random.Random):
    _, rest = task.dataset.split(":", 1)
    owner, repo, path = rest.split("/", 2)
    r = await client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}")
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    rng.shuffle(rows)
    yield rows


async def fetch_task(client: httpx.AsyncClient, task: Task, n: int, rng: random.Random) -> list[dict[str, Any]]:
    per_class = max(1, n // len(task.classes))
    got: dict[Any, list[Any]] = {c: [] for c in task.classes}
    source = _github_csv if task.dataset.startswith("github:") else _hf_pages
    async for rows in source(client, task, rng):
        for row in rows:
            out = task.row(row)
            if out is None:
                continue
            state, label = out
            if label in got and len(got[label]) < per_class:
                got[label].append(state)
        if all(len(v) >= per_class for v in got.values()):
            break
    return [{"state": s, "label": lab} for lab, states in got.items() for s in states]


async def fetch_multitask(n_per_task: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for task in TASKS:
            for k, x in enumerate(await fetch_task(client, task, n_per_task, rng)):
                items.append({"id": f"{task.qid}-{k}", "state": x["state"], "labels": {task.qid: x["label"]}})
    rng.shuffle(items)
    return items


async def multitask_suite(n_per_task: int, seed: int, cache_dir: Path | None):
    from systemone_builder.system_one.bench import Item, Suite

    cache = cache_dir / f"multi-n{n_per_task}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = await fetch_multitask(n_per_task, seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
    items = [Item(id=x["id"], state=x["state"], labels=x["labels"]) for x in raw]
    desc = f"{len(TASKS)} decision tasks never trained on, ~{n_per_task} items each (seed {seed}): " + \
           ", ".join(t.qid for t in TASKS)
    return Suite("multi", desc, {t.qid: t.question for t in TASKS}, items, gate=None)


def macro_average(report: dict[str, Any]) -> float | None:
    """Mean per-task accuracy (score tasks: exact level) over the tasks that were answered."""
    vals = []
    for q in report.get("questions", {}).values():
        if not q.get("n"):
            continue
        v = q.get("accuracy", q.get("exact_level"))
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None
