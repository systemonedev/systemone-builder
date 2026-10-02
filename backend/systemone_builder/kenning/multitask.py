"""Multi-task training data for the System One model.

A model trained on one task learns that task's answer habits: trained only on
phishing, it answered "yes" to almost any yes/no question. Decisions from many
domains, with balanced and deliberately unrelated questions, teach it to read
the question instead. Each source below is a public, labelled Hugging Face
dataset (fetched through the datasets-server API, no extra dependencies),
mapped onto noul / choice / score questions:

==================  ============  ==============================================
source              licence       teaches
==================  ============  ==============================================
amazon_polarity     Apache-2.0    sentiment as noul and choice
dbpedia_14          CC-BY-SA-3.0  topic choice over 4-10 of 14 classes; "is it about X?"
clinc_oos (plus)    CC-BY-3.0     intent routing over 5-15 of 150 intents, "other" bucket
boolq               CC-BY-SA-3.0  reading-comprehension yes/no questions
multi_nli           CC-BY-3.0 *   "does the text imply ...?" (noul) and 3-way choice
civil_comments      CC0-1.0       soft yes/no targets (annotator fractions), 4-level score
==================  ============  ==============================================

\\* multi_nli mixes per-genre licences (mostly CC-BY-3.0 / public); see its card.

Two augmentations target the "yes" bias directly:

* **balanced class questions**: "Is this about <class>?" is asked with the
  true class (yes) and with a wrong one (no) equally often;
* **genre questions** across sources: "Is this text a product review?" asked
  of an encyclopedia article, an assistant request, a comment ... (no unless
  it is one). Most questions about unrelated content must come out "no".

The out-of-domain suite in :mod:`systemone_builder.system_one.bench` uses tasks that are never
trained on (SMS spam, emotion, news topics) to measure generality.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from systemone_builder.system_one.bench import hf_get
MAX_CHARS = 1500

GENRES = {
    "amazon": "a customer review of a product",
    "dbpedia": "an encyclopedia article",
    "clinc": "a request someone typed to a virtual assistant",
    "boolq": "a factual reference passage",
    "nli": "a short passage of everyday text",
    "civil": "a comment posted on a news website",
}


@dataclass
class Source:
    key: str
    dataset: str
    config: str
    split: str
    license: str
    make: Callable[[dict[str, Any], "Ctx"], dict[str, Any] | None]
    accept: Callable[[dict[str, Any], "Ctx"], bool] = lambda row, ctx: True


@dataclass
class Ctx:
    rng: random.Random
    names: dict[str, list[str]]  # ClassLabel names per field


def _clip(text: str) -> str:
    return (text or "").strip()[:MAX_CHARS]


def _humanize(label: str) -> str:
    return label.replace("_", " ").replace("EducationalInstitution", "Educational institution") \
        .replace("MeanOfTransportation", "Means of transportation").replace("OfficeHolder", "Office holder") \
        .replace("NaturalPlace", "Natural place").replace("WrittenWork", "Written work")


def _genre_question(rng: random.Random, own: str) -> tuple[str, dict[str, Any], int]:
    """A noul asking whether the text is of some genre: its own one or another."""
    other = own if rng.random() < 0.35 else rng.choice([g for g in GENRES if g != own])
    return "q_genre", {"type": "noul", "instructions": f"Is this text {GENRES[other]}?"}, int(other == own)


def _row(state: Any, qs: dict[str, Any], targets: dict[str, Any], ctx: Ctx, genre: str) -> dict[str, Any]:
    if ctx.rng.random() < 0.3:
        qid, q, t = _genre_question(ctx.rng, genre)
        qs[qid], targets[qid] = q, t
    return {"state": state, "questions": qs, "targets": targets}


# ---------------------------------------------------------------- sources
def make_amazon(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    pos = ctx.names["label"][row["label"]] == "positive"
    r = ctx.rng
    state = {"review": {"title": _clip(row.get("title", "")), "text": _clip(row["content"])}}
    if r.random() < 0.5:
        ask_pos = r.random() < 0.5
        instr = r.choice(["Is this review positive?", "Is the customer happy with the product?",
                          "Would this reviewer recommend the product?"] if ask_pos else
                         ["Is this review negative?", "Is the customer unhappy with the product?",
                          "Is the reviewer complaining?"])
        qs = {"q_sentiment": {"type": "noul", "instructions": instr}}
        targets: dict[str, Any] = {"q_sentiment": int(pos == ask_pos)}
    else:
        opts = r.choice([("positive", "negative"), ("satisfied", "dissatisfied"), ("good experience", "bad experience")])
        items = [(opts[0], pos), (opts[1], not pos)]
        r.shuffle(items)
        qs = {"q_sentiment": {"type": "choice", "instructions": r.choice(["What is the sentiment of this review?",
                                                                          "How did the customer feel about the product?"]),
                              "criteria": {o: None for o, _ in items}}}
        targets = {"q_sentiment": next(o for o, ok in items if ok)}
    return _row(state, qs, targets, ctx, "amazon")


def make_dbpedia(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    r = ctx.rng
    names = [_humanize(n) for n in ctx.names["label"]]
    truth = names[row["label"]]
    state = {"article": {"title": row.get("title", ""), "text": _clip(row["content"])}}
    if r.random() < 0.5:
        k = r.randint(4, min(10, len(names)))
        opts = [truth] + r.sample([n for n in names if n != truth], k - 1)
        r.shuffle(opts)
        qs = {"q_topic": {"type": "choice", "instructions": r.choice(["What is this article about?",
                                                                      "Which category does this article belong to?"]),
                          "criteria": {o: None for o in opts}}}
        targets: dict[str, Any] = {"q_topic": truth}
    else:
        ask = truth if r.random() < 0.5 else r.choice([n for n in names if n != truth])
        qs = {"q_topic": {"type": "noul", "instructions": f"Is this article about a {ask.lower()}?"}}
        targets = {"q_topic": int(ask == truth)}
    return _row(state, qs, targets, ctx, "dbpedia")


def make_clinc(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    r = ctx.rng
    names = ctx.names["intent"]
    truth = names[row["intent"]]
    real = [n for n in names if n != "oos"]
    state = {"message": _clip(row["text"])}
    if r.random() < 0.65:
        include_truth = truth != "oos" and r.random() < 0.8
        pool = [n for n in real if n != truth]
        k = min(r.randint(5, 15), len(pool) + int(include_truth))
        opts = r.sample(pool, k - 1 if include_truth else k)
        if include_truth:
            opts.append(truth)
        r.shuffle(opts)
        crit = {_humanize(o): None for o in opts}
        other = r.choice(["other", "none of these", "something else"])
        crit[other] = "The request fits none of the other options"
        qs = {"q_intent": {"type": "choice", "instructions": r.choice(["What does the user want?", "Which handler should take this request?",
                                                                       "Route this request to the right intent."]),
                           "criteria": crit}}
        targets: dict[str, Any] = {"q_intent": _humanize(truth) if include_truth else other}
    else:
        if truth == "oos":
            return None
        ask = truth if r.random() < 0.5 else r.choice(real)
        qs = {"q_intent": {"type": "noul", "instructions": f"Is the user asking about: {_humanize(ask)}?"}}
        targets = {"q_intent": int(ask == truth)}
    return _row(state, qs, targets, ctx, "clinc")


def make_boolq(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    q = row["question"].strip()
    q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
    return _row({"passage": _clip(row["passage"])}, {"q_answer": {"type": "noul", "instructions": q}},
                {"q_answer": int(bool(row["answer"]))}, ctx, "boolq")


def make_nli(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    label = row["label"]  # 0 entailment, 1 neutral, 2 contradiction
    if label not in (0, 1, 2):
        return None
    hyp = row["hypothesis"].strip().rstrip(".")
    state = {"text": _clip(row["premise"])}
    if ctx.rng.random() < 0.6:
        qs = {"q_claim": {"type": "noul", "instructions": f"Does the text imply that {hyp[0].lower() + hyp[1:]}?"}}
        targets: dict[str, Any] = {"q_claim": int(label == 0)}
    else:
        crit = {"supported": "The text says or clearly implies it",
                "contradicted": "The text says the opposite",
                "not stated": "The text does not say either way"}
        items = list(crit.items())
        ctx.rng.shuffle(items)
        qs = {"q_claim": {"type": "choice", "instructions": f"Is this claim supported by the text: \"{hyp}\"?",
                          "criteria": dict(items)}}
        targets = {"q_claim": ["supported", "not stated", "contradicted"][label]}
    return _row(state, qs, targets, ctx, "nli")


TOXIC_LEVELS = ["Not toxic", "Somewhat rude or disrespectful", "Clearly toxic", "Severely toxic, hateful or threatening"]


def civil_level(row: dict[str, Any]) -> int:
    t = row["toxicity"]
    return 0 if t < 0.2 else 1 if t < 0.5 else 2 if t < 0.8 or row.get("severe_toxicity", 0) < 0.2 else 3


def make_civil(row: dict[str, Any], ctx: Ctx) -> dict[str, Any] | None:
    r = ctx.rng
    state = {"comment": _clip(row["text"])}
    if r.random() < 0.6:
        field, instr = r.choice([("toxicity", "Is this comment toxic?"), ("insult", "Is this comment insulting?"),
                                 ("threat", "Does this comment contain a threat?"),
                                 ("obscene", "Is this comment obscene?")])
        # soft target: the fraction of annotators who said yes
        qs = {"q_tox": {"type": "noul", "instructions": instr}}
        targets: dict[str, Any] = {"q_tox": round(float(row[field]), 4)}
    else:
        qs = {"q_tox": {"type": "score", "instructions": "How toxic is this comment?", "criteria": TOXIC_LEVELS}}
        targets = {"q_tox": civil_level(row)}
    return _row(state, qs, targets, ctx, "civil")


def accept_civil(row: dict[str, Any], ctx: Ctx) -> bool:
    # civil_comments is ~90% clean: keep every toxic comment, a fraction of clean ones
    return row["toxicity"] >= 0.2 or ctx.rng.random() < 0.25


SOURCES = {
    "amazon": Source("amazon", "fancyzhx/amazon_polarity", "amazon_polarity", "train", "Apache-2.0", make_amazon),
    "dbpedia": Source("dbpedia", "fancyzhx/dbpedia_14", "dbpedia_14", "train", "CC-BY-SA-3.0", make_dbpedia),
    "clinc": Source("clinc", "clinc/clinc_oos", "plus", "train", "CC-BY-3.0", make_clinc),
    "boolq": Source("boolq", "google/boolq", "default", "train", "CC-BY-SA-3.0", make_boolq),
    "nli": Source("nli", "nyu-mll/multi_nli", "default", "train", "CC-BY-3.0 (mixed per genre)", make_nli),
    "civil": Source("civil", "google/civil_comments", "default", "train", "CC0-1.0", make_civil, accept_civil),
}


# --------------------------------------------------------------- fetching
async def class_names(client: httpx.AsyncClient, src: Source) -> dict[str, list[str]]:
    info = await hf_get(client, "/info", {"dataset": src.dataset, "config": src.config})
    feats = info.get("dataset_info", {}).get("features", {})
    return {k: v["names"] for k, v in feats.items() if isinstance(v, dict) and v.get("names")}


async def sample_source(client: httpx.AsyncClient, src: Source, n: int, rng: random.Random) -> list[dict[str, Any]]:
    ctx = Ctx(rng, await class_names(client, src))
    first = await hf_get(client, "/rows", {"dataset": src.dataset, "config": src.config, "split": src.split,
                                           "offset": 0, "length": 1})
    total = int(first.get("num_rows_total") or 0)
    pages = list(range(0, max(total - 100, 1), 100))
    rng.shuffle(pages)
    out: list[dict[str, Any]] = []
    for offset in pages:
        if len(out) >= n:
            break
        try:
            data = await hf_get(client, "/rows", {"dataset": src.dataset, "config": src.config, "split": src.split,
                                                  "offset": offset, "length": 100})
        except (RuntimeError, httpx.HTTPError):
            continue  # skip a page the server keeps refusing
        rows = [x["row"] for x in data.get("rows", [])]
        rng.shuffle(rows)
        for row in rows[:25]:  # spread the sample over many pages
            if len(out) >= n:
                break
            if src.accept(row, ctx):
                made = src.make(row, ctx)
                if made:
                    made["source"] = src.key
                    out.append(made)
    return out


async def multitask_rows(per_source: int, seed: int, phishing_file: Path | None = None,
                         phishing_rows: int = 1500, sources: list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"seed": seed, "sources": {}}
    async with httpx.AsyncClient(timeout=60) as client:
        for key in sources or list(SOURCES):
            src = SOURCES[key]
            got = await sample_source(client, src, per_source, random.Random(rng.random()))
            rows.extend(got)
            manifest["sources"][key] = {"dataset": src.dataset, "config": src.config, "split": src.split,
                                        "license": src.license, "rows": len(got)}
            print(f"[kenning-data] {key:<8} {len(got):>5} rows  ({src.dataset}, {src.license})", flush=True)
    if phishing_file and phishing_file.exists() and phishing_rows:
        ph = [json.loads(x) for x in phishing_file.read_text().splitlines() if x.strip()]
        rng.shuffle(ph)
        for x in ph[:phishing_rows]:
            x["source"] = "phishing"
        rows.extend(ph[:phishing_rows])
        manifest["sources"]["phishing"] = {"dataset": "zefang-liu/phishing-email-dataset", "license": "see dataset card",
                                           "rows": min(phishing_rows, len(ph)), "file": str(phishing_file)}
        print(f"[kenning-data] phishing {min(phishing_rows, len(ph)):>5} rows  (from {phishing_file})", flush=True)
    rng.shuffle(rows)
    yes = [t for r in rows for q, t in r["targets"].items() if r["questions"][q]["type"] == "noul"]
    manifest["rows"] = len(rows)
    manifest["noul_mean_target"] = round(sum(float(t) for t in yes) / max(len(yes), 1), 3)
    return rows, manifest
