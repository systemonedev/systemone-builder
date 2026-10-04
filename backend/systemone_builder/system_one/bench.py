"""System One benchmark: the same questions over the same labelled items, per engine.

What it measures is what decides whether a System One model can be trusted
to act on its own:

* **accuracy** and, for probabilities, **Brier score** and **ECE** (expected
  calibration error): a 0.9 should be right about 90% of the time.
* **gating** on a ``noul`` question, as a SOC or agent would use it: act
  automatically at ``p >= hi`` or ``p <= lo``, send the rest to a human or a
  System 2 model. Reports the automation rate and the errors made
  automatically (false positives acted on, false negatives auto-closed).
* **latency** (p50 / p95 per request) and **errors** (failed calls or
  answers that broke the schema).
* **determinism**: the first items are asked twice; identical answers on
  repeat is what "deterministic" has to mean in practice.

Suites are built in (``phishing``: the public zefang-liu/phishing-email-dataset,
balanced) or loaded from JSONL (``{"id", "state", "labels": {qid: truth}}``
per line) with a questions JSON file (``{qid: {type, instructions, criteria}}``).
Sampled items are cached so later runs and other engines see the same ones.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.adapters.base import AdapterError
from systemone_builder.system_one.contract import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer, SystemOneRequest
from systemone_builder.system_one.engines import EngineError, SystemOneEngine

HF_ROWS = "https://datasets-server.huggingface.co"


@dataclass
class Item:
    id: str
    state: Any
    labels: dict[str, Any]


@dataclass
class Suite:
    name: str
    description: str
    questions: dict[str, Question]
    items: list[Item]
    gate: str | None = None  # noul question used for automation metrics


# --------------------------------------------------------------- datasets
PHISHING_DATASET = "zefang-liu/phishing-email-dataset"
PHISHING_QUESTIONS = {
    "is_malicious": Question(type="noul", instructions="Is this email a phishing attempt, scam or other security threat?"),
    "category": Question(type="choice", instructions="What kind of email is this?", criteria={
        "Safe": "Legitimate personal or business email",
        "Phishing": "Phishing, scam, fraud, credential theft or other malicious email",
    }),
}


async def hf_get(client: httpx.AsyncClient, path: str, params: dict[str, Any], tries: int = 6) -> dict[str, Any]:
    """GET from the Hugging Face datasets-server, retrying rate limits and hiccups.

    The server answers 429 (or an empty / non-JSON body) when hit hard; back off
    and retry instead of failing a whole benchmark or dataset build.
    """
    delay = 2.0
    for attempt in range(tries):
        try:
            r = await client.get(f"{HF_ROWS}{path}", params=params)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
        except (httpx.TransportError, ValueError):
            if attempt == tries - 1:
                raise
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"datasets-server kept failing for {path} {params}")


async def _hf_rows(client: httpx.AsyncClient, offset: int, length: int = 100) -> tuple[list[dict[str, Any]], int]:
    data = await hf_get(client, "/rows", {"dataset": PHISHING_DATASET, "config": "default", "split": "train",
                                          "offset": offset, "length": length})
    return [x["row"] for x in data.get("rows", [])], int(data.get("num_rows_total") or 0)


async def fetch_phishing(n: int, seed: int) -> list[Item]:
    """``n`` emails, half phishing and half safe, sampled with ``seed``."""
    want = {"Phishing Email": n // 2, "Safe Email": n - n // 2}
    got: dict[str, list[dict[str, Any]]] = {k: [] for k in want}
    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=30) as client:
        _, total = await _hf_rows(client, 0, 1)
        pages = list(range(0, max(total - 100, 1), 100))
        rng.shuffle(pages)
        for offset in pages:
            if all(len(got[k]) >= want[k] for k in want):
                break
            rows, _ = await _hf_rows(client, offset)
            rng.shuffle(rows)
            for row in rows:
                kind, text = row.get("Email Type"), (row.get("Email Text") or "").strip()
                if kind in got and len(got[kind]) < want[kind] and len(text) >= 20:
                    got[kind].append(row)
    if any(len(got[k]) < want[k] for k in want):
        raise RuntimeError(f"could not sample {n} balanced emails from {PHISHING_DATASET}")
    items = []
    for kind, rows in got.items():
        for row in rows:
            bad = kind == "Phishing Email"
            items.append(Item(
                id=f"phish-{len(items)}",
                state={"email": {"recipient": "employee@internal-corp.com", "body": row["Email Text"].strip()[:1500]}},
                labels={"is_malicious": 1 if bad else 0, "category": "Phishing" if bad else "Safe"},
            ))
    rng.shuffle(items)
    return items


async def phishing_suite(n: int, seed: int, cache_dir: Path | None) -> Suite:
    cache = cache_dir / f"phishing-n{n}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        items = [Item(**x) for x in json.loads(cache.read_text())]
    else:
        items = await fetch_phishing(n, seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([x.__dict__ for x in items]))
    return Suite("phishing", f"{PHISHING_DATASET}, {n} emails (balanced, seed {seed})", PHISHING_QUESTIONS, items,
                 gate="is_malicious")


OOD_QUESTIONS = {
    "spam": Question(type="noul", instructions="Is this message spam?"),
    "emotion": Question(type="choice", instructions="Which emotion does the writer express?",
                        criteria={"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}),
    "news_topic": Question(type="choice", instructions="What is this news story about?",
                           criteria={"World": "International news, politics, conflicts", "Sports": None,
                                     "Business": "Companies, markets, the economy", "Sci/Tech": "Science and technology"}),
}
# (dataset, config, split, text field, label field, question id, state key)
OOD_TASKS = [
    ("ucirvine/sms_spam", "plain_text", "train", "sms", "label", "spam", "message"),
    ("dair-ai/emotion", "split", "test", "text", "label", "emotion", "post"),
    ("fancyzhx/ag_news", "default", "test", "text", "label", "news_topic", "story"),
]


async def fetch_ood(n_per_task: int, seed: int) -> list[Item]:
    """Labelled items from tasks the System One model is never trained on."""
    rng = random.Random(seed)
    items: list[Item] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for dataset, config, split, text_f, label_f, qid, key in OOD_TASKS:
            info = await hf_get(client, "/info", {"dataset": dataset, "config": config})
            names = info["dataset_info"]["features"][label_f]["names"]
            first = await hf_get(client, "/rows", {"dataset": dataset, "config": config, "split": split,
                                                   "offset": 0, "length": 1})
            pages = list(range(0, max(int(first["num_rows_total"]) - 100, 1), 100))
            rng.shuffle(pages)
            per_class = max(1, n_per_task // len(names))
            got: dict[int, list[str]] = {i: [] for i in range(len(names))}
            for offset in pages:
                if all(len(v) >= per_class for v in got.values()):
                    break
                rows = (await hf_get(client, "/rows", {"dataset": dataset, "config": config, "split": split,
                                                        "offset": offset, "length": 100})).get("rows", [])
                for x in rows:
                    lab, text = x["row"][label_f], (x["row"].get(text_f) or "").strip()
                    if text and len(got[lab]) < per_class:
                        got[lab].append(text[:1500])
            for lab, texts in got.items():
                for text in texts:
                    truth: Any = int(names[lab] == "spam") if qid == "spam" else names[lab]
                    items.append(Item(id=f"{qid}-{len(items)}", state={key: text}, labels={qid: truth}))
    rng.shuffle(items)
    return items


async def ood_suite(n_per_task: int, seed: int, cache_dir: Path | None) -> Suite:
    cache = cache_dir / f"ood-n{n_per_task}-seed{seed}.json" if cache_dir else None
    if cache and cache.exists():
        items = [Item(**x) for x in json.loads(cache.read_text())]
    else:
        items = await fetch_ood(n_per_task, seed)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([x.__dict__ for x in items]))
    desc = ", ".join(t[0] for t in OOD_TASKS)
    return Suite("ood", f"out-of-domain tasks never trained on ({desc}), ~{n_per_task} each, seed {seed}",
                 OOD_QUESTIONS, items, gate="spam")


LAYOUTS = ("original", "headers_json", "plain_text", "nested_metadata")


async def layout_suite(n: int, seed: int, cache_dir: Path | None) -> Suite:
    """The phishing benchmark emails, each in four layouts: is the answer layout-independent?

    Accuracy per layout appears as one question per layout (``malicious@<layout>``).
    """
    from systemone_builder.kenning.layouts import email_layouts

    base = await phishing_suite(n, seed, cache_dir)
    rng = random.Random(seed + 1)
    q = "Is this email a phishing attempt, scam or other security threat?"
    questions = {f"malicious@{name}": Question(type="noul", instructions=q) for name in LAYOUTS}
    items = []
    for it in base.items:
        truth = it.labels["is_malicious"]
        for name, state in email_layouts(it.state["email"]["body"], rng).items():
            items.append(Item(id=f"{it.id}@{name}", state=state, labels={f"malicious@{name}": truth}))
    return Suite("layouts", f"{base.description}, each in {len(LAYOUTS)} layouts ({', '.join(LAYOUTS)})", questions, items,
                 gate="malicious@headers_json")


def jsonl_suite(path: Path, questions_path: Path, gate: str | None = None) -> Suite:
    questions = {k: Question.model_validate(v) for k, v in json.loads(questions_path.read_text()).items()}
    items = []
    for i, line in enumerate(path.read_text().splitlines()):
        if line.strip():
            row = json.loads(line)
            items.append(Item(id=str(row.get("id", i)), state=row["state"], labels=row.get("labels") or {}))
    if gate is None:
        gate = next((k for k, q in questions.items() if q.type == "noul"), None)
    return Suite(path.stem, f"{path.name} ({len(items)} items)", questions, items, gate=gate)


# ------------------------------------------------------------------ running
async def run_engine(engine: SystemOneEngine, suite: Suite, concurrency: int) -> tuple[list[dict[str, Any]], float]:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: Item) -> dict[str, Any]:
        # mixed suites: each item is asked only the questions it has labels for
        qs = {k: suite.questions[k] for k in item.labels if k in suite.questions} or suite.questions
        req = SystemOneRequest(state=item.state, questions=qs)
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await engine.answer(req)
                return {"id": item.id, "answers": {k: a.model_dump() for k, a in resp.answers.items()},
                        "model": resp.model, "latency_ms": resp.latency_ms or (time.perf_counter() - t0) * 1000, "error": None}
            except (EngineError, AdapterError, httpx.HTTPError, ValueError) as exc:
                return {"id": item.id, "answers": None, "latency_ms": (time.perf_counter() - t0) * 1000,
                        "error": str(exc)}

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(it) for it in suite.items))
    return list(results), time.perf_counter() - t0


async def determinism(engine: SystemOneEngine, suite: Suite, first: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Ask the first ``k`` answered items again; count identical answers."""
    done = [r for r in first if r["answers"] is not None][:k]
    if not done:
        return {"checked": 0, "identical": 0}
    by_id = {it.id: it for it in suite.items}
    again, _ = await run_engine(engine, Suite(suite.name, "", suite.questions, [by_id[r["id"]] for r in done]), k)
    same = sum(1 for a, b in zip(done, again) if a["answers"] == b["answers"])
    return {"checked": len(done), "identical": same}


# ------------------------------------------------------------------ metrics
def ece(conf: list[float], correct: list[float], bins: int = 10) -> float | None:
    if not conf:
        return None
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if idx:
            err += len(idx) / total * abs(statistics.fmean(conf[i] for i in idx) - statistics.fmean(correct[i] for i in idx))
    return err


def _truth_bit(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str) and v.strip().lower() in ("yes", "true", "1", "no", "false", "0"):
        return int(v.strip().lower() in ("yes", "true", "1"))
    return None


def question_metrics(qid: str, q: Question, suite: Suite, results: list[dict[str, Any]], hi: float, lo: float) -> dict[str, Any]:
    labels = {it.id: it.labels.get(qid) for it in suite.items}
    rows = [(r["answers"][qid], labels[r["id"]]) for r in results
            if r["answers"] and qid in r["answers"] and labels.get(r["id"]) is not None]
    out: dict[str, Any] = {"type": q.type, "n": len(rows)}
    if not rows:
        return out
    if q.type == "noul":
        ps = [NoulAnswer.model_validate(a).noul for a, _ in rows]
        ys = [_truth_bit(y) for _, y in rows]
        pairs = [(p, y) for p, y in zip(ps, ys) if y is not None]
        ps, ys = [p for p, _ in pairs], [y for _, y in pairs]
        out.update(accuracy=statistics.fmean(int((p >= 0.5) == bool(y)) for p, y in pairs),
                   brier=statistics.fmean((p - y) ** 2 for p, y in pairs),
                   ece=ece(ps, [float(y) for y in ys]))
        if qid == suite.gate:
            asked = sum(1 for it in suite.items if it.labels.get(qid) is not None)
            auto_pos = [y for p, y in pairs if p >= hi]
            auto_neg = [y for p, y in pairs if p <= lo]
            fp, fn = auto_pos.count(0), auto_neg.count(1)
            automated = len(auto_pos) + len(auto_neg)
            out["gate"] = {
                "hi": hi, "lo": lo,
                "auto_positive": len(auto_pos), "auto_negative": len(auto_neg),
                "to_human": asked - automated,
                "automation_rate": automated / asked if asked else None,
                "false_positives_acted": fp, "false_negatives_closed": fn,
                "automated_accuracy": (automated - fp - fn) / automated if automated else None,
            }
    elif q.type == "choice":
        answers = [ChoiceAnswer.model_validate(a) for a, _ in rows]
        correct = [float(a.choice == y) for a, (_, y) in zip(answers, rows)]
        top = [max(a.probabilities.values()) for a in answers]
        out.update(accuracy=statistics.fmean(correct), mean_top_probability=statistics.fmean(top), ece=ece(top, correct))
    else:
        answers = [ScoreAnswer.model_validate(a) for a, _ in rows]
        out.update(mae=statistics.fmean(abs(a.score - float(y)) for a, (_, y) in zip(answers, rows)),
                   exact_level=statistics.fmean(float(round(a.score) == int(y)) for a, (_, y) in zip(answers, rows)))
    return out


def engine_report(name: str, suite: Suite, results: list[dict[str, Any]], wall_s: float, hi: float, lo: float,
                  det: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in results if r["error"] is None]
    lat = sorted(r["latency_ms"] for r in ok)

    def pct(p: float) -> float | None:
        return lat[min(len(lat) - 1, math.ceil(p * len(lat)) - 1)] if lat else None

    errors = [r["error"] for r in results if r["error"]]
    models = sorted({r["model"] for r in ok if r.get("model")})
    return {
        "engine": name,
        # what the engine served (e.g. kenning-large-v0.4), so runs can be compared later
        "model": models[0] if len(models) == 1 else ", ".join(models) or None,
        "items": len(results), "errors": len(errors), "error_examples": sorted(set(errors))[:3],
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "mean": statistics.fmean(lat) if lat else None},
        "wall_s": wall_s, "throughput_per_s": len(results) / wall_s if wall_s else None,
        "determinism": det,
        "questions": {qid: question_metrics(qid, q, suite, results, hi, lo) for qid, q in suite.questions.items()},
    }


# ------------------------------------------------------------------- output
def _f(v: Any, fmt: str = "{:.3f}") -> str:
    return "-" if v is None else fmt.format(v)


def format_reports(suite: Suite, reports: list[dict[str, Any]]) -> str:
    cols = [r["engine"] for r in reports]
    w = max(26, *(len(c) + 2 for c in cols))
    lines = [f"System One benchmark - suite '{suite.name}': {suite.description}", ""]

    def row(label: str, vals: list[str]) -> None:
        lines.append(f"{label:<30}" + "".join(f"{v:<{w}}" for v in vals))

    row("", cols)
    lines.append("-" * (30 + w * len(cols)))
    row("errors / items", [f"{r['errors']} / {r['items']}" for r in reports])
    row("latency p50 / p95 (ms)", [f"{_f(r['latency_ms']['p50'], '{:.0f}')} / {_f(r['latency_ms']['p95'], '{:.0f}')}" for r in reports])
    row("throughput (items/s)", [_f(r["throughput_per_s"], "{:.2f}") for r in reports])
    row("identical on repeat", [f"{r['determinism']['identical']}/{r['determinism']['checked']}" for r in reports])
    if any("macro_accuracy" in r for r in reports):
        row("MACRO ACCURACY (tasks)", [_f(r.get("macro_accuracy")) for r in reports])
        for fam in sorted({f for r in reports for f in (r.get("family_accuracy") or {})}):
            row(f"  family: {fam}", [_f((r.get("family_accuracy") or {}).get(fam)) for r in reports])
    for qid, q in suite.questions.items():
        m = [r["questions"][qid] for r in reports]
        lines.append(f"[{qid}] ({q.type})")
        if q.type == "noul":
            row("  accuracy @0.5", [_f(x.get("accuracy")) for x in m])
            row("  Brier (lower better)", [_f(x.get("brier")) for x in m])
            row("  ECE (lower better)", [_f(x.get("ece")) for x in m])
            if qid == suite.gate:
                g = [x.get("gate") or {} for x in m]
                hi, lo = next(((x["hi"], x["lo"]) for x in g if x), (None, None))
                row(f"  automated (p>={hi} or <={lo})", [_f(x.get("automation_rate"), "{:.1%}") for x in g])
                row("  accuracy when automated", [_f(x.get("automated_accuracy"), "{:.1%}") for x in g])
                row("  false positives acted on", [str(x.get("false_positives_acted", "-")) for x in g])
                row("  false negatives auto-closed", [str(x.get("false_negatives_closed", "-")) for x in g])
                row("  sent to human / System 2", [str(x.get("to_human", "-")) for x in g])
        elif q.type == "choice":
            row("  accuracy", [_f(x.get("accuracy")) for x in m])
            row("  mean top probability", [_f(x.get("mean_top_probability")) for x in m])
            row("  ECE (lower better)", [_f(x.get("ece")) for x in m])
        else:
            row("  MAE (levels)", [_f(x.get("mae")) for x in m])
            row("  exact level", [_f(x.get("exact_level")) for x in m])
    for r in reports:
        for e in r["error_examples"]:
            lines.append(f"! {r['engine']}: {e}")
    return "\n".join(lines)


@dataclass
class BenchResult:
    suite: Suite
    reports: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"suite": {"name": self.suite.name, "description": self.suite.description, "gate": self.suite.gate,
                          "questions": {k: q.model_dump(exclude_none=True) for k, q in self.suite.questions.items()},
                          "items": [it.__dict__ for it in self.suite.items]},
                "reports": self.reports, "results": self.raw, "ts": time.time()}


async def run_benchmark(suite: Suite, engines: list[SystemOneEngine], *, concurrency: int = 8, hi: float = 0.9,
                        lo: float = 0.1, repeat_check: int = 10) -> BenchResult:
    out = BenchResult(suite)
    for engine in engines:  # one engine at a time: no contention between them
        results, wall = await run_engine(engine, suite, concurrency)
        det = await determinism(engine, suite, results, repeat_check) if repeat_check else {"checked": 0, "identical": 0}
        report = engine_report(engine.name, suite, results, wall, hi, lo, det)
        if len(suite.questions) > 3:  # multi-task suites: one headline number
            from systemone_builder.system_one.multitask_suite import macro_average
            report["macro_accuracy"] = macro_average(report)
            if suite.name == "general":
                from systemone_builder.system_one.general_suite import family_averages
                report["family_accuracy"] = family_averages(report)
        out.reports.append(report)
        out.raw[engine.name] = results
    return out
