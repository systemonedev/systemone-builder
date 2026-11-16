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
                usage = resp.usage.model_dump() if resp.usage else {"input_tokens": 0, "output_tokens": 0}
                return {"id": item.id, "answers": {k: a.model_dump() for k, a in resp.answers.items()},
                        "model": resp.model, "usage": usage,
                        "latency_ms": resp.latency_ms or (time.perf_counter() - t0) * 1000, "error": None}
            except (EngineError, AdapterError, httpx.HTTPError, ValueError) as exc:
                return {"id": item.id, "answers": None, "usage": None,
                        "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}

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


def roc_auc(ps: list[float], ys: list[int]) -> float | None:
    """Exact Wilcoxon-Mann-Whitney ROC-AUC with tie handling."""
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p_pos in pos:
        for p_neg in neg:
            if p_pos > p_neg:
                wins += 1.0
            elif p_pos == p_neg:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def pr_auc(ps: list[float], ys: list[int]) -> float | None:
    """Average precision (area under Precision-Recall curve)."""
    total_pos = sum(ys)
    if total_pos == 0 or total_pos == len(ys):
        return None
    ranked = sorted(zip(ps, ys), key=lambda x: x[0], reverse=True)
    tp = 0
    ap = 0.0
    for i, (_p, y) in enumerate(ranked, start=1):
        if y == 1:
            tp += 1
            ap += tp / i
    return ap / total_pos


def recall_at_fpr(ps: list[float], ys: list[int], max_fpr: float = 0.01) -> float | None:
    """Highest recall achievable while keeping False Positive Rate <= max_fpr."""
    total_pos = sum(ys)
    total_neg = len(ys) - total_pos
    if total_pos == 0 or total_neg == 0:
        return None
    thresholds = sorted(set(ps), reverse=True)
    best_recall = 0.0
    for tau in thresholds:
        tp = sum(1 for p, y in zip(ps, ys) if p >= tau and y == 1)
        fp = sum(1 for p, y in zip(ps, ys) if p >= tau and y == 0)
        if fp / total_neg <= max_fpr:
            best_recall = max(best_recall, tp / total_pos)
    return best_recall


def _pair_key(item: Item) -> str | None:
    if isinstance(item.state, dict) and item.state.get("pair_id"):
        return str(item.state["pair_id"])
    if len(item.id) >= 4 and item.id[-1] in ("a", "b") and item.id[-2].isdigit():
        return item.id[:-1]
    return None


def pairwise_flip_accuracy(suite: Suite, item_preds: dict[str, tuple[float, int]]) -> float | None:
    """Fraction of counterfactual twin pairs where the model gets BOTH twins right at 0.5."""
    by_pair: dict[str, list[tuple[float, int]]] = {}
    for it in suite.items:
        pk = _pair_key(it)
        if pk and it.id in item_preds:
            by_pair.setdefault(pk, []).append(item_preds[it.id])
    twins = [v for v in by_pair.values() if len(v) >= 2]
    if not twins:
        return None
    both_ok = sum(1 for pair in twins if all((p >= 0.5) == bool(y) for p, y in pair))
    return both_ok / len(twins)


# Approximate API pricing (USD per 1M input / 1M output tokens) for frontier cost comparison
MODEL_PRICING_PER_1M: list[tuple[str, float, float]] = [
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.00),
    ("o3-mini", 1.10, 4.40),
    ("gpt-5", 2.50, 10.00),
    ("claude-haiku", 0.80, 4.00),
    ("claude-sonnet", 3.00, 15.00),
    ("claude-3-7-sonnet", 3.00, 15.00),
    ("gemini-2.5-flash", 0.15, 0.60),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("jev", 0.20, 0.0),
    ("glide", 0.15, 0.0),
]


def estimate_cost_per_100k_usd(engine_name: str, model_name: str | None,
                               mean_in_tokens: float, mean_out_tokens: float) -> float:
    tag = f"{engine_name} {model_name or ''}".lower()
    for substr, in_rate, out_rate in MODEL_PRICING_PER_1M:
        if substr in tag:
            return round((mean_in_tokens * in_rate + mean_out_tokens * out_rate) / 10.0, 2)
    # Self-hosted Kenning / Clef on single 24GB GPU (~$0.50/hr at ~28 items/s -> ~$0.50 per 100k emails)
    if "kenning" in tag or "clef" in tag:
        return 0.50
    return 0.0


def question_metrics(qid: str, q: Question, suite: Suite, results: list[dict[str, Any]], hi: float, lo: float) -> dict[str, Any]:
    labels = {it.id: it.labels.get(qid) for it in suite.items}
    asked = sum(1 for it in suite.items if it.labels.get(qid) is not None)
    rows = [(r["id"], r["answers"][qid], labels[r["id"]]) for r in results
            if r["answers"] and qid in r["answers"] and labels.get(r["id"]) is not None]
    k_opts = 2 if q.type == "noul" else (len(q.criteria) if isinstance(q.criteria, (dict, list)) and len(q.criteria) >= 2 else 2)
    out: dict[str, Any] = {"type": q.type, "n": len(rows), "k_options": k_opts}
    if not rows:
        return out
    if q.type == "noul":
        triples = [(rid, NoulAnswer.model_validate(a).noul, _truth_bit(y)) for rid, a, y in rows]
        valid = [(rid, p, y) for rid, p, y in triples if y is not None]
        ps, ys = [p for _, p, _ in valid], [y for _, _, y in valid]
        pairs = list(zip(ps, ys))
        item_preds = {rid: (p, y) for rid, p, y in valid}
        acc = statistics.fmean(int((p >= 0.5) == bool(y)) for p, y in pairs)
        out.update(
            accuracy=acc,
            chance_corrected_score=round(((acc - 0.5) / 0.5) * 100.0 * (len(pairs) / max(asked, len(pairs), 1)), 2),
            roc_auc=roc_auc(ps, ys),
            pr_auc=pr_auc(ps, ys),
            recall_at_1pct_fpr=recall_at_fpr(ps, ys, 0.01),
            recall_at_5pct_fpr=recall_at_fpr(ps, ys, 0.05),
            pairwise_flip_accuracy=pairwise_flip_accuracy(suite, item_preds),
            brier=statistics.fmean((p - y) ** 2 for p, y in pairs),
            ece=ece(ps, [float(y) for y in ys]),
        )
        if qid == suite.gate:
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
        answers = [ChoiceAnswer.model_validate(a) for _, a, _ in rows]
        correct = [float(a.choice == y) for a, (_, _, y) in zip(answers, rows)]
        top = [max(a.probabilities.values()) for a in answers]
        acc = statistics.fmean(correct)
        chance = 1.0 / k_opts
        cc = ((acc - chance) / (1.0 - chance)) * 100.0 * (len(rows) / max(asked, len(rows), 1))
        out.update(accuracy=acc, chance_corrected_score=round(cc, 2), mean_top_probability=statistics.fmean(top), ece=ece(top, correct))
    else:
        answers = [ScoreAnswer.model_validate(a) for _, a, _ in rows]

        def picked(a: ScoreAnswer) -> int | None:
            return int(max(a.probabilities, key=lambda k: a.probabilities[k])) if a.probabilities else None

        acc = statistics.fmean(float(picked(a) == int(y)) for a, (_, _, y) in zip(answers, rows))
        chance = 1.0 / k_opts
        cc = ((acc - chance) / (1.0 - chance)) * 100.0 * (len(rows) / max(asked, len(rows), 1))
        out.update(accuracy=acc, chance_corrected_score=round(cc, 2),
                   mae=statistics.fmean(abs(a.score - float(y)) for a, (_, _, y) in zip(answers, rows)),
                   exact_level=statistics.fmean(float(round(a.score) == int(y)) for a, (_, _, y) in zip(answers, rows)))
    return out


def engine_report(name: str, suite: Suite, results: list[dict[str, Any]], wall_s: float, hi: float, lo: float,
                  det: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in results if r["error"] is None]
    lat = sorted(r["latency_ms"] for r in ok)

    def pct(p: float) -> float | None:
        return lat[min(len(lat) - 1, math.ceil(p * len(lat)) - 1)] if lat else None

    errors = [r["error"] for r in results if r["error"]]
    models = sorted({r["model"] for r in ok if r.get("model")})
    model_str = models[0] if len(models) == 1 else ", ".join(models) or None
    in_toks = [int((r.get("usage") or {}).get("input_tokens") or 0) for r in ok]
    out_toks = [int((r.get("usage") or {}).get("output_tokens") or 0) for r in ok]
    mean_in = statistics.fmean(in_toks) if in_toks else 0.0
    mean_out = statistics.fmean(out_toks) if out_toks else 0.0
    q_metrics = {qid: question_metrics(qid, q, suite, results, hi, lo) for qid, q in suite.questions.items()}
    cc_vals = [qm["chance_corrected_score"] for qm in q_metrics.values() if qm.get("chance_corrected_score") is not None]
    rep: dict[str, Any] = {
        "engine": name,
        "model": model_str,
        "items": len(results), "errors": len(errors), "error_examples": sorted(set(errors))[:3],
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "mean": statistics.fmean(lat) if lat else None},
        "usage": {
            "mean_input_tokens": round(mean_in, 1),
            "mean_output_tokens": round(mean_out, 1),
            "est_cost_per_100k_usd": estimate_cost_per_100k_usd(name, model_str, mean_in, mean_out),
        },
        "wall_s": wall_s, "throughput_per_s": len(results) / wall_s if wall_s else None,
        "determinism": det,
        "chance_corrected_macro": round(statistics.fmean(cc_vals), 2) if cc_vals else None,
        "questions": q_metrics,
    }
    if suite.name == "decision_index":
        from systemone_builder.system_one.decision_index_suite import decision_index_breakdown

        di = decision_index_breakdown(rep, suite)
        rep["decision_index_score"] = di["index_score"]
        rep["family_accuracy"] = {k: v / 100.0 for k, v in (di.get("area_scores") or {}).items()}
        rep["area_scores_100"] = di.get("area_scores") or {}
    return rep


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
    row("est. cost / 100k items ($)", [f"${_f((r.get('usage') or {}).get('est_cost_per_100k_usd'), '{:.2f}')}" for r in reports])
    row("identical on repeat", [f"{r['determinism']['identical']}/{r['determinism']['checked']}" for r in reports])
    if any("decision_index_score" in r for r in reports):
        row("DECISION INDEX (0..100)", [_f(r.get("decision_index_score"), "{:.1f}") for r in reports])
        for area in sorted({a for r in reports for a in (r.get("area_scores_100") or {})}):
            row(f"  area: {area}", [_f((r.get("area_scores_100") or {}).get(area), "{:.1f}") for r in reports])
    if any("chance_corrected_macro" in r for r in reports):
        row("chance-corrected macro (0..100)", [_f(r.get("chance_corrected_macro"), "{:.1f}") for r in reports])
    if any("macro_accuracy" in r for r in reports):
        row("MACRO ACCURACY (tasks)", [_f(r.get("macro_accuracy")) for r in reports])
        for fam in sorted({f for r in reports for f in (r.get("family_accuracy") or {})}):
            row(f"  family: {fam}", [_f((r.get("family_accuracy") or {}).get(fam)) for r in reports])
    per_question = suite.questions.items() if len(suite.questions) <= 60 else []
    for qid, q in per_question:
        m = [r["questions"][qid] for r in reports]
        lines.append(f"[{qid}] ({q.type})")
        if q.type == "noul":
            row("  accuracy @0.5", [_f(x.get("accuracy")) for x in m])
            if any(x.get("pairwise_flip_accuracy") is not None for x in m):
                row("  counterfactual pair flip", [_f(x.get("pairwise_flip_accuracy")) for x in m])
            if any(x.get("roc_auc") is not None for x in m):
                row("  ROC-AUC", [_f(x.get("roc_auc")) for x in m])
                row("  Recall @ 1% FPR", [_f(x.get("recall_at_1pct_fpr")) for x in m])
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
        try:
            results, wall = await run_engine(engine, suite, concurrency)
            det = await determinism(engine, suite, results, repeat_check) if repeat_check else {"checked": 0, "identical": 0}
        finally:
            await engine.aclose()
        report = engine_report(engine.name, suite, results, wall, hi, lo, det)
        if len(suite.questions) > 3:  # multi-task suites: one headline number
            from systemone_builder.system_one.multitask_suite import macro_average
            report["macro_accuracy"] = macro_average(report)
            if suite.name == "general":
                from systemone_builder.system_one.general_suite import family_averages
                report["family_accuracy"] = family_averages(report)
            elif suite.name == "workflowevals":
                from systemone_builder.system_one.workflowevals_suite import workflow_averages
                report["family_accuracy"] = workflow_averages(report)
        out.reports.append(report)
        out.raw[engine.name] = results
    return out
