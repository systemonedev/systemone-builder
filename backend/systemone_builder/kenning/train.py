"""Fine-tune a System One model on labelled or distilled decisions.

Input JSONL, one decision context per line::

    {"state": "..." | {...},
     "questions": {qid: {type, instructions, criteria}},
     "targets": {qid: target}}

A target is a hard label (noul: 0/1 or bool; choice: option name; score:
level index) or a soft distribution from a teacher (noul: probability of
yes; choice: {option: p}; score: {"0": p, "1": p, ...}). Soft targets let
the model learn a teacher's uncertainty, not just its answer.

Each (line, question) is one group of pairs; the loss is cross-entropy
between the target distribution and ``softmax(group scores)``. After
training, one temperature per question type is fitted on the held-out split
(minimum NLL), which is what makes the probabilities calibrated. The
held-out metrics before training (zero-shot) and after are written to
``kenning.json`` next to the weights (safetensors).

    python -m systemone_builder.kenning.train --data train.jsonl --out /workspace/kenning/models/my-model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from systemone_builder.system_one.contract import Question
from systemone_builder.kenning.model import CONFIG_FILE, DEFAULT_BASE, TEMPLATE_VERSION, entailment_index, hypotheses, make_deterministic, pair_scores


# ------------------------------------------------------------------- data
def micro_batches(groups: list[dict[str, Any]], max_pairs: int) -> list[list[dict[str, Any]]]:
    """Split a step's groups into runs of at most ``max_pairs`` pairs (a group is never split)."""
    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    n = 0
    for g in groups:
        k = len(g["hyps"])
        if cur and n + k > max_pairs:
            out.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += k
    if cur:
        out.append(cur)
    return out


def target_vector(q: Question, options: list[Any], target: Any) -> list[float]:
    n = len(options)
    if q.type == "noul":
        p = float(target)  # bool, 0/1 or a probability of yes
        return [p, 1.0 - p]
    if isinstance(target, dict):  # soft distribution from a teacher
        keys = [str(o) for o in options] if q.type == "choice" else [str(i) for i in range(n)]
        v = [float(target.get(k, 0.0)) for k in keys]
        s = sum(v)
        if s <= 0:
            raise ValueError("empty target distribution")
        return [x / s for x in v]
    idx = options.index(target) if q.type == "choice" else int(target)
    if not 0 <= idx < n:
        raise ValueError(f"target {target!r} out of range")
    return [1.0 if i == idx else 0.0 for i in range(n)]


def load_groups(path: Path) -> list[dict[str, Any]]:
    from systemone_builder.system_one.contract import render_state

    groups = []
    for line_no, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        premise = render_state(row["state"])
        for qid, qd in row["questions"].items():
            if qid not in row.get("targets", {}):
                continue
            q = Question.model_validate(qd)
            texts, options = hypotheses(q)
            groups.append({"row": line_no, "type": q.type, "premise": premise, "hyps": texts,
                           "target": target_vector(q, options, row["targets"][qid])})
    return groups


# ---------------------------------------------------------------- metrics
def evaluate(model: Any, tok: Any, groups: list[dict[str, Any]], pos: int, max_length: int, device: str,
             temps: dict[str, float] | None = None) -> tuple[dict[str, Any], list[tuple[str, list[float], list[float]]]]:
    """Accuracy, NLL, Brier, ECE on groups; also returns raw scores for temperature fitting."""
    import torch

    raw: list[tuple[str, list[float], list[float]]] = []
    model.eval()
    with torch.inference_mode():
        for g in groups:
            enc = tok([g["premise"]] * len(g["hyps"]), g["hyps"], truncation="only_first", max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            s = pair_scores(model, model(**enc).logits, pos).tolist()
            raw.append((g["type"], s, g["target"]))
    return metrics(raw, temps or {}), raw


def _softmax(z: list[float], t: float) -> list[float]:
    m = max(x / t for x in z)
    e = [math.exp(x / t - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def metrics(raw: list[tuple[str, list[float], list[float]]], temps: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in [*sorted({k for k, _, _ in raw}), "all"]:
        rows = [r for r in raw if kind == "all" or r[0] == kind]
        if not rows:
            continue
        acc = nll = brier = 0.0
        conf: list[float] = []
        hit: list[float] = []
        for k, s, y in rows:
            p = _softmax(s, temps.get(k, 1.0))
            top = max(range(len(p)), key=p.__getitem__)
            truth = max(range(len(y)), key=y.__getitem__)
            acc += float(top == truth)
            nll -= sum(yi * math.log(max(pi, 1e-12)) for yi, pi in zip(y, p))
            brier += sum((pi - yi) ** 2 for pi, yi in zip(p, y))
            conf.append(p[top])
            hit.append(float(top == truth))
        n = len(rows)
        ece = 0.0
        for b in range(10):
            idx = [i for i, c in enumerate(conf) if b / 10 <= c < (b + 1) / 10 or (b == 9 and c == 1.0)]
            if idx:
                ece += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(hit[i] for i in idx) / len(idx))
        out[kind] = {"n": n, "accuracy": acc / n, "nll": nll / n, "brier": brier / n, "ece": ece}
    return out


def fit_temperatures(raw: list[tuple[str, list[float], list[float]]]) -> dict[str, float]:
    """Per question type, the temperature with the lowest held-out NLL."""
    grid = [math.exp(x / 20) for x in range(-60, 61)]  # 0.05 .. 20
    temps: dict[str, float] = {}
    for kind in {k for k, _, _ in raw}:
        rows = [(s, y) for k, s, y in raw if k == kind]

        def nll(t: float) -> float:
            return -sum(sum(yi * math.log(max(pi, 1e-12)) for yi, pi in zip(y, _softmax(s, t))) for s, y in rows)

        temps[kind] = round(min(grid, key=nll), 4)
    return temps


# ---------------------------------------------------------------- training
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base", default=DEFAULT_BASE, help="HF id or path of the cross-encoder to start from")
    p.add_argument("--name", default=None, help="model name reported in responses (default: --out's folder name)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-groups", type=int, default=8, help="question groups per optimizer step")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-pairs", type=int, default=48,
                   help="(state, answer) pairs per forward pass; a step is split into micro-batches of at most this "
                        "many pairs (same gradient, bounded memory)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-grad-checkpointing", action="store_true", help="faster, but needs far more VRAM")
    p.add_argument("--max-vram-fraction", type=float, default=0.85,
                   help="hard cap on this process's share of GPU memory. Over it, training fails with an OOM error "
                        "instead of spilling into shared system memory (slow, and risky on WSL2). 0 = no cap")
    a = p.parse_args(argv)

    # Before torch initialises CUDA: growable segments avoid the fragmentation that makes a
    # capped process run out of memory with gigabytes reserved but unused.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    make_deterministic()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and a.max_vram_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(a.max_vram_fraction)

    groups = load_groups(a.data)
    rows = sorted({g["row"] for g in groups})
    random.Random(a.seed).shuffle(rows)
    val_rows = set(rows[: max(1, int(len(rows) * a.val_fraction))])
    train = [g for g in groups if g["row"] not in val_rows]
    val = [g for g in groups if g["row"] in val_rows]
    print(f"[kenning-train] {len(groups)} question groups from {len(rows)} rows: {len(train)} train / {len(val)} held out",
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, dtype=torch.float32).to(device)
    pos = entailment_index(model)

    before, _ = evaluate(model, tok, val, pos, a.max_length, device)
    print(f"[kenning-train] zero-shot held-out: {json.dumps(before['all'])}", flush=True)

    steps = max(1, math.ceil(len(train) * a.epochs / a.batch_groups))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    order: list[dict[str, Any]] = []
    while len(order) < steps * a.batch_groups:
        chunk = train[:]
        random.shuffle(chunk)
        order.extend(chunk)
    if not a.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    t0 = time.time()
    skipped = 0
    for step in range(steps):
        batch = order[step * a.batch_groups:(step + 1) * a.batch_groups]
        try:
            total = 0.0
            for micro in micro_batches(batch, a.max_pairs):
                premises = [g["premise"] for g in micro for _ in g["hyps"]]
                hyps = [h for g in micro for h in g["hyps"]]
                enc = tok(premises, hyps, truncation="only_first", max_length=a.max_length, padding=True,
                          return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(**enc).logits
                scores = pair_scores(model, logits, pos)
                loss = torch.zeros((), device=device)
                i = 0
                for g in micro:
                    n = len(g["hyps"])
                    logp = torch.log_softmax(scores[i:i + n], dim=-1)
                    loss = loss - (torch.tensor(g["target"], device=device) * logp).sum()
                    i += n
                # Same gradient as one big batch: every group is weighted 1/len(batch).
                loss = loss / len(batch)
                loss.backward()
                total += loss.item()
                del enc, logits, scores, loss
        except torch.OutOfMemoryError:
            # A rare oversized step (very long states x many options): skip it, don't lose the run.
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
            print(f"[kenning-train] step {step + 1}: out of memory, skipped ({skipped} so far)", flush=True)
            if skipped > max(10, steps // 100):
                raise SystemExit("[kenning-train] too many out-of-memory steps; lower --max-pairs or --max-length")
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == steps - 1:
            peak = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device == "cuda" else ""
            print(f"[kenning-train] step {step + 1}/{steps} loss {total:.4f} ({time.time() - t0:.0f}s{peak})", flush=True)

    after_raw_metrics, raw = evaluate(model, tok, val, pos, a.max_length, device)
    temps = fit_temperatures(raw)
    after = metrics(raw, temps)
    print(f"[kenning-train] trained held-out (T=1): {json.dumps(after_raw_metrics['all'])}", flush=True)
    print(f"[kenning-train] calibrated held-out (T={temps}): {json.dumps(after['all'])}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    (a.out / CONFIG_FILE).write_text(json.dumps({
        "name": a.name or a.out.name,
        "base_model": a.base,
        "template_version": TEMPLATE_VERSION,
        "positive_index": pos,
        "max_length": a.max_length,
        "temperature": temps,
        "trained_on": str(a.data),
        "train_groups": len(train),
        "heldout": {"zero_shot": before, "trained": after_raw_metrics, "calibrated": after},
        "hyperparameters": {k: v for k, v in vars(a).items() if k in ("epochs", "lr", "batch_groups", "seed")},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))
    print(f"[kenning-train] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
