"""Train Kenning-XL: LoRA fine-tune of a decoder so its constrained readout answers the System One
wire format. The loss is the KL/cross-entropy between the student's answer-token distribution at the
decision position and the target distribution (hard label, or Clef's soft label) -- i.e. we train the
exact one-pass readout that ``xl.py`` serves. No generation, no separate head beyond the LM head.

    python -m systemone_builder.kenning.train_xl --data rows.jsonl --out /models/kenning-xl-v0.6 \
        --base Qwen/Qwen3-1.7B --epochs 1 --max-length 2048

Rows are the usual ``{state, questions, targets}``. The merged model is saved as a plain CausalLM
directory with ``kenning_xl.json`` (temperature, max_length), so ``KenningXL(<dir>)`` serves it directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from systemone_builder.kenning.model import make_deterministic
from systemone_builder.kenning.xl import CHOICE_LETTERS, CONFIG_FILE, _answer_tokens, _prompt, answer_cue, reason_prompt
from systemone_builder.system_one.contract import Question


def _gold_surface(q: Question, options: list[str], target: Any) -> str:
    """The gold answer surface form for deliberate (CoT) training."""
    if q.type == "noul":
        return "yes" if (float(target) >= 0.5) else "no"
    if q.type == "choice":
        best = max(target, key=target.get) if isinstance(target, dict) else target
        return CHOICE_LETTERS[options.index(best)]
    return str(int(max(target, key=target.get)) if isinstance(target, dict) else int(target))


def _deliberate_examples(path: Path, tok: Any, max_length: int) -> list[dict[str, Any]]:
    """Causal-LM examples: reason_prompt (masked) then ' <trace><cue> <answer>' (supervised)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trace = row.get("trace")
        if not trace:
            continue
        state = row["state"]
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ": "))
        for qid, target in row.get("targets", {}).items():
            spec = row["questions"].get(qid)
            if not spec:
                continue
            q = Question.model_validate(spec)
            legend = (list(q.criteria) if q.type == "choice"
                      else q.criteria if q.type == "score" else [])
            options = list(q.criteria) if q.type == "choice" else (
                [str(i) for i in range(len(q.criteria))] if q.type == "score" else ["yes", "no"])
            rp = reason_prompt(state_text, q)
            completion = f" {trace}{answer_cue(q, legend)} {_gold_surface(q, options, target)}"
            prompt_ids = tok(rp, truncation=True, max_length=max_length)["input_ids"]
            full_ids = tok(rp + completion, truncation=True, max_length=max_length)["input_ids"]
            if len(full_ids) <= len(prompt_ids):
                continue
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            out.append({"input_ids": full_ids, "labels": labels[:len(full_ids)]})
    return out


def target_probs(q: Question, options: list[str], target: Any) -> list[float]:
    """Target distribution aligned to the readout pieces (noul: yes,no; choice: options; score: levels)."""
    if q.type == "noul":
        t = float(target)
        return [t, 1.0 - t]
    if isinstance(target, dict):
        keys = options if q.type == "choice" else [str(i) for i in range(len(options))]
        v = [float(target.get(k, 0.0)) for k in keys]
        s = sum(v) or 1.0
        return [x / s for x in v]
    if q.type == "choice":
        return [1.0 if o == target else 0.0 for o in options]
    return [1.0 if i == int(target) else 0.0 for i in range(len(options))]


def _examples(path: Path, tok: Any, max_length: int) -> list[dict[str, Any]]:
    """Flatten rows into (input_ids, answer_token_ids, target) readout examples."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = row["state"]
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ": "))
        for qid, target in row.get("targets", {}).items():
            spec = row["questions"].get(qid)
            if not spec:
                continue
            q = Question.model_validate(spec)
            options = (list(q.criteria) if q.type == "choice"
                       else [str(i) for i in range(len(q.criteria))] if q.type == "score" else ["yes", "no"])
            prompt, pieces = _prompt(state_text, q, options if q.type == "choice" else
                                     (q.criteria if q.type == "score" else []))
            ids = tok(prompt, truncation=True, max_length=max_length)["input_ids"]
            ans_ids = _answer_tokens(tok, pieces)
            out.append({"input_ids": ids, "ans_ids": ans_ids,
                        "target": target_probs(q, options, target), "type": q.type})
    return out


def _temps(raw: list[tuple[str, list[float], list[float]]]) -> dict[str, float]:
    """Per type, the temperature minimising held-out NLL over the readout scores."""
    grid = [math.exp(x / 20) for x in range(-40, 61)]
    temps = {}
    for kind in {k for k, _, _ in raw}:
        rows = [(z, y) for k, z, y in raw if k == kind]

        def nll(t: float) -> float:
            s = 0.0
            for z, y in rows:
                m = max(v / t for v in z)
                e = [math.exp(v / t - m) for v in z]
                tot = sum(e)
                s -= sum(yi * math.log(max(ei / tot, 1e-12)) for yi, ei in zip(y, e))
            return s
        temps[kind] = round(min(grid, key=nll), 4)
    return temps


def main(argv: list[str] | None = None) -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--deliberate", action="store_true",
                    help="train causal-LM on gold reasoning traces then the answer (rows need a 'trace' field)")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)
    out = Path(a.out)
    make_deterministic()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.base)
    kw: dict[str, Any] = {"dtype": torch.bfloat16}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if not a.load_4bit:
        model.to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ex = (_deliberate_examples if a.deliberate else _examples)(Path(a.data), tok, a.max_length)
    random.Random(a.seed).shuffle(ex)
    n_val = max(50, int(len(ex) * a.val_fraction))
    val, train = ex[:n_val], ex[n_val:]
    print(f"[xl-train] {len(ex)} {'deliberate' if a.deliberate else 'readout'} examples: "
          f"{len(train)} train / {len(val)} val", flush=True)

    def forward_loss(e: dict[str, Any]) -> torch.Tensor:
        ids = torch.tensor([e["input_ids"]], device=model.device)
        if a.deliberate:
            labels = torch.tensor([e["labels"]], device=model.device)
            return model(input_ids=ids, labels=labels).loss
        logits = model(input_ids=ids).logits[0, -1, :]
        z = logits[torch.tensor(e["ans_ids"], device=model.device)].float()
        logp = torch.log_softmax(z, dim=-1)
        y = torch.tensor(e["target"], device=model.device, dtype=torch.float32)
        return -(y * logp).sum()

    @torch.inference_mode()
    def evaluate() -> tuple[float, list[tuple[str, list[float], list[float]]]]:
        model.eval()
        if a.deliberate:  # readout accuracy isn't meaningful here; report mean val LM loss as "accuracy proxy"
            losses = [forward_loss(e).item() for e in val]
            return sum(losses) / len(losses), []
        hit = 0
        raw = []
        for e in val:
            ids = torch.tensor([e["input_ids"]], device=model.device)
            z = model(input_ids=ids).logits[0, -1, :][torch.tensor(e["ans_ids"], device=model.device)].float().tolist()
            raw.append((e["type"], z, e["target"]))
            if max(range(len(z)), key=z.__getitem__) == max(range(len(e["target"])), key=e["target"].__getitem__):
                hit += 1
        return hit / len(val), raw

    acc0, _ = evaluate()
    print(f"[xl-train] zero-shot val {'loss' if a.deliberate else 'accuracy'} {acc0:.3f}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = int(len(train) * a.epochs)
    model.train()
    t0 = time.time()
    opt.zero_grad()
    for i in range(steps):
        e = train[i % len(train)]
        loss = forward_loss(e) / a.grad_accum
        loss.backward()
        if (i + 1) % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if i % (a.grad_accum * 20) == 0:
            peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
            print(f"[xl-train] step {i}/{steps} loss {loss.item() * a.grad_accum:.4f} ({time.time() - t0:.0f}s, {peak:.1f} GiB)", flush=True)
        model.train()
    acc1, raw = evaluate()
    temps = _temps(raw)
    print(f"[xl-train] trained val accuracy {acc1:.3f} (zero-shot {acc0:.3f}); temps {temps}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / CONFIG_FILE).write_text(json.dumps({
        "name": out.name, "base_model": a.base, "max_length": a.max_length, "temperature": temps,
        "trained_on": str(a.data), "val_accuracy": {"zero_shot": acc0, "trained": acc1},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))
    print(f"[xl-train] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
