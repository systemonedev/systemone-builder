"""Kenning-XL: a decoder LM as a System One decision model via constrained one-pass readout.

The 435M cross-encoder (``model.py``) scores each option with an independent entailment pass over <=512
tokens, so it cannot compute over the state (sum amounts, scan a table column, count log lines). Kenning-XL
answers the same wire format with a small decoder LM instead: it renders (state, question) as a prompt and
reads the **next-token distribution at one decision position**, restricted to the answer tokens. One forward
pass, no generation, deterministic, temperature-calibrated per type -- but the single pass runs the full LM
over the whole (4k+) state, so latent reasoning over numbers, tables and long logs is now possible.

    noul    -> "... Answer (yes or no): "  read P over {" yes"," no"}
    choice  -> "... Answer (A/B/...): "     read P over the per-option letter tokens, mapped back to options
    score   -> "... Answer (0..k): "        read P over the level-index tokens

An optional ``deliberate`` mode first generates a short reasoning trace, then reads the answer at the same
position -- a latency/accuracy dial for the hardest questions. Off by default.

This module is inference + readout only; QLoRA distillation training lives in ``train_xl.py``.
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

from systemone_builder.kenning.model import make_deterministic, read_config
from systemone_builder.system_one.contract import (
    Question, SystemOneRequest, SystemOneResponse, Usage,
    choice_answer, describe, noul_answer, score_answer,
)

DEFAULT_XL = "Qwen/Qwen3-1.7B"
CONFIG_FILE = "kenning_xl.json"
CHOICE_LETTERS = string.ascii_uppercase  # A, B, C ... one sentinel token per option


def render_state(state: Any) -> str:
    import json
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ": "))


def _answer_tokens(tok: Any, pieces: list[str]) -> list[int]:
    """One token id per answer piece: the id of the piece's first sub-token with a leading space.

    Answer vocabularies are single surface tokens for a decoder (" yes", " A", " 0"), so the first
    sub-token's logit is the readout. Rendered with a leading space to match mid-sentence tokenization.
    """
    ids = []
    for p in pieces:
        enc = tok(" " + p, add_special_tokens=False)["input_ids"]
        ids.append(enc[0])
    return ids


def _prompt(state_text: str, q: Question, options: list[str]) -> tuple[str, list[str]]:
    """The decision prompt and the answer surface forms to read logits over."""
    instr = q.instructions.strip()
    if q.type == "noul":
        body = f"{state_text}\n\nQuestion: {instr}\nAnswer with yes or no.\nAnswer:"
        return body, ["yes", "no"]
    if q.type == "choice":
        lines = "\n".join(f"{CHOICE_LETTERS[i]}. {o}" + (f" - {describe(q.criteria[o])}" if isinstance(q.criteria, dict) and q.criteria.get(o) else "")
                          for i, o in enumerate(options))
        body = f"{state_text}\n\nQuestion: {instr}\nOptions:\n{lines}\nAnswer with the letter.\nAnswer:"
        return body, [CHOICE_LETTERS[i] for i in range(len(options))]
    # score: ordered levels 0..k
    legend = "; ".join(f"{i} = {describe(lv)}" for i, lv in enumerate(options))
    body = f"{state_text}\n\nQuestion: {instr}\nRate on a scale ({legend}).\nAnswer with the number.\nAnswer:"
    return body, [str(i) for i in range(len(options))]


class KenningXL:
    """A decoder LM served as a System One decision model. One request at a time (thread-safe)."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_XL, device: str | None = None,
                 max_length: int | None = None, load_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        make_deterministic()
        self.path = str(path)
        self.config: dict[str, Any] = read_config(self.path) if Path(self.path).is_dir() else {}
        # calibration lives in kenning_xl.json when present
        xlcfg = {}
        cfg_file = Path(self.path) / CONFIG_FILE
        if cfg_file.exists():
            import json
            xlcfg = json.loads(cfg_file.read_text())
        self.temperature: dict[str, float] = {"noul": 1.0, "choice": 1.0, "score": 1.0, **xlcfg.get("temperature", {})}
        self.name = xlcfg.get("name") or Path(self.path).name or self.path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or int(xlcfg.get("max_length", 4096))
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        kw: dict[str, Any] = {"dtype": torch.bfloat16 if self.device.startswith("cuda") else torch.float32}
        if load_4bit and self.device.startswith("cuda"):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **kw)
        if not load_4bit:
            self.model.to(self.device)
        self.model.eval()
        self._lock = __import__("threading").Lock()

    def _readout(self, prompt: str, pieces: list[str]) -> list[float]:
        """Probabilities over the answer pieces from one forward pass at the final position."""
        import torch
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[0, -1, :]  # next-token logits at the decision position
        ids = _answer_tokens(self.tokenizer, pieces)
        z = logits[ids].float()
        return z.tolist()

    def system_one(self, req: SystemOneRequest) -> SystemOneResponse:
        import math
        import time

        import torch
        t0 = time.perf_counter()
        state_text = render_state(req.state)
        answers: dict[str, Any] = {}
        with self._lock:
            for qid, q in req.questions.items():
                options = (list(q.criteria) if q.type == "choice"
                           else [str(i) for i in range(len(q.criteria))] if q.type == "score" else ["yes", "no"])
                prompt, pieces = _prompt(state_text, q, options if q.type == "choice" else
                                         (q.criteria if q.type == "score" else []))
                scores = self._readout(prompt, pieces)
                t = self.temperature.get(q.type, 1.0)
                m = max(s / t for s in scores)
                probs = [math.exp(s / t - m) for s in scores]
                s = sum(probs)
                probs = [p / s for p in probs]
                if q.type == "noul":
                    answers[qid] = noul_answer(probs[0], probs[1])
                elif q.type == "choice":
                    answers[qid] = choice_answer(options, probs)
                else:
                    answers[qid] = score_answer(q.criteria, probs)
        _ = torch  # keep torch import meaningful for the forward pass above
        return SystemOneResponse(model=self.name, answers=answers, usage=Usage(),
                                 latency_ms=(time.perf_counter() - t0) * 1000)
