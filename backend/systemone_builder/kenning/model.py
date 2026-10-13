"""Inference for the local System One model.

Each question becomes a group of (state, hypothesis) pairs, one per candidate
answer:

* noul   -> "<instructions> Answer: yes." / "... Answer: no."
* choice -> "<instructions> Answer: <option>. <description>"
* score  -> "<instructions> Answer: <level description>."

The backbone is a sequence-classification cross-encoder. With an NLI head the
pair score is the entailment log-odds; with a single-logit head it is that
logit. The group's distribution is ``softmax(scores / T[type])``, with ``T``
fitted on held-out data at training time. :func:`hypotheses` is shared with
training, so the model is always queried exactly as it was trained.

Determinism: one request is one batch (split into fixed-size chunks), runs
alone under a lock with deterministic kernels, and the softmax is done in
float32. Same request, same weights, same hardware -> same answer.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from systemone_builder.system_one.contract import (
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    choice_answer,
    describe,
    noul_answer,
    render_state,
    score_answer,
)

CONFIG_FILE = "kenning.json"
LEGACY_CONFIG_FILES = ("s1_config.json",)  # written by models trained before the Kenning rename
TEMPLATE_VERSION = 1
# Served when nothing else is configured: the published, calibrated Kenning (Apache-2.0).
DEFAULT_MODEL = "systemonedev/kenning-large-v0.5"
# Starting point for training: a zero-shot NLI model with no non-commercial data (MIT).
DEFAULT_BASE = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"


def hypotheses(q: Question) -> tuple[list[str], list[Any]]:
    """Hypothesis texts for one question and the options they stand for."""
    stem = q.instructions.strip()
    if q.type == "noul":
        # A {"true": ..., "false": ...} criteria dict describes the two *answers*, so each
        # hypothesis gets only its own side (as choice does). Folding both into both leaves the
        # pair identical apart from "yes."/"no.", which is no signal at all.
        crit = q.criteria
        if isinstance(crit, dict) and ({"true", "false"} & set(crit) or {"yes", "no"} & set(crit)):
            yes_d = describe(crit.get("true", crit.get("yes")))
            no_d = describe(crit.get("false", crit.get("no")))
            return ([f"{stem} Answer: yes." + (f" {yes_d}" if yes_d else ""),
                     f"{stem} Answer: no." + (f" {no_d}" if no_d else "")], ["yes", "no"])
        shared = describe(crit)
        suffix = f" ({shared})" if shared else ""
        return [f"{stem}{suffix} Answer: yes.", f"{stem}{suffix} Answer: no."], ["yes", "no"]
    if q.type == "choice":
        opts = list(q.criteria)  # type: ignore[arg-type]
        texts = []
        for o, d in q.criteria.items():  # type: ignore[union-attr]
            desc = describe(d)
            texts.append(f"{stem} Answer: {o}." + (f" {desc}" if desc else ""))
        return texts, opts
    levels = list(q.criteria)  # type: ignore[arg-type]
    return [f"{stem} Answer: {describe(lv)}." for lv in levels], levels


def make_deterministic() -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pair_scores(model: Any, logits: Any, positive_index: int) -> Any:
    """Entailment log-odds (NLI head) or the raw logit (single-logit head)."""
    import torch

    logits = logits.float()
    if logits.shape[-1] == 1:
        return logits[:, 0]
    pos = logits[:, positive_index]
    others = torch.cat([logits[:, :positive_index], logits[:, positive_index + 1:]], dim=-1)
    return pos - torch.logsumexp(others, dim=-1)


def read_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """A model directory's Kenning config ({} for a plain Hugging Face model)."""
    for name in (CONFIG_FILE, *LEGACY_CONFIG_FILES):
        f = Path(path) / name
        if f.exists():
            return json.loads(f.read_text())
    return {}


def local_dir(path: str) -> str:
    """A model directory on disk: ``path`` itself, or a Hugging Face repo id downloaded
    (weights, tokenizer and kenning.json; the config is what makes it calibrated)."""
    if Path(path).is_dir():
        return path
    from huggingface_hub import snapshot_download

    snap = snapshot_download(path, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    if not any(Path(snap).glob("*.safetensors")):  # older repos ship only PyTorch .bin weights
        snap = snapshot_download(path, allow_patterns=["*.json", "*.bin", "*.model", "*.txt"])
    return snap


def entailment_index(model: Any) -> int:
    label2id = {k.lower(): v for k, v in (getattr(model.config, "label2id", None) or {}).items()}
    return int(label2id.get("entailment", 0))


class Kenning:
    """A loaded System One model. Thread-safe; one request runs at a time."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_MODEL, device: str | None = None,
                 max_length: int | None = None, chunk: int = 32) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        make_deterministic()
        self.path = str(path)
        local = local_dir(self.path)
        self.config: dict[str, Any] = read_config(local)
        self.temperature: dict[str, float] = {"noul": 1.0, "choice": 1.0, "score": 1.0, **self.config.get("temperature", {})}
        # Decision bias, in post-temperature logit units, added to the first option ("yes"). Temperature
        # cannot move a two-option boundary -- p >= 0.5 is the same test as score_yes >= score_no -- so
        # this is the only term that can. Absent or 0.0 leaves behaviour exactly as before.
        self.bias: dict[str, float] = {"noul": 0.0, **(self.config.get("bias") or {})}
        self.name = self.config.get("name") or Path(self.path).name or self.path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.model = AutoModelForSequenceClassification.from_pretrained(local, dtype=dtype).to(self.device).eval()
        self.positive_index = int(self.config.get("positive_index", entailment_index(self.model)))
        self.max_length = max_length or int(self.config.get("max_length", 2048))
        self.chunk = chunk
        self._lock = threading.Lock()

    def _tok_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def premises_for(self, premise: str, groups: list[tuple[str, Question, list[Any], int, int]],
                     hyps: list[str]) -> list[str]:
        """One premise per hypothesis, truncated to a budget shared by every option of its question.

        ``truncation="only_first"`` trims the premise to whatever room each hypothesis leaves, so a
        long option description would see less of the state than a short one -- and the group's
        softmax would then compare scores taken over different evidence. Budgeting per question
        keeps every option on identical evidence and makes the answer independent of how verbosely
        the options happen to be written.
        """
        specials = self.tokenizer.num_special_tokens_to_add(pair=True)
        prem_ids = self.tokenizer(premise, add_special_tokens=False)["input_ids"]
        prems: list[str] = [premise] * len(hyps)
        for _qid, _q, _options, start, n in groups:
            budget = max(0, self.max_length - specials - max(self._tok_len(h) for h in hyps[start:start + n]))
            text = (premise if len(prem_ids) <= budget
                    else self.tokenizer.decode(prem_ids[:budget], skip_special_tokens=True))
            for j in range(start, start + n):
                prems[j] = text
        return prems

    def scores(self, premises: list[str], hyps: list[str]) -> tuple[list[float], int]:
        """Pair scores for aligned premises and hypotheses, plus input tokens used."""
        import torch

        out: list[float] = []
        tokens = 0
        with torch.inference_mode():
            for i in range(0, len(hyps), self.chunk):
                part = hyps[i:i + self.chunk]
                enc = self.tokenizer(premises[i:i + self.chunk], part, truncation="only_first",
                                     max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                tokens += int(enc["attention_mask"].sum())
                logits = self.model(**enc).logits
                out.extend(pair_scores(self.model, logits, self.positive_index).tolist())
        return out, tokens

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        t0 = time.perf_counter()
        premise = render_state(req.state)
        groups: list[tuple[str, Question, list[Any], int, int]] = []
        hyps: list[str] = []
        for qid, q in req.questions.items():
            texts, options = hypotheses(q)
            groups.append((qid, q, options, len(hyps), len(texts)))
            hyps.extend(texts)
        with self._lock:
            flat, tokens = self.scores(self.premises_for(premise, groups, hyps), hyps)
        answers: dict[str, Any] = {}
        for qid, q, options, start, n in groups:
            t = self.temperature.get(q.type, 1.0)
            z = [s / t for s in flat[start:start + n]]
            b = self.bias.get(q.type, 0.0)
            if b and q.type == "noul":
                z[0] += b
            m = max(z)
            probs = [math.exp(x - m) for x in z]
            if q.type == "noul":
                answers[qid] = noul_answer(probs[0], probs[1])
            elif q.type == "choice":
                answers[qid] = choice_answer(options, probs)
            else:
                answers[qid] = score_answer(options, probs)
        return SystemOneResponse(model=self.name, answers=answers, usage=Usage(input_tokens=tokens, output_tokens=0),
                                 latency_ms=(time.perf_counter() - t0) * 1000)
