"""Run an exported Kenning bundle in-process (``pip install "systemone-client[local]"``).

    from systemone import Kenning, Noul

    model = Kenning.from_pretrained("./kenning-large-v0.1")   # the unzipped export bundle
    r = model.system_one(state={"ticket": "..."}, questions={"billing": Noul("Is this about billing?")})

Same answers as the Kenning server for the same request: every candidate answer
is scored against the state in one batch, and each question's distribution is
``softmax(scores / T)`` with the calibration temperatures stored in the bundle.
This file must stay in step with SystemOne Builder's ``systemone_builder.kenning.model``
(a parity test there checks the templates).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from systemone.types import Question, Response, question_dict

CONFIG_FILES = ("kenning.json", "s1_config.json")


def describe(desc: Any) -> str:
    if desc is None:
        return ""
    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        parts = []
        for k, v in desc.items():
            v = "; ".join(map(str, v)) if isinstance(v, list) else str(v)
            parts.append(v if k == "what" else f"{k.replace('_', ' ')}: {v}")
        return " | ".join(parts)
    return str(desc)


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False, default=str)


def hypotheses(q: dict[str, Any]) -> tuple[list[str], list[Any]]:
    stem = q["instructions"].strip()
    crit = q.get("criteria")
    if q["type"] == "noul":
        c = describe(crit)
        suffix = f" ({c})" if c else ""
        return [f"{stem}{suffix} Answer: yes.", f"{stem}{suffix} Answer: no."], ["yes", "no"]
    if q["type"] == "choice":
        texts = []
        for o, d in crit.items():
            dd = describe(d)
            texts.append(f"{stem} Answer: {o}." + (f" {dd}" if dd else ""))
        return texts, list(crit)
    return [f"{stem} Answer: {describe(lv)}." for lv in crit], list(crit)


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


def spread_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    return round(min(max((max(probs) - 1 / n) / (1 - 1 / n), 0.0), 1.0), 6)


def _normalize(p: list[float]) -> list[float]:
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1 / len(p)] * len(p)


class Kenning:
    """A Kenning model loaded in this process. Thread-safe; one request runs at a time."""

    def __init__(self, path: str | os.PathLike[str], device: str | None = None, max_length: int | None = None,
                 chunk: int = 32) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError('local inference needs: pip install "systemone-client[local]"') from exc
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        self._torch = torch
        self.path = str(path)
        local = local_dir(self.path)
        self.config: dict[str, Any] = {}
        for name in CONFIG_FILES:
            f = Path(local) / name
            if f.exists():
                self.config = json.loads(f.read_text())
                break
        self.temperature = {"noul": 1.0, "choice": 1.0, "score": 1.0, **self.config.get("temperature", {})}
        self.name = self.config.get("name") or Path(self.path).name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.model = AutoModelForSequenceClassification.from_pretrained(local, dtype=dtype).to(self.device).eval()
        label2id = {k.lower(): v for k, v in (getattr(self.model.config, "label2id", None) or {}).items()}
        self.positive_index = int(self.config.get("positive_index", label2id.get("entailment", 0)))
        self.max_length = max_length or int(self.config.get("max_length", 2048))
        self.chunk = chunk
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike[str], **kw: Any) -> "Kenning":
        """A model directory or a Hugging Face repo id, e.g. "systemonedev/kenning-large-v0.4"."""
        return cls(path, **kw)

    def _scores(self, premise: str, hyps: list[str]) -> tuple[list[float], int]:
        torch = self._torch
        out: list[float] = []
        tokens = 0
        with torch.inference_mode():
            for i in range(0, len(hyps), self.chunk):
                part = hyps[i:i + self.chunk]
                enc = self.tokenizer([premise] * len(part), part, truncation="only_first", max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                tokens += int(enc["attention_mask"].sum())
                logits = self.model(**enc).logits.float()
                if logits.shape[-1] == 1:
                    s = logits[:, 0]
                else:
                    pos = self.positive_index
                    others = torch.cat([logits[:, :pos], logits[:, pos + 1:]], dim=-1)
                    s = logits[:, pos] - torch.logsumexp(others, dim=-1)
                out.extend(s.tolist())
        return out, tokens

    def system_one(self, state: Any, questions: Mapping[str, Question | dict[str, Any]]) -> Response:
        t0 = time.perf_counter()
        premise = render_state(state)
        groups, hyps = [], []
        for qid, q in questions.items():
            qd = question_dict(q)
            texts, options = hypotheses(qd)
            groups.append((qid, qd["type"], options, len(hyps), len(texts)))
            hyps.extend(texts)
        with self._lock:
            flat, tokens = self._scores(premise, hyps)
        answers: dict[str, Any] = {}
        for qid, kind, options, start, n in groups:
            t = self.temperature.get(kind, 1.0)
            z = [s / t for s in flat[start:start + n]]
            m = max(z)
            p = _normalize([math.exp(x - m) for x in z])
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": round(p[0], 6)}
            elif kind == "choice":
                best = max(range(len(p)), key=p.__getitem__)
                answers[qid] = {"type": "choice", "choice": options[best], "confidence": spread_confidence(p),
                                "probabilities": {o: round(x, 6) for o, x in zip(options, p)}}
            else:
                answers[qid] = {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 6),
                                "confidence": spread_confidence(p),
                                "legend": {str(i): describe(lv) for i, lv in enumerate(options)},
                                "probabilities": {str(i): round(x, 6) for i, x in enumerate(p)}}
        return Response.from_wire({"model": self.name, "answers": answers,
                                   "usage": {"input_tokens": tokens, "output_tokens": 0},
                                   "latency_ms": (time.perf_counter() - t0) * 1000})
