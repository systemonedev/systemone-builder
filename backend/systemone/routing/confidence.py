"""Dynamic confidence scoring for Fast-Slow routing (Module D).

The student emits a self-reported ``confidence_score`` with every action.
Self-reports alone are poorly calibrated, so the router blends them with the
model's *token-level* certainty about the decision itself:

``p_tokens`` is the joint probability of the tokens that spell the decision
fields (the action name, target id, verdict, IOCs ...). It is computed from
the streamed logprobs by locating each field's value span inside the output
text and summing the logprobs of the overlapping tokens. The weakest field
wins (``min``), because one uncertain field makes the whole action uncertain.

``confidence = self_reported^(1-w) * p_tokens^w`` (geometric blend), then an
optional Platt calibration ``sigmoid(a * logit(c) + b)`` fitted on held-out
evaluation data (Module G). Hard gates force the score to 0: unparseable or
schema-invalid output, ungrounded references (hallucinations), or an explicit
``ESCALATE`` action.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from systemone.adapters.base import TokenLogprob
from systemone.domains.spec import DomainSpec, ValidationResult

DECISION_FIELDS = ("action", "immediate_action", "verdict", "target_id", "target_ioc", "coordinates", "text")
ESCALATE = "ESCALATE"


@dataclass
class Calibration:
    a: float = 1.0
    b: float = 0.0

    def apply(self, p: float) -> float:
        if self.a == 1.0 and self.b == 0.0:
            return p
        p = min(max(p, 1e-6), 1 - 1e-6)
        z = self.a * math.log(p / (1 - p)) + self.b
        return 1 / (1 + math.exp(-z))

    @classmethod
    def fit(cls, scores: list[float], correct: list[bool], iters: int = 500, lr: float = 0.1) -> "Calibration":
        """Platt scaling by gradient descent on log-loss."""
        if len(scores) < 10 or len(set(correct)) < 2:
            return cls()
        xs = [math.log(min(max(s, 1e-6), 1 - 1e-6) / (1 - min(max(s, 1e-6), 1 - 1e-6))) for s in scores]
        ys = [1.0 if c else 0.0 for c in correct]
        a, b, n = 1.0, 0.0, len(xs)
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = 1 / (1 + math.exp(-(a * x + b)))
                ga += (p - y) * x
                gb += p - y
            a -= lr * ga / n
            b -= lr * gb / n
        return cls(a=a, b=b)


@dataclass
class ConfidenceReport:
    confidence: float
    self_reported: float | None
    token_confidence: float | None
    field_confidence: dict[str, float] = field(default_factory=dict)
    gates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _value_spans(text: str, fields: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Character span of each top-level field's JSON value inside ``text``."""
    spans: dict[str, tuple[int, int]] = {}
    decoder = json.JSONDecoder()
    for f in fields:
        m = re.search(r'"%s"\s*:\s*' % re.escape(f), text)
        if not m:
            continue
        start = m.end()
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if text[start:end] != "null":
            spans[f] = (start, end)
    return spans


def token_field_confidence(text: str, logprobs: list[TokenLogprob], fields: tuple[str, ...] = DECISION_FIELDS) -> dict[str, float]:
    if not logprobs:
        return {}
    offsets, pos = [], 0
    for tl in logprobs:
        offsets.append((pos, pos + len(tl.token)))
        pos += len(tl.token)
    # The token stream should reproduce the text; if not (e.g. special tokens),
    # align by searching for the first JSON brace.
    joined = "".join(t.token for t in logprobs)
    shift = 0
    if joined != text:
        i, j = joined.find("{"), text.find("{")
        shift = (j - i) if i >= 0 and j >= 0 else 0
    out: dict[str, float] = {}
    for f, (a, b) in _value_spans(text, fields).items():
        total = 0.0
        hit = False
        for (s, e), tl in zip(offsets, logprobs):
            s, e = s + shift, e + shift
            if e > a and s < b:
                total += tl.logprob
                hit = True
        if hit:
            out[f] = math.exp(total)
    return out


class ConfidenceScorer:
    def __init__(self, logprob_weight: float = 0.5, calibrations: dict[str, Calibration] | None = None) -> None:
        self.w = logprob_weight
        self.calibrations = calibrations or {}

    def score(
        self,
        domain: DomainSpec,
        text: str,
        action: dict[str, Any] | None,
        validation: ValidationResult | None,
        logprobs: list[TokenLogprob] | None,
    ) -> ConfidenceReport:
        gates: list[str] = []
        if action is None:
            return ConfidenceReport(0.0, None, None, gates=["unparseable_output"])
        raw_self = action.get("confidence_score")
        self_rep = float(raw_self) if isinstance(raw_self, (int, float)) else None
        if self_rep is not None:
            self_rep = min(max(self_rep, 0.0), 1.0)
        fields = tuple(f for f in DECISION_FIELDS if f in action)
        per_field = token_field_confidence(text, logprobs or [], fields)
        tok = min(per_field.values()) if per_field else None
        if tok is None and logprobs:
            tok = math.exp(sum(t.logprob for t in logprobs) / len(logprobs))

        if self_rep is not None and tok is not None:
            conf = (max(self_rep, 1e-9) ** (1 - self.w)) * (max(tok, 1e-9) ** self.w)
        elif self_rep is not None:
            conf = self_rep
        elif tok is not None:
            conf = tok
        else:
            conf = 0.0
            gates.append("no_confidence_signal")
        conf = self.calibrations.get(domain.id, Calibration()).apply(conf)

        if validation is not None and not validation.ok:
            gates.append("schema_invalid")
        if validation is not None and validation.hallucinated:
            gates.append("ungrounded_reference")
        if domain.action_label(action) == ESCALATE:
            gates.append("explicit_escalate")
        if gates:
            conf = 0.0
        return ConfidenceReport(round(conf, 4), self_rep, None if tok is None else round(tok, 4), per_field, gates)
