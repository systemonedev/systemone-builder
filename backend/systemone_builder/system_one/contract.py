"""The System One wire contract.

The format is compatible with TypeSafe AI's System One API (``POST
/v1/systemone``), so clients written for either work with both. This project is
not affiliated with TypeSafe AI.

Request::

    {"state": "<text>" | {...},
     "model": "kenning",                      # optional
     "questions": {
       "refund": {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
       "queue":  {"type": "choice", "instructions": "Which queue?",
                  "criteria": {"billing": "Charges, refunds", "account": null}},
       "tone":   {"type": "score",  "instructions": "How heated is the message?",
                  "criteria": ["Polite", "Impatient", "Hostile"]}}}

Response::

    {"model": "kenning-large-v0.1",
     "answers": {
       "refund": {"type": "noul", "noul": 0.96},
       "queue":  {"type": "choice", "choice": "billing", "confidence": 0.9,
                  "probabilities": {"billing": 0.95, "account": 0.05}},
       "tone":   {"type": "score", "score": 1.2, "confidence": 0.31,
                  "legend": {"0": "Polite", "1": "Impatient", "2": "Hostile"},
                  "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}},
     "usage": {"input_tokens": 211, "output_tokens": 0}}

Semantics:

* ``noul`` is the probability that the answer is yes.
* ``score`` is the probability-weighted mean of the level indices (0..n-1).
* ``confidence`` measures how concentrated ``probabilities`` is:
  ``(max(p) - 1/n) / (1 - 1/n)``, which reproduces the values in the published
  System One examples (0.85 over three options -> 0.78; 0.57 over three
  levels -> 0.35).

``latency_ms`` on the response is a local extension.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1)
    # noul: optional str/object; choice: {option: description|null|object};
    # score: ordered [level description|object, ...]
    criteria: dict[str, Any] | list[Any] | str | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> "Question":
        c = self.criteria
        if self.type == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice criteria must map 2-{MAX_CHOICE_OPTIONS} option names to descriptions (or null)")
        elif self.type == "score":
            if not isinstance(c, list) or not MIN_SCORE_LEVELS <= len(c) <= MAX_SCORE_LEVELS:
                raise ValueError(f"score criteria must be an ordered list of {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels")
        elif c is not None and not isinstance(c, (str, dict)):
            raise ValueError("noul criteria must be a string or an object")
        return self


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any]
    model: str | None = None
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float]


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None  # local extension

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}


# ------------------------------------------------------------------ helpers
def spread_confidence(probs: Sequence[float]) -> float:
    """1.0 when all mass is on one outcome, 0.0 when it is spread evenly."""
    n = len(probs)
    if n < 2:
        return 1.0
    c = (max(probs) - 1 / n) / (1 - 1 / n)
    return round(min(max(c, 0.0), 1.0), 6)


def describe(desc: Any) -> str:
    """Flatten a criteria description (None, text, or {what, not_for, examples, ...})."""
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


def render_state(state: str | dict[str, Any]) -> str:
    return state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False, default=str)


def _normalize(probs: Sequence[float]) -> list[float]:
    total = sum(probs)
    if total <= 0:
        return [1 / len(probs)] * len(probs)
    return [p / total for p in probs]


def noul_answer(p_yes: float, p_no: float) -> NoulAnswer:
    yes, _ = _normalize([p_yes, p_no])
    return NoulAnswer(noul=round(yes, 6))


def choice_answer(options: Sequence[str], probs: Sequence[float]) -> ChoiceAnswer:
    p = _normalize(probs)
    best = max(range(len(p)), key=p.__getitem__)  # first maximum: deterministic tie-break
    return ChoiceAnswer(choice=options[best], confidence=spread_confidence(p),
                        probabilities={o: round(x, 6) for o, x in zip(options, p)})


def score_answer(levels: Sequence[Any], probs: Sequence[float]) -> ScoreAnswer:
    p = _normalize(probs)
    return ScoreAnswer(
        score=round(sum(i * x for i, x in enumerate(p)), 6),
        confidence=spread_confidence(p),
        legend={str(i): describe(lv) for i, lv in enumerate(levels)},
        probabilities={str(i): round(x, 6) for i, x in enumerate(p)},
    )
