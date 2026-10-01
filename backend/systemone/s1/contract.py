"""The System One wire contract, field-for-field compatible with TypeSafe Jev.

Request (``POST /v1/systemone``)::

    {"state": "<text>" | {...},
     "model": "jev-latest",
     "questions": {
       "urgent":   {"type": "noul",   "instructions": "...", "criteria": "..."},
       "team":     {"type": "choice", "instructions": "...",
                    "criteria": {"returns": "Exchanges ...", "billing": null}},
       "severity": {"type": "score",  "instructions": "...",
                    "criteria": ["Cosmetic", "Degraded", "Blocking"]}}}

Response::

    {"model": "...",
     "answers": {
       "urgent":   {"type": "noul", "noul": 0.97},
       "team":     {"type": "choice", "choice": "returns", "confidence": 1.0,
                    "probabilities": {"returns": 1.0, "billing": 0.0}},
       "severity": {"type": "score", "score": 1.43, "confidence": 0.35,
                    "legend": {"0": "Cosmetic", ...},
                    "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43}}},
     "usage": {"input_tokens": 392, "output_tokens": 65}}

Semantics (TypeSafe docs):

* ``noul`` is the probability that the answer is yes.
* ``score`` is the probability-weighted mean of the level indices (0..n-1).
* ``confidence`` measures how concentrated ``probabilities`` is. Jev's
  published examples match ``(max(p) - 1/n) / (1 - 1/n)`` (0.85 over three
  options -> 0.78; 0.57 over three levels -> 0.35), so local engines use the
  same formula.

``latency_ms`` on the response is a local extension; Jev does not send it.
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
