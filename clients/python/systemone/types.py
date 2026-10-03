"""Questions and answers of the System One wire format.

Questions (what you ask about the state):

* ``Noul(instructions, criteria=None)``: yes/no. The answer is ``noul``, the
  probability of "yes". It is a float: test ``answer.noul >= 0.9``, never
  ``bool(answer.noul)``.
* ``Choice(instructions, criteria={option: description | None, ...})``: one of
  2-255 options. The answer has ``choice``, ``probabilities`` and ``confidence``.
* ``Score(instructions, criteria=[level, ...])``: an ordered scale of 2-10
  levels. ``score`` is the probability-weighted level index (0..n-1); check
  ``probabilities`` too, since a mean of 1.0 can hide "either 0 or 2".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: str | dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
            raise ValueError("Choice criteria must map 2-255 option names to a description (or None)")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, (list, tuple)) or not 2 <= len(self.criteria) <= 10:
            raise ValueError("Score criteria must be an ordered list of 2-10 levels")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Union[Noul, Choice, Score]


def question_dict(q: Question | dict[str, Any]) -> dict[str, Any]:
    return q if isinstance(q, dict) else q.to_dict()


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, str] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    model: str
    nouls: dict[str, NoulAnswer] = field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "Response":
        r = cls(model=data.get("model", ""), raw=data, latency_ms=data.get("latency_ms"),
                usage=Usage(**{k: int(v) for k, v in (data.get("usage") or {}).items()
                               if k in ("input_tokens", "output_tokens")}))
        for qid, a in (data.get("answers") or {}).items():
            t = a.get("type")
            if t == "noul":
                r.nouls[qid] = NoulAnswer(float(a["noul"]))
            elif t == "choice":
                r.choices[qid] = ChoiceAnswer(a["choice"], float(a["confidence"]), dict(a["probabilities"]))
            elif t == "score":
                r.scores[qid] = ScoreAnswer(float(a["score"]), float(a["confidence"]), dict(a["probabilities"]),
                                            dict(a.get("legend") or {}))
        return r
