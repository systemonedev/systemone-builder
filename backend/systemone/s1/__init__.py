"""System One: typed, calibrated decisions instead of generated text.

The wire contract (:mod:`systemone.s1.contract`) is the one TypeSafe's Jev
exposes at ``POST /v1/systemone``: program ``state`` plus typed ``questions``
(``noul``, ``choice``, ``score``) in, typed ``answers`` with probabilities out.
Every engine speaks it, so application code can target TypeSafe Jev or a
locally run System One model interchangeably, and the benchmark harness
(:mod:`systemone.s1.bench`) can compare them question for question.
"""

from systemone.s1.contract import (
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

__all__ = ["ChoiceAnswer", "NoulAnswer", "Question", "ScoreAnswer", "SystemOneRequest", "SystemOneResponse"]
