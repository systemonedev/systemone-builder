"""Criteria for the noul questions in training data.

A noul question may carry ``criteria`` as a ``{"true": ..., "false": ...}`` dict describing the two
answers. :func:`systemone_builder.kenning.model.hypotheses` gives each hypothesis only its own side,
so that dict is where a question's discriminating detail lives: "did the customer report a charge
they did not authorise?" is decided by the *false* case ("recognises the activity, disputes only the
amount") far more than by the stem.

Engines speaking this wire format send questions shaped that way. Every noul question in our own
training data through v0.6 carried no criteria at all, so the model never learned to read the field.
Measured on TypeSafe WorkflowEvals, its yes/no accuracy sat at chance (0.528, and 0.479 whether the
state fit the context window or not) while a hosted engine reached 0.894 on the same items.

Presence is randomised here, exactly as option descriptions already are for choice questions in
``synthetic_tasks.questions_for``. Training on criteria-bearing questions *only* would teach the
model to depend on a field our own benchmark suites never send, so each question carries them about
``PRESENT`` of the time and is otherwise unchanged.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

# How often a noul question that *could* carry criteria actually does.
PRESENT = 0.6


def _sentence(text: str) -> str:
    """A label description reads as a sentence once it follows "Answer: yes.".

    The generators store fragments ("no time pressure"), while questions from other engines carry
    full sentences. Capitalising and closing them keeps our training text in the same shape.
    """
    t = " ".join(str(text).split())
    if not t:
        return t
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


def noul(rng: random.Random, instructions: str, when_true: str | None = None,
         when_false: str | None = None, present: float = PRESENT) -> dict[str, Any]:
    """A noul question, carrying ``{"true", "false"}`` criteria some of the time.

    Falls back to a bare question when either side is missing, so a caller with only one usable
    description still produces valid data rather than a half-filled dict.
    """
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if when_true and when_false and rng.random() < present:
        q["criteria"] = {"true": _sentence(when_true), "false": _sentence(when_false)}
    return q


def about_label(rng: random.Random, instructions: str, label: str, description: str | None = None,
                others: Iterable[str] = (), present: float = PRESENT) -> dict[str, Any]:
    """Criteria for the "is the answer <label>?" shape the generators use for balanced negatives.

    The whole decision is what ``label`` means, which is precisely what the stem leaves implicit, so
    this is the shape where criteria carry the most information.
    """
    rest = [o for o in others if o != label]
    when_true = f"It is {label}" + (f": {description}" if description else "")
    when_false = "It is not " + label
    if rest:
        shown = list(rest)[:4]
        when_false += ", but one of the other options (" + ", ".join(shown) + ")"
    return noul(rng, instructions, when_true, when_false, present)
