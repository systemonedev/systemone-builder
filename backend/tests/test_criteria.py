"""Criteria on generated noul questions, and the guarantee that they discriminate.

Through v0.6 no noul question in our training data carried criteria, so the model never learned to
read the field (see :mod:`systemone_builder.kenning.criteria`). These tests cover the generators now
attaching them, and guard the rendering bug that made criteria useless: both hypotheses used to
receive the whole criteria blob, leaving them identical apart from "yes."/"no.".
"""

from __future__ import annotations

import random

from systemone_builder.kenning import criteria
from systemone_builder.kenning.model import hypotheses
from systemone_builder.kenning.synthetic_tasks import TASKS, questions_for
from systemone_builder.system_one.contract import Question


def test_noul_attaches_both_sides_when_present():
    q = criteria.noul(random.Random(0), "Is it urgent?", "A deadline is named.", "No time pressure.",
                      present=1.0)
    assert q["type"] == "noul"
    assert q["criteria"] == {"true": "A deadline is named.", "false": "No time pressure."}


def test_noul_omits_criteria_when_not_present_or_incomplete():
    rng = random.Random(0)
    assert "criteria" not in criteria.noul(rng, "Is it urgent?", "yes side", "no side", present=0.0)
    # one side missing: a half-filled dict would be worse than none
    assert "criteria" not in criteria.noul(rng, "Is it urgent?", "yes side", None, present=1.0)
    assert "criteria" not in criteria.noul(rng, "Is it urgent?", None, "no side", present=1.0)


def test_about_label_describes_the_label_on_the_true_side():
    q = criteria.about_label(random.Random(0), "Is this about billing?", "billing",
                             "charges, refunds, invoices", others=["billing", "technical", "sales"],
                             present=1.0)
    assert "billing" in q["criteria"]["true"]
    assert "charges, refunds, invoices" in q["criteria"]["true"]
    assert q["criteria"]["false"].startswith("It is not billing")
    # the label itself is never offered as one of the alternatives
    assert "technical" in q["criteria"]["false"]


def test_criteria_bearing_noul_renders_two_different_hypotheses():
    """The regression guard: the pair must differ by more than the answer token."""
    q = Question.model_validate(criteria.noul(
        random.Random(0), "Did the customer report an unauthorised charge?",
        "Someone else made the charge.", "The customer recognises the charge.", present=1.0))
    (yes, no), options = hypotheses(q)
    assert options == ["yes", "no"]
    assert yes != no
    assert yes.replace("Answer: yes.", "") != no.replace("Answer: no.", "")
    # each side carries only its own description
    assert "Someone else made the charge." in yes and "Someone else made the charge." not in no
    assert "The customer recognises the charge." in no and "The customer recognises the charge." not in yes


def test_noul_without_criteria_renders_as_before():
    """All of our pre-v0.7 training data has no criteria; that rendering must not move."""
    q = Question.model_validate({"type": "noul", "instructions": "Is this email phishing?"})
    assert hypotheses(q)[0] == ["Is this email phishing? Answer: yes.",
                                "Is this email phishing? Answer: no."]


def test_generated_tasks_mix_criteria_presence():
    """Some noul questions carry criteria and some do not, so the model sees both shapes."""
    with_crit = without = 0
    for seed in range(60):
        rng = random.Random(seed)
        for task in TASKS:
            labels = {a.name: rng.choice(list(a.labels)) for a in task.attrs}
            out = questions_for(task.attrs, labels, rng)
            for q in out["questions"].values():
                if q["type"] != "noul":
                    continue
                Question.model_validate(q)  # always a valid question either way
                if q.get("criteria"):
                    with_crit += 1
                    assert set(q["criteria"]) == {"true", "false"}
                    assert all(q["criteria"].values())
                else:
                    without += 1
    assert with_crit > 0, "no generated noul question carried criteria"
    assert without > 0, "every generated noul question carried criteria; the model would come to depend on it"


def test_generated_criteria_always_discriminate():
    """Whatever the generators produce, the two hypotheses must not collapse to the same string."""
    for seed in range(40):
        rng = random.Random(seed)
        for task in TASKS:
            labels = {a.name: rng.choice(list(a.labels)) for a in task.attrs}
            for q in questions_for(task.attrs, labels, rng)["questions"].values():
                if q["type"] != "noul" or not q.get("criteria"):
                    continue
                yes, no = hypotheses(Question.model_validate(q))[0]
                assert yes.replace("Answer: yes.", "") != no.replace("Answer: no.", "")
