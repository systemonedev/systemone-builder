"""Criteria on generated noul questions, and the guarantee that they discriminate.

Through v0.6 no noul question in our training data carried criteria, so the model never learned to
read the field (see :mod:`systemone_builder.kenning.criteria`). These tests cover the generators now
attaching them, and guard the rendering bug that made criteria useless: both hypotheses used to
receive the whole criteria blob, leaving them identical apart from "yes."/"no.".
"""

from __future__ import annotations

import json
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
            out = questions_for(task.attrs, labels, rng, task.name)
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
            for q in questions_for(task.attrs, labels, rng, task.name)["questions"].values():
                if q["type"] != "noul" or not q.get("criteria"):
                    continue
                yes, no = hypotheses(Question.model_validate(q))[0]
                assert yes.replace("Answer: yes.", "") != no.replace("Answer: no.", "")


def test_teacher_variants_are_used_when_present(tmp_path, monkeypatch):
    """A template with teacher-written variants uses one instead of the terse description."""
    f = tmp_path / "criteria_variants.json"
    f.write_text(json.dumps({"version": 1, "criteria": {"demo.attr": {"kind": "yes_no", "variants": [
        {"true": "A deadline, travel date or payment due date is named.",
         "false": "Only general impatience, with no date or deadline given."}]}}}), encoding="utf-8")
    monkeypatch.setattr(criteria, "VARIANTS_FILE", f)
    criteria._variants.cache_clear()
    try:
        q = criteria.noul(random.Random(0), "Is it urgent?", "terse yes", "terse no",
                          present=1.0, key="demo.attr")
        assert q["criteria"]["true"].startswith("A deadline, travel date")
        assert "terse" not in json.dumps(q)
        # an unknown key still falls back to the generator's own descriptions
        other = criteria.noul(random.Random(0), "Is it urgent?", "terse yes", "terse no",
                              present=1.0, key="nope.missing")
        assert other["criteria"] == {"true": "Terse yes.", "false": "Terse no."}
    finally:
        criteria._variants.cache_clear()


def test_missing_variants_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(criteria, "VARIANTS_FILE", tmp_path / "absent.json")
    criteria._variants.cache_clear()
    try:
        assert criteria.variants_for("anything") == []
        q = criteria.noul(random.Random(0), "Is it urgent?", "yes side", "no side", present=1.0)
        assert q["criteria"] == {"true": "Yes side.", "false": "No side."}
    finally:
        criteria._variants.cache_clear()


def test_phishing_asks_the_same_decision_both_ways_and_swaps_the_criteria(tmp_path, monkeypatch):
    """q_threat and q_safe are the same decision inverted, so the two sides must swap."""
    from systemone_builder.kenning import data as phishing_data

    # isolate from the shipped teacher variants: this is about the polarity logic
    monkeypatch.setattr(criteria, "VARIANTS_FILE", tmp_path / "absent.json")
    criteria._variants.cache_clear()
    try:
        threat = safe = 0
        for seed in range(80):
            qs, targets = phishing_data.questions_for(random.Random(seed), malicious=True)
            if "q_threat" in qs and qs["q_threat"].get("criteria"):
                c = qs["q_threat"]["criteria"]
                assert c["true"] == phishing_data.THREAT_CRITERION
                assert c["false"] == phishing_data.SAFE_CRITERION
                threat += 1
            if "q_safe" in qs and qs["q_safe"].get("criteria"):
                c = qs["q_safe"]["criteria"]
                assert c["true"] == phishing_data.SAFE_CRITERION
                assert c["false"] == phishing_data.THREAT_CRITERION
                safe += 1
        assert threat and safe, "both phrasings should appear across 80 seeds"
    finally:
        criteria._variants.cache_clear()


def test_safe_criterion_names_the_hard_false_positives():
    """Urgent, automated, sign-in-requesting mail is the FP class that matters; say so explicitly."""
    from systemone_builder.kenning.data import SAFE_CRITERION

    low = SAFE_CRITERION.lower()
    assert "urgent" in low and "automated" in low and "sign in" in low


def test_problem_rows_carry_criteria():
    """problems.py reuses synthetic_tasks.questions_for, keyed by problem name."""
    from systemone_builder.kenning.problems import builtin_problems, training_rows

    seen_noul = seen_criteria = 0
    for name, p in builtin_problems().items():
        cases = [{"state": {"x": "..."}, "labels": {a.name: list(a.labels)[0] for a in p.attrs}}
                 for _ in range(12)]
        for row in training_rows(p, cases, seed=3):
            for q in row["questions"].values():
                if q["type"] != "noul":
                    continue
                Question.model_validate(q)
                seen_noul += 1
                if q.get("criteria"):
                    seen_criteria += 1
                    assert set(q["criteria"]) == {"true", "false"}
    assert seen_noul, "no problem produced a noul question"
    assert seen_criteria, "no problem noul question carried criteria"


def test_every_generator_template_key_has_teacher_variants():
    """A key typo would silently fall back to the terse text, so pin the coverage."""
    from systemone_builder.kenning.enrich_criteria import templates

    criteria._variants.cache_clear()
    missing = sorted(t["key"] for t in templates() if not criteria.variants_for(t["key"]))
    assert not missing, f"templates with no teacher variants: {missing[:10]} ({len(missing)} total)"
