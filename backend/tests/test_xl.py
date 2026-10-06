"""Kenning-XL logic without a GPU: gold traces, readout prompts, target mapping, server escalation."""

from __future__ import annotations

import random
from types import SimpleNamespace

from systemone_builder.kenning import traces, xl, xl_serve
from systemone_builder.kenning.train_xl import _gold_surface, target_probs
from systemone_builder.system_one.contract import Question


# ------------------------------------------------------------------ traces
def test_every_record_family_emits_a_consistent_trace():
    from systemone_builder.kenning import structured as st
    for kind in traces.RECORD_FAMILIES:
        for state, _attrs, labels in st.cases(kind, 40, 2):
            t = traces.trace_for(kind, state, labels)
            assert t and len(t) > 15 and ("->" in t or "<" in t or ">" in t), (kind, t)


def test_deliberate_rows_carry_trace_and_valid_questions():
    rows = traces.deliberate_rows(10, 1)
    assert rows and all("trace" in r and r["trace"] for r in rows)
    for r in rows[:20]:
        for q in r["questions"].values():
            Question.model_validate(q)
        assert set(r["targets"]) == set(r["questions"])


# ------------------------------------------------------------------ prompts / targets
def test_prompt_and_cue_cover_every_answer_piece():
    noul = Question(type="noul", instructions="Is it overdue?")
    body, pieces = xl._prompt("state here", noul, [])
    assert pieces == ["yes", "no"] and "Answer:" in body and "Answer:" in xl.answer_cue(noul, [])

    choice = Question(type="choice", instructions="Which team?", criteria={"billing": None, "eng": None, "sales": None})
    body, pieces = xl._prompt("s", choice, list(choice.criteria))
    assert pieces == ["A", "B", "C"] and "A. billing" in body
    assert "A. billing" in xl.answer_cue(choice, list(choice.criteria))

    score = Question(type="score", instructions="How urgent?", criteria=["low", "high"])
    body, pieces = xl._prompt("s", score, score.criteria)
    assert pieces == ["0", "1"]


def test_target_probs_and_gold_surface_align():
    noul = Question(type="noul", instructions="?")
    assert target_probs(noul, ["yes", "no"], 1) == [1.0, 0.0]
    assert _gold_surface(noul, ["yes", "no"], 0) == "no"
    ch = Question(type="choice", instructions="?", criteria={"a": None, "b": None})
    assert target_probs(ch, ["a", "b"], "b") == [0.0, 1.0] and _gold_surface(ch, ["a", "b"], "b") == "B"
    assert target_probs(ch, ["a", "b"], {"a": 0.25, "b": 0.75}) == [0.25, 0.75]
    sc = Question(type="score", instructions="?", criteria=["x", "y", "z"])
    assert target_probs(sc, ["0", "1", "2"], 2) == [0.0, 0.0, 1.0] and _gold_surface(sc, ["0", "1", "2"], 2) == "2"


# ------------------------------------------------------------------ server escalation
def test_server_escalates_only_unsure_questions(monkeypatch):
    calls = {"onepass": 0, "deliberate": []}

    class Stub:
        name = "stub-xl"
        max_length = 4096
        temperature = {"noul": 1.0}

        def system_one(self, req, deliberate=False):
            if deliberate:
                calls["deliberate"].append(sorted(req.questions))
                ans = {q: SimpleNamespace(type="noul", noul=0.99) for q in req.questions}
            else:
                calls["onepass"] += 1
                # "sure" is confident (0.95), "unsure" is near 0.5
                ans = {q: SimpleNamespace(type="noul", noul=(0.95 if q == "sure" else 0.52)) for q in req.questions}
            return SimpleNamespace(answers=ans)

    monkeypatch.setattr(xl_serve, "_model", Stub())
    monkeypatch.setenv("S1_XL_ESCALATE", "0.75")
    req = SimpleNamespace(state="s", questions={"sure": Question(type="noul", instructions="?"),
                                                "unsure": Question(type="noul", instructions="?")})
    resp = xl_serve.answer(req)
    assert calls["onepass"] == 1 and calls["deliberate"] == [["unsure"]]  # only the unsure one re-run
    assert resp.answers["sure"].noul == 0.95 and resp.answers["unsure"].noul == 0.99  # deliberate result kept

    # threshold 0 = never escalate
    calls["deliberate"].clear()
    monkeypatch.setenv("S1_XL_ESCALATE", "0")
    xl_serve.answer(req)
    assert calls["deliberate"] == []


def test_confidence_helper():
    assert xl_serve._confidence(SimpleNamespace(type="noul", noul=1.0)) == 1.0
    assert xl_serve._confidence(SimpleNamespace(type="noul", noul=0.5)) == 0.0
    assert xl_serve._confidence(SimpleNamespace(type="choice", confidence=0.8)) == 0.8
    _ = random  # keep import meaningful
