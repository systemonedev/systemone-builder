"""WorkflowEvals suite label logic and workflow aggregation (no network)."""

from __future__ import annotations

from systemone_builder.system_one.workflowevals_suite import _target, workflow_averages


def _cons(pairs):
    return {"status": "answered", "probabilities": [{"option": o, "probability": p} for o, p in pairs]}


def test_target_maps_each_kind_from_the_consensus_argmax():
    # noul -> 1 if P(true) >= P(false), else 0
    assert _target("noul", _cons([("true", 0.98), ("false", 0.02)])) == 1
    assert _target("noul", _cons([("true", 0.4), ("false", 0.6)])) == 0
    # choice -> the winning option string
    assert _target("choice", _cons([("a", 0.1), ("b", 0.7), ("c", 0.2)])) == "b"
    # score -> the winning level as an int
    assert _target("score", _cons([("0", 0.1), ("2", 0.6), ("1", 0.3)])) == 2
    # no distribution -> no label
    assert _target("noul", {"status": "answered", "probabilities": []}) is None


def test_workflow_averages_groups_by_workflow_prefix():
    report = {"questions": {
        "invoice::a": {"n": 1, "accuracy": 1.0},
        "invoice::b": {"n": 1, "accuracy": 0.0},
        "security::c": {"n": 1, "exact_level": 1.0},    # score question uses exact_level
        "security::d": {"n": 0},                         # unanswered -> ignored
    }}
    assert workflow_averages(report) == {"invoice": 0.5, "security": 1.0}
