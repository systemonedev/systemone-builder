"""System One model: templates, targets, calibration and training data (no torch needed)."""

from __future__ import annotations

import json
import math
import random

import pytest

from systemone.s1.contract import Question
from systemone.s1.factory import api_engine, build_engine
from systemone.s1model.data import SAFE_OPTION, body_hash, excluded_hashes, questions_for
from systemone.s1model.model import hypotheses
from systemone.s1model.train import fit_temperatures, metrics, target_vector


def q(**kw):
    return Question.model_validate(kw)


def test_hypotheses_cover_every_answer_in_order():
    texts, opts = hypotheses(q(type="noul", instructions="Is it bad?", criteria="phishing counts"))
    assert opts == ["yes", "no"] and texts[0] == "Is it bad? (phishing counts) Answer: yes."
    texts, opts = hypotheses(q(type="choice", instructions="Kind?", criteria={"Safe": "fine", "Phish": None}))
    assert opts == ["Safe", "Phish"] and texts == ["Kind? Answer: Safe. fine", "Kind? Answer: Phish."]
    texts, opts = hypotheses(q(type="score", instructions="Severity?", criteria=["Low", {"what": "High"}]))
    assert texts == ["Severity? Answer: Low.", "Severity? Answer: High."]


def test_target_vectors_hard_and_soft():
    noul = q(type="noul", instructions="x")
    assert target_vector(noul, ["yes", "no"], True) == [1.0, 0.0]
    assert target_vector(noul, ["yes", "no"], 0.25) == [0.25, 0.75]
    choice = q(type="choice", instructions="x", criteria={"a": None, "b": None, "c": None})
    assert target_vector(choice, ["a", "b", "c"], "b") == [0.0, 1.0, 0.0]
    assert target_vector(choice, ["a", "b", "c"], {"a": 2, "c": 2}) == [0.5, 0.0, 0.5]
    score = q(type="score", instructions="x", criteria=["l", "m", "h"])
    assert target_vector(score, ["l", "m", "h"], 2) == [0.0, 0.0, 1.0]
    assert target_vector(score, ["l", "m", "h"], {"0": 1, "1": 3}) == [0.25, 0.75, 0.0]
    with pytest.raises(ValueError):
        target_vector(choice, ["a", "b", "c"], "zzz")


def test_temperature_fit_fixes_overconfidence():
    # scores say 99.99% but the model is right only 70% of the time -> T must grow
    rng = random.Random(0)
    raw = []
    for _ in range(400):
        right = rng.random() < 0.7
        raw.append(("noul", [10.0, 0.0], [1.0, 0.0] if right else [0.0, 1.0]))
    t = fit_temperatures(raw)["noul"]
    assert t > 5
    calibrated = metrics(raw, {"noul": t})["noul"]
    assert calibrated["ece"] < metrics(raw, {})["noul"]["ece"]
    p_yes = 1 / (1 + math.exp(-10 / t))
    assert p_yes == pytest.approx(0.7, abs=0.05)


def test_training_data_excludes_benchmark_and_varies_questions(tmp_path):
    bench = [{"id": "x", "state": {"email": {"body": "Win a prize now"}}, "labels": {}}]
    (tmp_path / "phishing-n50-seed42.json").write_text(json.dumps(bench))
    assert excluded_hashes(tmp_path) == {body_hash("Win a prize now")}
    rng = random.Random(1)
    inverted = 0
    for _ in range(200):
        qs, targets = questions_for(rng, malicious=True)
        assert targets["q_kind"] not in SAFE_OPTION  # malicious email -> the unsafe option
        if "q_safe" in qs:
            inverted += 1
            assert targets["q_safe"] == 0  # "is it safe?" about a phishing email -> no
        else:
            assert targets["q_threat"] == 1
    assert 20 < inverted < 100


def test_engine_factory_targets_local_model_server():
    from systemone.config import Settings

    s = Settings(_env_file=None)
    e = build_engine("s1", s)
    assert str(e.client.base_url) == "http://s1:8000" and e._require_key is False
    assert api_engine(s).name.startswith("s1")
    assert api_engine(Settings(_env_file=None, system_one_backend="logprob")).name.startswith("local")
