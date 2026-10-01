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


# ------------------------------------------------------------ multi-task data
def _ctx(seed=0):
    from systemone.s1model.multitask import Ctx

    return Ctx(random.Random(seed), {
        "label": ["negative", "positive"],
        "intent": ["oos", "transfer_money", "check_balance", "book_flight", "weather", "timer", "alarm", "translate",
                   "recipe", "calories", "pay_bill", "credit_score", "order_status", "lost_card", "pin_change", "spelling"],
    })


SAMPLES = {
    "amazon": {"label": 1, "title": "Great", "content": "Loved it, works perfectly."},
    "clinc": {"intent": 1, "text": "send 20 dollars to mom"},
    "boolq": {"question": "is the sky blue", "passage": "The sky appears blue due to Rayleigh scattering.", "answer": True},
    "nli": {"premise": "A man is playing a guitar on stage.", "hypothesis": "A person is performing music.", "label": 0},
    "civil": {"text": "You are an idiot.", "toxicity": 0.9, "severe_toxicity": 0.3, "insult": 0.85, "threat": 0.0, "obscene": 0.1},
}


@pytest.mark.parametrize("key", list(SAMPLES))
def test_multitask_rows_fit_the_contract_and_trainer(key):
    from systemone.s1model.multitask import SOURCES

    for seed in range(40):  # every random branch of the maker
        row = SOURCES[key].make(dict(SAMPLES[key]), _ctx(seed))
        if row is None:
            continue
        for qid, qd in row["questions"].items():
            qq = Question.model_validate(qd)
            _, options = hypotheses(qq)
            vec = target_vector(qq, options, row["targets"][qid])
            assert sum(vec) == pytest.approx(1.0)


def test_multitask_class_questions_are_balanced_and_genre_questions_mostly_no():
    from systemone.s1model.multitask import make_clinc, make_dbpedia

    from systemone.s1model.multitask import Ctx

    ctx = Ctx(random.Random(3), {"label": ["Company", "Artist", "Athlete", "Village", "Album", "Film", "Plant", "Animal",
                                           "Building", "NaturalPlace", "WrittenWork", "Politician", "School", "Vehicle"]})
    nouls, genre = [], []
    for _ in range(600):
        row = make_dbpedia({"label": 1, "title": "X", "content": "An artist."}, ctx)
        if row["questions"]["q_topic"]["type"] == "noul":
            nouls.append(row["targets"]["q_topic"])
        if "q_genre" in row["targets"]:
            genre.append(row["targets"]["q_genre"])
    assert 0.4 < sum(nouls) / len(nouls) < 0.6
    assert 0.2 < sum(genre) / len(genre) < 0.5  # most genre questions are about another genre
    other_bucket = 0
    for seed in range(200):
        row = make_clinc(dict(SAMPLES["clinc"]), _ctx(seed))
        q = row["questions"]["q_intent"]
        if q["type"] == "choice":
            assert row["targets"]["q_intent"] in q["criteria"]
            other_bucket += row["targets"]["q_intent"] in ("other", "none of these", "something else")
    assert other_bucket > 0


def test_civil_noul_targets_are_soft():
    from systemone.s1model.multitask import make_civil

    seen = set()
    for seed in range(60):
        row = make_civil(dict(SAMPLES["civil"]), _ctx(seed))
        if row["questions"]["q_tox"]["type"] == "noul":
            seen.add(row["targets"]["q_tox"])
    assert seen & {0.9, 0.85, 0.1}  # annotator fractions, not 0/1
