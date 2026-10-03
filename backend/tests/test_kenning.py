"""System One model: templates, targets, calibration and training data (no torch needed)."""

from __future__ import annotations

import json
import math
import random

import pytest

from systemone_builder.system_one.contract import Question
from systemone_builder.system_one.factory import api_engine, build_engine
from systemone_builder.kenning.data import SAFE_OPTION, body_hash, excluded_hashes, questions_for
from systemone_builder.kenning.model import hypotheses
from systemone_builder.kenning.train import fit_temperatures, metrics, micro_batches, target_vector


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


def test_engine_factory_targets_the_kenning_server():
    from systemone_builder.config import Settings

    s = Settings(_env_file=None)
    e = build_engine("kenning", s)
    assert str(e.client.base_url) == "http://kenning:8000" and e._require_key is False
    assert api_engine(s).name == "kenning"
    assert api_engine(Settings(_env_file=None, system_one_backend="model")).name == "kenning"  # old value still works
    assert api_engine(Settings(_env_file=None, system_one_backend="logprob")).name.startswith("local")


# ------------------------------------------------------------ multi-task data
def _ctx(seed=0):
    from systemone_builder.kenning.multitask import Ctx

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
    from systemone_builder.kenning.multitask import SOURCES

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
    from systemone_builder.kenning.multitask import make_clinc, make_dbpedia

    from systemone_builder.kenning.multitask import Ctx

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
    from systemone_builder.kenning.multitask import make_civil

    seen = set()
    for seed in range(60):
        row = make_civil(dict(SAMPLES["civil"]), _ctx(seed))
        if row["questions"]["q_tox"]["type"] == "noul":
            seen.add(row["targets"]["q_tox"])
    assert seen & {0.9, 0.85, 0.1}  # annotator fractions, not 0/1


# ------------------------------------------------------------ registry & export
def _fake_model(home, name, created="2026-10-01T00:00:00Z", legacy=False):
    d = home / "models" / name
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\0" * 64)
    (d / "tokenizer.json").write_text("{}")
    cfg = {"name": name, "base_model": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "created": created,
           "trained_on": "/workspace/kenning/datasets/multitask-train.jsonl", "temperature": {"noul": 1.5},
           "heldout": {"zero_shot": {"all": {"accuracy": 0.7}}, "calibrated": {"all": {"accuracy": 0.93, "ece": 0.008}}}}
    (d / ("s1_config.json" if legacy else "kenning.json")).write_text(json.dumps(cfg))
    return d


def test_registry_lists_activates_and_protects_the_active_model(tmp_path):
    from systemone_builder.kenning import registry

    _fake_model(tmp_path, "kenning-a", created="2026-10-01T00:00:00Z")
    _fake_model(tmp_path, "kenning-b", created="2026-10-02T00:00:00Z", legacy=True)
    names = [m["name"] for m in registry.list_models(tmp_path)]
    assert names == ["kenning-b", "kenning-a"]  # newest first, legacy config still read
    registry.set_active(tmp_path, "kenning-a")
    assert registry.active(tmp_path) == "kenning-a"
    assert registry.summary(tmp_path, "kenning-a")["heldout"]["calibrated"]["accuracy"] == 0.93
    with pytest.raises(registry.RegistryError, match="active"):
        registry.delete_model(tmp_path, "kenning-a")
    registry.delete_model(tmp_path, "kenning-b")
    assert [m["name"] for m in registry.list_models(tmp_path)] == ["kenning-a"]
    for bad in ("../etc", "a/b", ".hidden", ""):
        with pytest.raises(registry.RegistryError):
            registry.model_dir(tmp_path, bad)


def test_export_bundle_has_card_notice_and_checksums(tmp_path):
    import hashlib
    import zipfile

    from systemone_builder.kenning import registry

    _fake_model(tmp_path, "kenning-a")
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "multitask-train.manifest.json").write_text(json.dumps({"sources": {
        "boolq": {"dataset": "google/boolq", "rows": 10, "license": "CC-BY-SA-3.0"}}}))
    out = registry.export_bundle(tmp_path, "kenning-a")
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"kenning-a/model.safetensors", "kenning-a/kenning.json", "kenning-a/README.md",
                "kenning-a/NOTICE.md", "kenning-a/SHA256SUMS"} <= names
        card = z.read("kenning-a/README.md").decode()
        assert "google/boolq" in card and "not affiliated" in card
        # the fake model sits on a base with non-commercial fine-tuning data: no Apache-2.0
        assert "license: other" in card and "Not released under Apache-2.0" in card
        assert "kenning-a/LICENSE" not in names
        assert "CC-BY-SA-3.0" in z.read("kenning-a/NOTICE.md").decode()
        sums = dict(line.split("  ")[::-1] for line in z.read("kenning-a/SHA256SUMS").decode().split("\n") if line)
        assert sums["kenning-a/model.safetensors"] == hashlib.sha256(z.read("kenning-a/model.safetensors")).hexdigest()
    assert registry.export_bundle(tmp_path, "kenning-a") == out  # reused while the model is unchanged


def test_clean_base_bundle_is_apache_2_with_licence_text(tmp_path):
    import zipfile

    from systemone_builder.kenning import registry

    d = _fake_model(tmp_path, "kenning-c")
    cfg = json.loads((d / "kenning.json").read_text())
    cfg["base_model"] = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
    (d / "kenning.json").write_text(json.dumps(cfg))
    with zipfile.ZipFile(registry.export_bundle(tmp_path, "kenning-c")) as z:
        card = z.read("kenning-c/README.md").decode()
        assert card.startswith("---\nlicense: apache-2.0\n") and "Apache License 2.0" in card
        assert "Apache License" in z.read("kenning-c/LICENSE").decode()
        assert "MIT" in z.read("kenning-c/NOTICE.md").decode()


def test_server_only_loads_registered_models(tmp_path, monkeypatch):
    from systemone_builder.kenning import serve

    monkeypatch.setenv("KENNING_HOME", str(tmp_path))
    monkeypatch.setenv("KENNING_MODEL", "org/base-model")
    _fake_model(tmp_path, "kenning-a")
    assert serve.resolve("kenning-a") == str(tmp_path / "models" / "kenning-a")
    assert serve.resolve("org/base-model") == "org/base-model"  # the configured default
    for bad in ("../x", "/etc", "org/other-model", "missing"):
        with pytest.raises(ValueError):
            serve.resolve(bad)
    (tmp_path / "active.json").write_text(json.dumps({"model": "kenning-a"}))
    assert serve.startup_model().endswith("kenning-a")
    (tmp_path / "active.json").write_text(json.dumps({"model": "gone"}))
    assert serve.startup_model() == "org/base-model"


def test_client_library_templates_match_the_server():
    """systemone.local (embedded mode) must ask the model exactly what the server asks."""
    local = pytest.importorskip("systemone.local")
    cases = [
        {"type": "noul", "instructions": "Is it bad?", "criteria": "phishing counts"},
        {"type": "noul", "instructions": "Is it bad?"},
        {"type": "choice", "instructions": "Kind?", "criteria": {"Safe": {"what": "fine", "examples": ["a", "b"]}, "Phish": None}},
        {"type": "score", "instructions": "Severity?", "criteria": ["Low", {"what": "High", "not_for": "typos"}]},
    ]
    for c in cases:
        assert local.hypotheses(c) == hypotheses(Question.model_validate(c))
    assert local.spread_confidence([0.6, 0.3, 0.1]) == pytest.approx(
        __import__("systemone_builder.system_one.contract", fromlist=["x"]).spread_confidence([0.6, 0.3, 0.1]))


# ------------------------------------------------------------ state layouts
def _all_text(x):
    if isinstance(x, dict):
        return " ".join(_all_text(v) for v in x.values())
    return str(x)


def test_layout_variation_keeps_the_content_and_varies_the_shape():
    from systemone_builder.kenning.layouts import vary

    body = "Your account is locked, confirm your password at http://reset.example"
    shapes = set()
    for seed in range(200):
        out = vary({"email": {"recipient": "a@b.com", "body": body}}, random.Random(seed))
        assert body in _all_text(out)  # the content is never lost
        shapes.add(type(out).__name__ + ":" + (",".join(sorted(out)) if isinstance(out, dict) else "text"))
    assert len(shapes) > 20  # many different layouts
    assert any(s.startswith("str:") for s in shapes)  # some plain text


def test_synthetic_headers_do_not_depend_on_the_label():
    from systemone_builder.kenning.layouts import enrich_email

    # same rng state -> same sender/date whatever the body (and so whatever the label)
    a = enrich_email({"body": "Hello team, lunch at noon"}, random.Random(5))
    b = enrich_email({"body": "Send gift cards now or your account closes"}, random.Random(5))
    assert a["from"] == b["from"] and a.get("date") == b.get("date")


def test_email_layouts_cover_the_four_benchmark_shapes():
    from systemone_builder.kenning.layouts import email_layouts

    layouts = email_layouts("Click here to verify your bank details", random.Random(1))
    assert set(layouts) == {"original", "headers_json", "plain_text", "nested_metadata"}
    assert all("verify your bank details" in _all_text(v) for v in layouts.values())
    assert layouts["plain_text"].startswith("From: ")


async def test_synthetic_emails_take_labels_from_the_scenario(monkeypatch):
    import httpx

    from systemone_builder.kenning import synthetic_email as se

    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})  # dropped
        body = json.loads(req.content)
        text = body["messages"][0]["content"]
        email = {"from": "a@b.example", "subject": "Hi", "body": f"email {calls['n']} :: {text[:90]}"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(email)}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    rows = await se.generate("http://teacher/v1", "m", 10, seed=1, concurrency=4)
    assert 0 < len(rows) < 20  # one in five answers was not valid JSON
    for r in rows:
        assert r["malicious"] == ("phishing" in r["email"]["body"])  # label follows the requested scenario
        assert set(r["email"]) == {"from", "subject", "body"}


def test_synthetic_task_rows_follow_the_requested_labels():
    from systemone_builder.kenning.synthetic_tasks import TASKS, rows_for

    rng = random.Random(0)
    for task in TASKS:
        for seed in range(30):
            labels = {a.name: random.Random(seed).choice(list(a.labels)) for a in task.attrs}
            row = rows_for(task, "some text", labels, rng)
            for qid, qd in row["questions"].items():
                q = Question.model_validate(qd)
                _, options = hypotheses(q)
                vec = target_vector(q, options, row["targets"][qid])
                assert sum(vec) == pytest.approx(1.0)
                if q.type == "choice":
                    assert row["targets"][qid] == labels[qid]
                if q.type == "score":
                    attr = next(a for a in task.attrs if a.name == qid)
                    assert row["targets"][qid] == list(attr.labels).index(labels[qid])


def test_nli_fiction_genre_is_excluded():
    from systemone_builder.kenning.multitask import accept_nli

    ctx = _ctx()
    assert accept_nli({"genre": "government"}, ctx) and not accept_nli({"genre": "fiction"}, ctx)


def test_distillation_blends_teacher_and_labels_and_reports_agreement(monkeypatch):
    import httpx

    from systemone_builder.kenning import distill

    rows = [
        {"source": "a", "state": "s1", "questions": {"q": {"type": "noul", "instructions": "x?"}}, "targets": {"q": 1}},
        {"source": "a", "state": "s2", "questions": {"c": {"type": "choice", "instructions": "k?",
                                                           "criteria": {"x": None, "y": None}}}, "targets": {"c": "x"}},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        reqs = json.loads(req.content)["requests"]
        out = []
        for r in reqs:
            if "q" in r["questions"]:
                out.append({"answers": {"q": {"type": "noul", "noul": 0.8}}})
            else:
                out.append({"answers": {"c": {"type": "choice", "choice": "y", "confidence": 0.6,
                                              "probabilities": {"x": 0.4, "y": 0.6}}}})
        return httpx.Response(200, json={"responses": out})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    out, report = distill.label_rows(rows, "http://teacher", alpha=0.5)
    assert out[0]["targets"]["q"] == pytest.approx(0.9)            # (1 + 0.8) / 2
    assert out[1]["targets"]["c"] == pytest.approx({"x": 0.7, "y": 0.3})  # (1,0)/2 + (0.4,0.6)/2
    assert out[0]["label_targets"] == {"q": 1}
    assert report["a"]["questions"] == 2 and report["a"]["agree"] == 1  # agrees on q, not on c
    for row in out:  # the blended targets still feed the trainer
        for qid, qd in row["questions"].items():
            qq = Question.model_validate(qd)
            assert sum(target_vector(qq, hypotheses(qq)[1], row["targets"][qid])) == pytest.approx(1.0)


def test_micro_batches_bound_pairs_and_keep_groups_whole():
    groups = [{"hyps": ["h"] * k} for k in (5, 30, 20, 60, 2, 2)]
    runs = micro_batches(groups, 48)
    assert [g for run in runs for g in run] == groups  # order kept, nothing dropped
    assert all(sum(len(g["hyps"]) for g in run) <= 48 or len(run) == 1 for run in runs)
    assert [len(run) for run in runs] == [2, 1, 1, 2]  # an oversized group runs alone


def test_pipeline_actions_are_refused_when_the_pipeline_is_off():
    from types import SimpleNamespace

    from fastapi import HTTPException

    from systemone_builder.api.routes.training import pipeline_on

    with pytest.raises(HTTPException) as exc:
        pipeline_on(SimpleNamespace(settings=SimpleNamespace(pipeline=False)))
    assert exc.value.status_code == 409 and "S1_PIPELINE" in exc.value.detail
    rt = SimpleNamespace(settings=SimpleNamespace(pipeline=True))
    assert pipeline_on(rt) is rt
