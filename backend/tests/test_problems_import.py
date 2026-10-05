"""Training for your own problem: problem specs, imported data, hold-out suites, and the jobs around them."""

from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from systemone_builder.api.routes import kenning as routes
from systemone_builder.config import Settings
from systemone_builder.kenning import importer, problems
from systemone_builder.kenning import jobs as kj
from systemone_builder.system_one.contract import Question

SPEC = {
    "name": "door_access", "writes": "a request to open a building door",
    "state": {"fields": {"person": "who is asking", "door": "which door and when"}},
    "questions": [
        {"id": "allow", "ask": "Should the door open?", "labels": {"yes": "authorised", "no": "not authorised"}},
        {"id": "risk", "ask": ["How risky is it?"], "ordinal": True, "labels": {"low": "", "medium": "", "high": ""}},
        {"id": "reason", "ask": "Why?", "labels": {"badge": "valid badge", "escort": "escorted", "none": "no reason"}},
    ],
    "constraints": [{"if": {"allow": "no"}, "then": {"reason": "none"}}],
}


# ------------------------------------------------------------------ specs
def test_builtin_problems_are_valid_and_ask_valid_questions():
    built = problems.builtin_problems()
    assert {"computer_use", "insurance_claim", "chat_moderation"} <= set(built)
    for p in built.values():
        for q in problems.canonical_questions(p).values():
            Question.model_validate(q)


@pytest.mark.parametrize("change, msg", [
    ({"name": "Bad Name"}, "name"),
    ({"state": {}}, "state"),
    ({"questions": []}, "questions"),
    ({"questions": [{"id": "a", "ask": "?", "labels": {"only": ""}}]}, "labels"),
    ({"questions": [{"id": "a", "ask": "?", "ordinal": True, "labels": {"yes": "", "no": ""}}]}, "ordinal"),
    ({"constraints": [{"if": {"allow": "maybe"}, "then": {"reason": "none"}}]}, "constraints"),
    ({"constraints": [{"if": {"allow": "yes"}, "then": {"allow": "no"}}, {"if": {"allow": "no"}, "then": {"allow": "yes"}}]},
     "rule out"),
])
def test_bad_specs_are_rejected_with_a_reason(change, msg):
    with pytest.raises(ValueError, match=msg):
        problems.parse_problem({**SPEC, **change})


def test_drawn_labels_respect_constraints():
    p = problems.parse_problem(SPEC)
    rng = random.Random(1)
    for _ in range(300):
        labels = p.draw(rng)
        assert labels["allow"] == "yes" or labels["reason"] == "none"


def test_rows_and_holdout_items_follow_the_labels():
    p = problems.parse_problem(SPEC)
    cases = [{"state": {"person": f"p{i}", "door": "main"}, "labels": p.draw(random.Random(i))} for i in range(60)]
    train, held = problems.split(cases, 0.1, 7, p.name)
    assert len(held) == 20 and len(train) == 40  # at least 20 held out
    assert not {json.dumps(c) for c in train} & {json.dumps(c) for c in held}
    for row, case in zip(problems.training_rows(p, train, 7), train):
        assert row["state"] == case["state"]
        for qid, q in row["questions"].items():
            Question.model_validate(q)
            if q["type"] == "score":
                assert row["targets"][qid] == ["low", "medium", "high"].index(case["labels"][qid])
    for item, case in zip(problems.holdout_items(p, held), held):
        assert item["labels"]["allow"] == int(case["labels"]["allow"] == "yes")
        assert item["labels"]["risk"] == ["low", "medium", "high"].index(case["labels"]["risk"])
        assert item["labels"]["reason"] == case["labels"]["reason"]
    assert problems.split(cases[:30], 0.1, 7, "x") == (cases[:30], [])  # too few to hold out


def test_write_cases_keeps_only_cases_that_pass_the_blind_check():
    p = problems.parse_problem(SPEC)
    calls = {"write": 0, "check": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if prompt.startswith("Write"):
            calls["write"] += 1
            out = {"person": f"visitor {calls['write']}: " + prompt[prompt.index("allow: "):][:40], "door": "front"}
        else:  # the check: agree only when the intended allow label was "yes"
            calls["check"] += 1
            allow = "yes" if "allow: yes" in prompt else "no"
            out = {"allow": allow, "risk": "low", "reason": "badge" if allow == "yes" else "none"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(out)}}]})

    real = httpx.AsyncClient

    def fake_client(**kw):
        return real(transport=httpx.MockTransport(handler), **kw)

    stats: dict[str, int] = {}
    orig = problems.httpx.AsyncClient
    problems.httpx.AsyncClient = fake_client
    try:
        cases = asyncio.run(problems.write_cases(p, 6, 1, "http://teacher/v1", "m", stats=stats))
    finally:
        problems.httpx.AsyncClient = orig
    assert stats["disagreed"] > 0 and stats["asked"] <= 30  # retries, but never past 5x
    for c in cases:  # every kept case matches what the checker answered
        assert c["labels"]["risk"] == "low"
        assert c["labels"]["reason"] == ("badge" if c["labels"]["allow"] == "yes" else "none")


# ------------------------------------------------------------------ imports
def _write(path, lines):
    path.write_text("".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines), encoding="utf-8")


def test_import_accepts_both_formats_and_soft_targets(tmp_path):
    q = {"ok": {"type": "noul", "instructions": "OK?"},
         "kind": {"type": "choice", "instructions": "Kind?", "criteria": {"a": None, "b": None}},
         "lvl": {"type": "score", "instructions": "Level?", "criteria": ["low", "high"]}}
    _write(tmp_path / "rows.jsonl", [
        {"state": "x", "questions": {"ok": q["ok"]}, "targets": {"ok": 0.8}},
        {"state": {"t": "y"}, "questions": {"kind": q["kind"], "lvl": q["lvl"]},
         "targets": {"kind": {"a": 0.3, "b": 0.7}, "lvl": 1}},
    ])
    rows, qs = importer.load_rows(tmp_path / "rows.jsonl")
    assert len(rows) == 2 and set(qs) == {"ok", "kind", "lvl"}
    items = importer.holdout_items(rows, qs, "rows")
    assert items[0]["labels"] == {"ok": 1} and items[1]["labels"] == {"kind": "b", "lvl": 1}

    _write(tmp_path / "items.jsonl", [{"state": "z", "labels": {"ok": 1, "kind": "a"}}])
    (tmp_path / "q.json").write_text(json.dumps(q))
    rows, _ = importer.load_rows(tmp_path / "items.jsonl", tmp_path / "q.json")
    assert rows[0]["targets"] == {"ok": 1, "kind": "a"} and set(rows[0]["questions"]) == {"ok", "kind"}


def test_import_reports_every_kind_of_bad_line(tmp_path):
    q = {"type": "choice", "instructions": "Kind?", "criteria": {"a": None, "b": None}}
    _write(tmp_path / "bad.jsonl", [
        "not json",
        {"questions": {}, "targets": {}},
        {"state": "s", "questions": {"k": q}, "targets": {"k": "c"}},
        {"state": "s", "questions": {"k": q}, "targets": {"k": {"a": 0.9, "b": 0.9}}},
        {"state": "s", "questions": {"k": {"type": "noul", "instructions": "?"}}, "targets": {"k": 2}},
        {"state": "s", "labels": {"k": 1}},
        {"state": "s", "questions": {"k": q}, "targets": {"other": "a"}},
    ])
    with pytest.raises(importer.InvalidData) as e:
        importer.load_rows(tmp_path / "bad.jsonl")
    msg = str(e.value)
    for n, part in [(1, "not JSON"), (2, "state"), (3, "one of: a, b"), (4, "sum to 1"), (5, "noul target"),
                    (6, "questions file"), (7, "unknown question")]:
        assert f"line {n}:" in msg and part in msg, (n, msg)


def test_reworded_questions_are_left_out_of_the_holdout():
    fixed = {"ok": {"type": "noul", "instructions": "OK?"}}
    rows = [{"state": "a", "questions": fixed, "targets": {"ok": 1}},
            {"state": "b", "questions": {"ok": {"type": "noul", "instructions": "Is it fine?"}}, "targets": {"ok": 0}}]
    assert [x["state"] for x in importer.holdout_items(rows, fixed, "k")] == ["a"]


# ------------------------------------------------------------------ API and jobs
def _rt(home):
    return SimpleNamespace(settings=Settings(kenning_home=home))


def test_upload_validates_before_keeping_anything(tmp_path):
    rt = _rt(tmp_path)
    q = {"ok": {"type": "noul", "instructions": "OK?"}}
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes.upload_import(routes.ImportRequest(name="mine", license="CC0", content='{"state": "x"}\n',
                                                              questions=q), rt))
    assert e.value.status_code == 400 and "mine.jsonl" in e.value.detail
    assert not list((tmp_path / "imports").iterdir())  # nothing left behind
    good = "".join(json.dumps({"state": f"s{i}", "labels": {"ok": i % 2}}) + "\n" for i in range(50))
    out = asyncio.run(routes.upload_import(routes.ImportRequest(name="mine", license="CC0", content=good, questions=q), rt))
    assert out["rows"] == 50 and out["questions"] == ["ok"]
    assert kj.list_imports(tmp_path) == [{"name": "mine", "rows": 50, "questions": ["ok"], "license": "CC0",
                                          "size_bytes": len(good.encode())}]
    with pytest.raises(HTTPException, match="409|exists"):
        asyncio.run(routes.upload_import(routes.ImportRequest(name="mine", license="CC0", content=good, questions=q), rt))


def test_saved_problems_are_listed_and_builtins_protected(tmp_path):
    rt = _rt(tmp_path)
    asyncio.run(routes.save_problem(SPEC, rt))
    names = {p["name"]: p["builtin"] for p in kj.list_problems(tmp_path)}
    assert names["door_access"] is False and names["computer_use"] is True
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes.save_problem({**SPEC, "name": "computer_use"}, rt))
    assert e.value.status_code == 409
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes.save_problem({**SPEC, "questions": []}, rt))
    assert e.value.status_code == 400


def test_data_job_with_problems_and_imports(tmp_path):
    (tmp_path / "imports").mkdir()
    (tmp_path / "imports" / "mine.jsonl").write_text("{}\n")
    (tmp_path / "imports" / "mine.meta.json").write_text(json.dumps({"license": "CC0"}))
    with pytest.raises(kj.JobError, match="teacher"):
        kj.validate("data", {"name": "d", "sources": [], "problems": {"computer_use": 100}}, tmp_path,
                    pipeline=False, has_jev_key=False)
    with pytest.raises(kj.JobError, match="at least one"):
        kj.validate("data", {"name": "d", "sources": []}, tmp_path, pipeline=False, has_jev_key=False)
    with pytest.raises(kj.JobError, match="imports"):
        kj.validate("data", {"name": "d", "imports": ["../../etc/passwd"]}, tmp_path, pipeline=False, has_jev_key=False)
    with pytest.raises(kj.JobError, match="teacher_url"):
        kj.validate("data", {"name": "d", "teacher_url": "file:///etc", "teacher_model": "m"}, tmp_path,
                    pipeline=False, has_jev_key=False)
    p = kj.validate("data", {"name": "d", "sources": [], "problems": {"computer_use": 100}, "imports": ["mine"],
                             "teacher_url": "http://mac:11434/v1", "teacher_model": "qwen3.8:27b", "holdout": 0.2},
                    tmp_path, pipeline=False, has_jev_key=False)
    (argv,) = kj.commands("data", p, tmp_path, "")
    joined = " ".join(argv)
    assert "--sources none" in joined and "--problem computer_use=100" in joined and "--holdout 0.2" in joined
    assert argv[argv.index("--import") + 1].endswith("mine.jsonl") and "--import-licence CC0" in joined
    assert "--teacher-url http://mac:11434/v1 --teacher-model qwen3.8:27b" in joined and "--no-check" not in joined


def test_holdout_suites_benchmark_their_own_files(tmp_path):
    d = tmp_path / "datasets"
    (d / "d.holdout").mkdir(parents=True)
    (d / "d.jsonl").write_text("{}\n")
    (d / "d.holdout" / "problem-computer_use.jsonl").write_text("{}\n")
    (d / "d.holdout" / "problem-computer_use.questions.json").write_text("{}")
    (d / "d.manifest.json").write_text(json.dumps({"sources": {}, "holdouts": {
        "problem:computer_use": {"items": 1, "file": "d.holdout/problem-computer_use.jsonl"}}}))
    assert kj.list_datasets(tmp_path)[0]["holdouts"] == {"problem:computer_use": 1}
    p = kj.validate("bench", {"suites": ["general", "holdout:d/problem:computer_use"]}, tmp_path,
                    pipeline=False, has_jev_key=False)
    general, own = kj.commands("bench", p, tmp_path, "")
    assert general[general.index("--suite") + 1] == "general"
    assert own[own.index("--suite") + 1].endswith("problem-computer_use.jsonl")
    assert own[own.index("--questions") + 1].endswith("problem-computer_use.questions.json")
    for bad in ("holdout:d/nope", "holdout:missing/problem:computer_use", "holdout:../x/y"):
        with pytest.raises(kj.JobError):
            kj.validate("bench", {"suites": [bad]}, tmp_path, pipeline=False, has_jev_key=False)


# ------------------------------------------------------------------ calibrate
def test_calibrate_recovers_the_temperature_that_fits_the_labels(tmp_path, monkeypatch):
    """Answers served at T=1 from fixed scores; labels drawn as if the honest temperature were 2.5."""
    import math

    from systemone_builder.kenning import calibrate as cal

    rng = random.Random(3)
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}
    scores, lines = {}, []
    for i in range(600):
        s = [rng.gauss(0, 3) for _ in range(3)]
        scores[f"s{i}"] = s
        e = [math.exp(x / 2.5) for x in s]
        label = rng.choices("abc", weights=e)[0]
        lines.append({"state": f"s{i}", "questions": {"k": q}, "targets": {"k": label}})
    _write(tmp_path / "data.jsonl", lines)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "kenning.json").write_text(json.dumps({"name": "mine", "temperature": {"choice": 1.0}}))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        e = [math.exp(x) for x in scores[body["state"]]]
        probs = {o: v / sum(e) for o, v in zip("abc", e)}
        return httpx.Response(200, json={"model": "mine", "answers": {"k": {
            "type": "choice", "choice": max(probs, key=probs.get), "confidence": 0.5, "probabilities": probs}}})

    real = httpx.AsyncClient
    monkeypatch.setattr(cal.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    report = cal.calibrate(tmp_path / "m", tmp_path / "data.jsonl", "http://kenning")
    assert 2.0 < report["temperature_after"]["choice"] < 3.1
    assert report["after"]["all"]["accuracy"] == report["before"]["all"]["accuracy"]  # temperature never flips a winner
    assert report["after"]["all"]["nll"] < report["before"]["all"]["nll"]
    cal.save(tmp_path / "m", report, "mine-calibrated")
    saved = json.loads((tmp_path / "m" / "kenning.json").read_text())
    assert saved["name"] == "mine-calibrated" and saved["temperature"] == report["temperature_after"]
    assert saved["calibration_history"][0]["temperature"]["choice"] == 1.0
