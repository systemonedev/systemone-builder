"""System One contract, engines and benchmark metrics (no network, no GPU)."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from systemone.adapters.base import Generation, ModelAdapter
from systemone.s1.bench import Item, Suite, engine_report, ece
from systemone.s1.contract import (
    Question,
    SystemOneRequest,
    SystemOneResponse,
    choice_answer,
    score_answer,
    spread_confidence,
)
from systemone.s1.engines import EngineError, JevEngine, LLMJsonEngine, LocalLogprobEngine, label_mass, local_prompt

# The response example from TypeSafe's quick start, verbatim.
JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {"type": "choice", "choice": "technical", "confidence": 0.78,
                       "probabilities": {"technical": 0.85, "sales": 0.0, "billing": 0.15}},
        "frustration": {"type": "score", "score": 1.0, "confidence": 1.0,
                        "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"},
                        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
        "is_urgent": {"type": "noul", "noul": 1.0},
    },
    "usage": {"input_tokens": 392, "output_tokens": 65},
}

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "Exchanges", "billing": None, "shipping": None}},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Cosmetic", "Degraded", "Blocking"]},
}


# ------------------------------------------------------------- contract
def test_parses_jev_documented_response():
    r = SystemOneResponse.model_validate(JEV_RESPONSE)
    assert r.choices["department"].choice == "technical"
    assert r.scores["frustration"].score == 1.0
    assert r.nouls["is_urgent"].noul == 1.0
    assert r.usage.input_tokens == 392


def test_confidence_matches_jev_examples():
    # Jev docs: 0.85 over three options -> 0.78; 0.57 over three levels -> 0.35
    assert spread_confidence([0.85, 0.0, 0.15]) == pytest.approx(0.78, abs=0.01)
    assert spread_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.35, abs=0.01)
    assert spread_confidence([1.0, 0.0]) == 1.0
    assert spread_confidence([0.5, 0.5]) == 0.0


def test_score_is_probability_weighted_level_mean():
    a = score_answer(["a", "b", "c"], [0.0, 0.57, 0.43])  # Jev docs example
    assert a.score == pytest.approx(1.43)
    assert a.legend == {"0": "a", "1": "b", "2": "c"}


def test_choice_ties_resolve_to_first_option():
    assert choice_answer(["x", "y"], [0.5, 0.5]).choice == "x"


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "x", "criteria": {"only": None}},
    {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
    {"type": "score", "instructions": "x", "criteria": ["one"]},
    {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
    {"type": "noul", "instructions": "x", "criteria": ["not", "allowed"]},
    {"type": "noul", "instructions": ""},
])
def test_rejects_invalid_questions(q):
    with pytest.raises(ValueError):
        Question.model_validate(q)


# -------------------------------------------------------------- local
def test_label_mass_sums_tokenizer_variants():
    top = [{"token": "Yes", "logprob": math.log(0.7)}, {"token": " Yes", "logprob": math.log(0.1)},
           {"token": "No", "logprob": math.log(0.15)}, {"token": "Maybe", "logprob": math.log(0.05)}]
    assert label_mass(top, ["Yes", "No"]) == pytest.approx([0.8, 0.15])


def test_local_prompt_labels():
    _, labels, options = local_prompt("s", Question.model_validate(QUESTIONS["team"]))
    assert labels == ["A", "B", "C"] and options == ["returns", "billing", "shipping"]
    _, labels, _ = local_prompt("s", Question.model_validate(QUESTIONS["severity"]))
    assert labels == ["0", "1", "2"]


def _fake_vllm(request: httpx.Request) -> httpx.Response:
    user = json.loads(request.content)["messages"][1]["content"]
    if "Yes or No" in user:
        top = [("Yes", 0.9), ("No", 0.1)]
    elif "option letter" in user:
        top = [("B", 0.6), ("A", 0.3), ("C", 0.1)]
    else:
        top = [("2", 0.5), ("1", 0.5)]
    return httpx.Response(200, json={
        "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top]}]}}],
        "usage": {"prompt_tokens": 50},
    })


async def test_local_engine_reads_distributions():
    eng = LocalLogprobEngine("http://local/v1", "m")
    eng.client = httpx.AsyncClient(base_url="http://local/v1", transport=httpx.MockTransport(_fake_vllm))
    r = await eng.answer(SystemOneRequest(state={"ticket": "x"}, questions=QUESTIONS))
    assert r.nouls["urgent"].noul == pytest.approx(0.9)
    assert r.choices["team"].choice == "billing"
    assert r.choices["team"].probabilities["returns"] == pytest.approx(0.3)
    assert r.scores["severity"].score == pytest.approx(1.5)
    assert r.usage.input_tokens == 150 and r.usage.output_tokens == 3


# ---------------------------------------------------------------- Jev
async def test_jev_engine_sends_contract_and_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=JEV_RESPONSE)

    eng = JevEngine("apikey_test")
    eng.client = httpx.AsyncClient(base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler),
                                   headers={"Authorization": "Bearer apikey_test"})
    r = await eng.answer(SystemOneRequest(state="hello", questions={"u": QUESTIONS["urgent"]}))
    assert seen["auth"] == "Bearer apikey_test"
    assert seen["body"] == {"state": "hello", "model": "jev-latest",
                            "questions": {"u": {"type": "noul", "instructions": "Is this urgent?"}}}
    assert r.model == "jev-1.13.0" and r.latency_ms is not None


async def test_jev_engine_requires_key():
    with pytest.raises(EngineError, match="S1_API_KEY"):
        await JevEngine(None).answer(SystemOneRequest(state="x", questions={"u": QUESTIONS["urgent"]}))


# ---------------------------------------------------------------- LLM
class FakeAdapter(ModelAdapter):
    kind = "fake"

    def __init__(self, text: str) -> None:
        super().__init__("http://x", "fake")
        self.text = text

    async def generate(self, messages, **kw):  # noqa: ANN001, ANN003
        return Generation(text=self.text, model="fake", prompt_tokens=10, completion_tokens=20)

    async def health(self) -> dict:
        return {"ok": True}

    async def list_models(self) -> list[str]:
        return ["fake"]


async def test_llm_engine_maps_json_and_rejects_bad_output():
    good = json.dumps({"urgent": {"probability_yes": 0.8}, "team": {"choice": "shipping", "confidence": 0.7},
                       "severity": {"level": 2, "confidence": 0.9}})
    r = await LLMJsonEngine(FakeAdapter(good)).answer(SystemOneRequest(state="x", questions=QUESTIONS))
    assert r.nouls["urgent"].noul == pytest.approx(0.8)
    assert r.choices["team"].choice == "shipping"
    assert r.scores["severity"].probabilities["2"] == pytest.approx(0.9)
    bad = json.dumps({"urgent": {"probability_yes": 0.8}, "team": {"choice": "nope", "confidence": 1}})
    with pytest.raises(EngineError):
        await LLMJsonEngine(FakeAdapter(bad)).answer(SystemOneRequest(state="x", questions=QUESTIONS))


# ------------------------------------------------------------ metrics
def test_ece_perfectly_calibrated_is_zero():
    assert ece([0.9] * 10, [1.0] * 9 + [0.0]) == pytest.approx(0.0)
    assert ece([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)


def test_gate_metrics():
    q = {"bad": Question(type="noul", instructions="bad?")}
    items = [Item(str(i), "s", {"bad": y}) for i, y in enumerate([1, 1, 0, 0, 1])]
    ps = [0.95, 0.5, 0.05, 0.92, 0.02]  # TP auto, human, TN auto, FP auto, FN auto
    results = [{"id": str(i), "answers": {"bad": {"type": "noul", "noul": p}}, "latency_ms": 10.0, "error": None}
               for i, p in enumerate(ps)]
    rep = engine_report("e", Suite("t", "", q, items, gate="bad"), results, 1.0, 0.9, 0.1, {"checked": 0, "identical": 0})
    g = rep["questions"]["bad"]["gate"]
    assert (g["auto_positive"], g["auto_negative"], g["to_human"]) == (2, 2, 1)
    assert (g["false_positives_acted"], g["false_negatives_closed"]) == (1, 1)
    assert g["automation_rate"] == pytest.approx(0.8)
    assert g["automated_accuracy"] == pytest.approx(0.5)
