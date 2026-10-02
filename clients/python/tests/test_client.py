"""systemone client: wire format, parsing, errors (no server needed)."""

from __future__ import annotations

import json

import httpx
import pytest

from systemone import AsyncClient, Choice, Client, Noul, Score, SystemOneError
from systemone.local import hypotheses, spread_confidence

WIRE = {
    "model": "kenning-large-v0.1",
    "answers": {
        "billing": {"type": "noul", "noul": 0.97},
        "team": {"type": "choice", "choice": "billing", "confidence": 0.9,
                 "probabilities": {"billing": 0.93, "technical": 0.07}},
        "urgency": {"type": "score", "score": 0.9, "confidence": 0.4, "legend": {"0": "later", "1": "now"},
                    "probabilities": {"0": 0.1, "1": 0.9}},
    },
    "usage": {"input_tokens": 120, "output_tokens": 0},
    "latency_ms": 31.5,
}
QUESTIONS = {
    "billing": Noul("Is this about billing?"),
    "team": Choice("Which team?", {"billing": "Charges, refunds", "technical": None}),
    "urgency": Score("How urgent?", ["later", "now"]),
}


def test_sends_wire_format_and_parses_answers():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"], seen["key"] = req.url.path, json.loads(req.content), req.headers.get("x-api-key")
        return httpx.Response(200, json=WIRE)

    c = Client("http://server/api", api_key="k")
    c._http = httpx.Client(base_url=c.base_url, transport=httpx.MockTransport(handler), headers={"X-API-Key": "k"})
    r = c.system_one(state={"ticket": "charged twice"}, questions=QUESTIONS)
    assert seen["path"] == "/api/v1/systemone" and seen["key"] == "k"
    assert seen["body"]["questions"]["team"] == {"type": "choice", "instructions": "Which team?",
                                                 "criteria": {"billing": "Charges, refunds", "technical": None}}
    assert seen["body"]["questions"]["billing"] == {"type": "noul", "instructions": "Is this about billing?"}
    assert r.nouls["billing"].noul == 0.97 and r.choices["team"].choice == "billing"
    assert r.scores["urgency"].probabilities["1"] == 0.9 and r.usage.input_tokens == 120


def test_http_errors_raise_with_detail():
    c = Client("http://server")
    c._http = httpx.Client(base_url=c.base_url,
                           transport=httpx.MockTransport(lambda r: httpx.Response(422, json={"detail": "bad question"})))
    with pytest.raises(SystemOneError, match="bad question") as e:
        c.system_one(state="x", questions={"a": Noul("?")})
    assert e.value.status == 422


async def test_async_client():
    c = AsyncClient("http://server")
    c._http = httpx.AsyncClient(base_url=c.base_url, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=WIRE)))
    r = await c.system_one(state="x", questions=QUESTIONS)
    assert r.model == "kenning-large-v0.1"
    await c.aclose()


def test_question_validation():
    with pytest.raises(ValueError):
        Choice("x", {"only": None})
    with pytest.raises(ValueError):
        Score("x", ["one"])
    with pytest.raises(ValueError):
        Client("http://server").system_one(state="x", questions={})


def test_local_templates_and_confidence():
    texts, opts = hypotheses(Choice("Kind?", {"Safe": "fine", "Phish": None}).to_dict())
    assert texts == ["Kind? Answer: Safe. fine", "Kind? Answer: Phish."] and opts == ["Safe", "Phish"]
    assert spread_confidence([0.85, 0.0, 0.15]) == pytest.approx(0.775)
