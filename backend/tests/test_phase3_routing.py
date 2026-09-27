from __future__ import annotations

import math

from systemone.adapters.base import TokenLogprob
from systemone.domains.builtin import COMPUTER_USE, SECOPS
from systemone.routing.confidence import Calibration, ConfidenceScorer, token_field_confidence
from systemone.routing.parse import parse_action

STATE = {"url": "u", "viewport_tree": [{"id": 3, "role": "button", "name": "Save"}], "temporal_buffer": []}


def toks(text: str, lp: dict[str, float] | None = None) -> list[TokenLogprob]:
    # one token per character; override logprob for chosen chars
    return [TokenLogprob(c, (lp or {}).get(c, 0.0)) for c in text]


def test_parse_action_variants():
    assert parse_action('{"action":"CLICK","target_id":3}')[0] == {"action": "CLICK", "target_id": 3}
    a, cot = parse_action("<think>the save button is id 3</think>\n```json\n{\"action\":\"CLICK\",\"target_id\":3}\n```")
    assert a["target_id"] == 3 and cot == "the save button is id 3"
    a, cot = parse_action('reasoning...</think>{"action_output": {"action": "ESCALATE"}}')
    assert a == {"action": "ESCALATE"} and cot == "reasoning..."
    assert parse_action("no json here") == (None, None)


def test_token_field_confidence_targets_decision_spans():
    text = '{"action":"CLICK","confidence_score":0.9,"target_id":3}'
    lps = [TokenLogprob(c, math.log(0.5) if c == "3" else 0.0) for c in text]
    fc = token_field_confidence(text, lps, ("action", "target_id"))
    assert fc["action"] == 1.0 and abs(fc["target_id"] - 0.5) < 1e-9


def test_scorer_blend_and_gates():
    sc = ConfidenceScorer(logprob_weight=0.5)
    text = '{"action":"CLICK","confidence_score":0.81,"target_id":3}'
    action = {"action": "CLICK", "confidence_score": 0.81, "target_id": 3}
    v = COMPUTER_USE.validate_action(action, STATE)
    lps = [TokenLogprob(c, math.log(0.64) if c == "3" else 0.0) for c in text]
    r = sc.score(COMPUTER_USE, text, action, v, lps)
    assert r.gates == [] and abs(r.confidence - math.sqrt(0.81 * 0.64)) < 1e-3

    bad = {"action": "CLICK", "confidence_score": 0.99, "target_id": 42}
    r = sc.score(COMPUTER_USE, text, bad, COMPUTER_USE.validate_action(bad, STATE), None)
    assert r.confidence == 0.0 and "ungrounded_reference" in r.gates

    esc = {"confidence_score": 0.99, "verdict": "SUSPICIOUS", "immediate_action": "ESCALATE", "target_ioc": []}
    r = sc.score(SECOPS, "{}", esc, SECOPS.validate_action(esc), None)
    assert r.confidence == 0.0 and "explicit_escalate" in r.gates

    r = sc.score(COMPUTER_USE, "garbage", None, None, None)
    assert r.gates == ["unparseable_output"]


def test_platt_calibration_fit_shrinks_overconfidence():
    scores = [0.95] * 50 + [0.9] * 50
    correct = [i % 2 == 0 for i in range(100)]  # 50% accurate but claims ~0.93
    cal = Calibration.fit(scores, correct)
    assert cal.apply(0.95) < 0.7
    assert Calibration().apply(0.42) == 0.42


def test_escalate_action_and_guided_schema():
    esc = SECOPS.escalate_action(0.3, {"verdict": "MALICIOUS"})
    assert esc["immediate_action"] == "ESCALATE" and esc["verdict"] == "MALICIOUS"
    assert SECOPS.validate_action(esc).ok
    assert COMPUTER_USE.validate_action(COMPUTER_USE.escalate_action(0.1)).ok
    assert COMPUTER_USE.guided_json_schema()["properties"]["action"]["enum"] == ["CLICK", "TYPE", "CLICK_XY", "ESCALATE"]
