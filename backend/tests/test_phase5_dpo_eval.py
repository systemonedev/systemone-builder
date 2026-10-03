from __future__ import annotations

from systemone_builder.domains.builtin import COMPUTER_USE, SECOPS
from systemone_builder.dpo.delta import compute_delta, infer_failure
from systemone_builder.evaluation.metrics import calibration, distribution, full_match, percentile

BEFORE = {"url": "https://app/form", "viewport_tree": [
    {"id": 1, "role": "textbox", "name": "Email"}, {"id": 2, "role": "button", "name": "Save", "bbox": [100, 200, 80, 30]}]}


def test_dom_delta_detects_error_banner():
    after = {"url": "https://app/form", "viewport_tree": [*BEFORE["viewport_tree"], {"id": 3, "role": "alert", "name": "Email is required"}]}
    d = compute_delta("computer_use", BEFORE, after)
    assert d["added"] == [{"role": "alert", "name": "Email is required"}]
    assert d["error_signals"] and infer_failure("computer_use", d)


def test_dom_delta_no_effect_is_failure_and_navigation_is_not():
    assert infer_failure("computer_use", compute_delta("computer_use", BEFORE, BEFORE))
    nav = {"url": "https://app/done", "viewport_tree": [{"id": 1, "role": "heading", "name": "Saved"}]}
    d = compute_delta("computer_use", BEFORE, nav)
    assert d["url_changed"] and not infer_failure("computer_use", d)


def test_log_delta_flags_successful_attack_after_allow():
    before = {"source": "suricata_eve", "src_ip": "1.2.3.4", "payload_snippet": "GET /../etc/passwd"}
    after = {"source": "suricata_eve", "src_ip": "1.2.3.4", "http": {"status": 200}, "payload_snippet": "root:x:0:0"}
    assert infer_failure("secops", compute_delta("secops", before, after))


def test_full_match_rules():
    exp = {"action": "CLICK", "target_id": 2}
    assert full_match(COMPUTER_USE, {"action": "CLICK", "target_id": 2}, [exp], BEFORE) == (True, True)
    assert full_match(COMPUTER_USE, {"action": "CLICK", "target_id": 1}, [exp], BEFORE) == (True, False)
    # spatial fallback landing inside the expected element's bbox counts
    xy = {"action": "CLICK_XY", "coordinates": {"x": 140, "y": 215}}
    assert full_match(COMPUTER_USE, {**xy, "action": "CLICK"}, [exp], BEFORE)[1]
    typed = {"action": "TYPE", "target_id": 1, "text": "a@b.c"}
    assert not full_match(COMPUTER_USE, {**typed, "text": "x"}, [typed], BEFORE)[1]
    sec = {"verdict": "MALICIOUS", "immediate_action": "DROP_AND_BLACKLIST_IP", "target_ioc": ["1.2.3.4"]}
    assert full_match(SECOPS, dict(sec), [sec], {})[1]
    assert not full_match(SECOPS, {**sec, "target_ioc": ["9.9.9.9"]}, [sec], {})[1]
    esc = {"verdict": "SUSPICIOUS", "immediate_action": "ESCALATE", "target_ioc": []}
    assert full_match(SECOPS, {**esc, "verdict": "MALICIOUS"}, [esc], {})[1]
    # acceptable alternatives
    assert full_match(COMPUTER_USE, {"action": "CLICK", "target_id": 1}, [exp, {"action": "CLICK", "target_id": 1}], BEFORE)[1]


def test_distribution_and_calibration():
    d = distribution([10, 20, 30, 40, 200])
    assert d["p50"] == 30 and d["under_100ms"] == 0.8 and sum(b["count"] for b in d["histogram"]) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    c = calibration([0.9, 0.9, 0.9, 0.9], [True, True, False, False])
    assert abs(c["ece"] - 0.4) < 1e-9 and abs(c["brier"] - 0.41) < 1e-9
