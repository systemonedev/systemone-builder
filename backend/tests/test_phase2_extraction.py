from __future__ import annotations

import base64

from systemone_builder.domains.builtin import COMPUTER_USE, SECOPS
from systemone_builder.extraction.fuzzy import EntityVault, FuzzyScrubber, ScrubPolicy
from systemone_builder.extraction.pipeline import Observation, StateExtractor
from systemone_builder.extraction.prompt import PrefixTracker, canonical_state, format_action

PAGE = """<html><body>
<h1>Sign in</h1><div id="lbl">Email address</div><input aria-labelledby="lbl">
<label for="pw">Password</label><input id="pw" type="password" value="hunter2">
<input type="hidden" name="csrf" value="abc"><div style="display:none"><button>Ghost</button></div>
<button id="ember{n}">Log in</button>
<p>Last refreshed {ts} - request {uuid}</p>
<div role="status">Updated {n} seconds ago</div>
</body></html>"""


def render(n: int, ts: str, uuid: str) -> str:
    return PAGE.format(n=n, ts=ts, uuid=uuid)


def test_scrubber_rules():
    s = FuzzyScrubber()
    out = s.scrub_text(
        "at 2024-09-26T12:00:01.123Z id 550e8400-e29b-41d4-a716-446655440000 "
        "tok eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U "
        "hash 9f86d081884c7d659a2feaa0c55ad015 epoch 1727395200 host 10.0.0.1:8080 5 minutes ago"
    )
    assert "<TS>" in out and "<UUID>" in out and "<JWT>" in out and "<HEX>" in out and "<EPOCH>" in out
    assert "10.0.0.1:8080" in out and "<RELTIME>" in out


def test_url_query_params_scrubbed():
    s = FuzzyScrubber()
    url = s.scrub_url("https://app.example.com/inbox?folder=spam&_=1727395200123&csrf=AbC123xyz#msg-99831")
    assert url == "https://app.example.com/inbox?folder=spam&_=<VAR>&csrf=<VAR>"


def test_entity_vault_roundtrip():
    s = FuzzyScrubber(ScrubPolicy(tokenize_ips=True))
    v = EntityVault()
    out = s.scrub({"msg": "from 203.0.113.9 to 10.0.0.2 and again 203.0.113.9"}, v)
    assert out["msg"] == "from <IP_1> to <IP_2> and again <IP_1>"
    action = {"target_ioc": ["<IP_1>"]}
    assert v.rehydrate(action) == {"target_ioc": ["203.0.113.9"]}
    assert EntityVault.from_dict(v.to_dict()).rehydrate("<IP_2>") == "10.0.0.2"


def test_dom_extraction_is_prefix_stable_across_volatile_changes():
    ex = StateExtractor(COMPUTER_USE)
    a = ex.extract(Observation(kind="html", data=render(4411, "12:00:01", "550e8400-e29b-41d4-a716-446655440000"), url="https://x.io/login?ts=1"))
    b = ex.extract(Observation(kind="html", data=render(9, "18:44:57", "123e4567-e89b-12d3-a456-426614174000"), url="https://x.io/login?ts=2"))
    assert a.canonical == b.canonical and a.state_hash == b.state_hash
    names = [(n["role"], n["name"]) for n in a.state["viewport_tree"]]
    assert ("textbox", "Email address") in names and ("textbox", "Password") in names
    assert ("button", "Log in") in names and all(n != "Ghost" for _, n in names)
    pw = next(n for n in a.state["viewport_tree"] if n["name"] == "Password")
    assert pw["value"] == "<SECRET>"
    assert a.validation_errors == []


def test_elements_with_bbox_and_viewport_filter():
    ex = StateExtractor(COMPUTER_USE)
    els = [
        {"role": "button", "name": "Save", "bbox": [10, 10, 80, 30], "selector": "#save"},
        {"role": "button", "name": "Below fold", "bbox": [10, 2000, 80, 30]},
        {"role": "link", "name": "Hidden", "bbox": [0, 0, 0, 0], "visible": False},
    ]
    res = ex.extract(Observation(kind="elements", data=els, url="https://a.b/", viewport=(1280, 800), temporal_buffer=["CLICK(1)"]))
    assert res.state["viewport_tree"] == [{"id": 1, "role": "button", "name": "Save", "bbox": [10, 10, 80, 30]}]
    assert res.index[1]["selector"] == "#save"
    assert res.state["temporal_buffer"] == ["CLICK(1)"]


def test_eve_extraction_drops_volatile_and_keeps_payload():
    eve = {
        "timestamp": "2024-09-26T12:00:01.000000+0000", "flow_id": 1234567890123456, "event_type": "alert",
        "src_ip": "192.168.1.150", "src_port": 51514, "dest_ip": "10.0.0.5", "dest_port": 80, "proto": "TCP",
        "alert": {"signature": "ET WEB_SERVER /etc/passwd Detected", "signature_id": 2049029, "severity": 1, "category": "Attempted Information Leak", "action": "allowed"},
        "payload": base64.b64encode(b"GET /../../../../etc/passwd HTTP/1.1\r\n\r\n").decode(),
    }
    res = StateExtractor(SECOPS).extract(Observation(kind="suricata_eve", data=eve))
    st = res.state
    assert st["source"] == "suricata_eve" and st["src_ip"] == "192.168.1.150"
    assert st["payload_snippet"] == "GET /../../../../etc/passwd HTTP/1.1\r\n\r\n"
    assert "timestamp" not in st and "flow_id" not in st and "src_port" not in st
    assert res.validation_errors == []
    eve2 = {**eve, "timestamp": "2024-09-27T01:02:03.000000+0000", "flow_id": 42, "src_port": 60000}
    assert StateExtractor(SECOPS).extract(Observation(kind="suricata_eve", data=eve2)).canonical == res.canonical


def test_syslog_extraction():
    line = "Sep 26 12:00:01 bastion sshd[4242]: Failed password for root from 203.0.113.77 port 51122 ssh2"
    st = StateExtractor(SECOPS).extract(Observation(kind="syslog", data=line)).state
    assert st["source"] == "syslog" and st["program"] == "sshd" and st["src_ip"] == "203.0.113.77"


def test_canonical_order_and_prefix_tracker():
    s = canonical_state({"temporal_buffer": ["A"], "url": "u", "viewport_tree": [{"role": "x", "id": 1}]}, COMPUTER_USE.prompt_key_order)
    assert s == '{"url":"u","viewport_tree":[{"id":1,"role":"x"}],"temporal_buffer":["A"]}'
    t = PrefixTracker()
    assert t.observe("s", "abcdef") == 0.0
    assert t.observe("s", "abcxyz") == 0.5


def test_format_action():
    assert format_action({"action": "CLICK", "target_id": 12}) == "CLICK(12)"
    assert format_action({"action": "TYPE", "target_id": 2, "text": "admin"}) == "TYPE(2, 'admin')"
    assert format_action({"action": "CLICK_XY", "coordinates": {"x": 450, "y": 820}}) == "CLICK_XY(450, 820)"


def test_contract_validation():
    st = {"url": "u", "viewport_tree": [{"id": 3, "role": "textbox", "name": "Password"}], "temporal_buffer": []}
    ok = COMPUTER_USE.validate_action({"confidence_score": 0.92, "action": "CLICK_XY", "target_id": None, "coordinates": {"x": 450, "y": 820}}, st)
    assert ok.ok, ok.errors
    bad = COMPUTER_USE.validate_action({"confidence_score": 0.9, "action": "CLICK", "target_id": 99}, st)
    assert bad.ok and bad.hallucinated
    missing = COMPUTER_USE.validate_action({"confidence_score": 0.9, "action": "TYPE", "target_id": 3}, st)
    assert not missing.ok
    sec_state = {"source": "suricata_eve", "src_ip": "192.168.1.150", "payload_snippet": "GET /../../etc/passwd"}
    r = SECOPS.validate_action({"confidence_score": 0.98, "verdict": "SUSPICIOUS", "immediate_action": "DROP_AND_BLACKLIST_IP", "target_ioc": ["192.168.1.150"]}, sec_state)
    assert r.ok and not r.hallucinated
    h = SECOPS.validate_action({"confidence_score": 0.98, "verdict": "MALICIOUS", "immediate_action": "DROP_AND_BLACKLIST_IP", "target_ioc": ["8.8.8.8"]}, sec_state)
    assert h.hallucinated
    assert not SECOPS.validate_action({"confidence_score": 0.5, "verdict": "BENIGN", "immediate_action": "REBOOT", "target_ioc": []}).ok
