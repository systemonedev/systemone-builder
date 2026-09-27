from __future__ import annotations

from systemone.domains.builtin import SECOPS
from systemone.telemetry.collector import parse_prometheus
from systemone.workflow.engine import WorkflowEngine, classify, slugify

PROM = """# HELP vllm:prefix_cache_hits_total hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="student"} 900.0
vllm:prefix_cache_queries_total{engine="0",model_name="student"} 1000.0
vllm:kv_cache_usage_perc{engine="0",model_name="student"} 0.25
vllm:num_requests_running{engine="0",model_name="student"} 3
"""


def test_parse_prometheus():
    m = parse_prometheus(PROM)
    assert m["vllm:prefix_cache_hits_total"] == 900 and m["vllm:num_requests_running"] == 3


def test_classify_and_slug():
    assert classify("block SSH brute force attempts seen in auth logs") == "secops"
    assert classify("click through the web form and submit the invoice") == "computer_use"
    assert classify("summarize quarterly spreadsheets") == "custom"
    assert slugify("SSH Brute-Force Reflex!") == "ssh_brute_force_reflex"


CUSTOM = {
    "id": "Ticket Router",
    "name": "Ticket router",
    "description": "Route support tickets",
    "kind": "custom",
    "supported_actions": ["ROUTE_BILLING", "ROUTE_TECH"],
    "action_field": "route",
    "state_schema": {"type": "object", "required": ["subject"], "properties": {"subject": {"type": "string"}}},
    "action_schema": {"type": "object", "required": ["confidence_score", "route"],
                      "properties": {"confidence_score": {"type": "number"}, "route": {"type": "string"}}},
    "threshold": 1.4,
    "system_prompt": "Route the ticket.",
    "few_shots": [{"state": {"subject": "refund please"}, "action": {"confidence_score": 0.9, "route": "ROUTE_BILLING"}}],
    "prompt_key_order": ["subject"],
    "extractor": {"type": "passthrough", "custom_patterns": [["TCK-\\\\d+", "<TICKET>"], ["bad"]]},
    "factory": {"scenarios": ["refunds", "outages", "password resets"]},
}


def test_workflow_validation_normalizes_custom_spec():
    eng = WorkflowEngine(teacher=None, store=None)
    assert eng._validate(CUSTOM, None) == []
    norm = eng._normalize(CUSTOM, None)
    assert norm["id"] == "ticket_router" and "ESCALATE" in norm["supported_actions"]
    assert norm["threshold"] == 0.99 and norm["extractor"]["custom_patterns"] == [["TCK-\\\\d+", "<TICKET>"]]


def test_workflow_validation_reports_errors_for_repair():
    eng = WorkflowEngine(teacher=None, store=None)
    bad = {**CUSTOM, "action_schema": {"type": "object", "properties": {"route": {"type": "string"}}},
           "few_shots": [{"state": {}, "action": {"route": "NOPE"}}], "factory": {"scenarios": []}}
    errs = eng._validate(bad, None)
    assert any("confidence_score" in e for e in errs)
    assert any("few_shots[0].state" in e for e in errs)
    assert any("scenarios" in e for e in errs)


def test_workflow_template_keeps_contract():
    eng = WorkflowEngine(teacher=None, store=None)
    raw = SECOPS.model_dump(mode="json", exclude={"source"})
    raw.update(id="ssh_bruteforce", action_field="something_else", extractor={"type": "passthrough", "tokenize_ips": True})
    norm = eng._normalize(raw, SECOPS)
    assert norm["action_field"] == "immediate_action" and norm["extractor"]["type"] == "log"
    assert eng._validate(raw, SECOPS) == []
