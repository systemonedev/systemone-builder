from __future__ import annotations

import asyncio
import json

import pytest

from systemone.adapters.byom import AdapterHandle, EndpointConfig, resolve, validate_hf_model
from systemone.adapters.factory import build_adapter, load_plugins
from systemone.config import Settings
from systemone.extraction.pipeline import Observation, StateExtractor
from systemone.extraction.prompt import PromptBuilder
from systemone.quickstart.webgen import generate
from systemone.templates import all_templates, list_templates


@pytest.mark.parametrize("tid", sorted(all_templates()))
def test_default_domains_and_templates_are_self_consistent(tid):
    spec = all_templates()[tid]
    assert "ESCALATE" in spec.supported_actions
    assert spec.validate_action(spec.escalate_action(0.1)).ok
    for shot in spec.few_shots:
        assert spec.validate_state(shot["state"]) == []
        v = spec.validate_action(shot["action"], shot["state"])
        assert v.ok and not v.hallucinated, v.errors + v.grounding_errors
    # extractor + prompt build without errors, and the prompt embeds the contract
    ex = StateExtractor(spec)
    res = ex.extract(Observation(kind="state", data=spec.few_shots[0]["state"]))
    msgs, _ = PromptBuilder(spec).messages(res.state)
    assert msgs[0]["role"] == "system" and "ESCALATE" in msgs[0]["content"]


def test_template_listing():
    ids = {t["id"] for t in list_templates()}
    assert {"computer_use", "secops", "desktop_vision", "auth_log_bruteforce"} <= ids


def test_auth_template_scrubs_ephemeral_port():
    spec = all_templates()["auth_log_bruteforce"]
    line = "Sep 26 12:00:01 bastion sshd[4242]: Failed password for root from 203.0.113.77 port 51122 ssh2"
    st = StateExtractor(spec).extract(Observation(kind="log", data=line)).state
    assert st["src_ip"] == "203.0.113.77" and "port <EPHEMERAL>" in st["payload_snippet"]


def test_quickstart_dataset_is_valid_diverse_and_deterministic():
    a = generate(300, seed=7)
    assert a == generate(300, seed=7)
    labels = {r["action"]["action"] for r in a}
    assert labels == {"CLICK", "TYPE", "CLICK_XY", "ESCALATE"}
    assert len({json.dumps(r["state"], sort_keys=True) for r in a}) > 250
    # volatile noise never reaches the state
    blob = json.dumps(a)
    assert "ember" not in blob and "ts=1" not in blob


def test_byom_layering(tmp_path, monkeypatch):
    f = tmp_path / "systemone.yaml"
    f.write_text("oracle:\n  url: http://mac:11434\n  model: llama3.3:70b\n  api_key_env: ORACLE_KEY\nstudent:\n  base_model: meta-llama/Llama-3.2-1B-Instruct\n")
    s = Settings(byom_file=str(f))
    cfg = resolve(s, {"triage": {"model": "Qwen/Qwen2.5-7B-Instruct"}})
    assert cfg["oracle"].url == "http://mac:11434" and cfg["oracle"].adapter == "ollama"
    assert cfg["student"].base_model == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg["triage"].model == "Qwen/Qwen2.5-7B-Instruct"
    monkeypatch.setenv("ORACLE_KEY", "k")
    assert cfg["oracle"].api_key() == "k"


def test_adapter_registry_and_handle_swap():
    assert {"openai", "ollama"} <= set(load_plugins())
    with pytest.raises(ValueError):
        build_adapter("nope", "http://x", "m", 1)
    a = build_adapter("openai", "http://a/v1", "m1", 1)
    h = AdapterHandle("triage", a, EndpointConfig(adapter="openai", url="http://a/v1", model="m1"))
    assert h.model == "m1" and h.describe()["role"] == "triage"
    b = build_adapter("ollama", "http://b", "m2", 1)
    asyncio.run(h.swap(b, EndpointConfig(adapter="ollama", url="http://b", model="m2")))
    assert h.kind == "ollama" and h.model == "m2"


def test_validate_local_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}))
    assert asyncio.run(validate_hf_model(str(tmp_path)))["ok"]
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "t5"}))
    assert not asyncio.run(validate_hf_model(str(tmp_path)))["ok"]
