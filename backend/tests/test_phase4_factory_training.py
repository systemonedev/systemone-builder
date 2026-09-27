from __future__ import annotations

import json

from systemone.config import Settings
from systemone.datastore.store import JsonStore
from systemone.domains.builtin import SECOPS
from systemone.domains.registry import DomainRegistry
from systemone.factory.dataset import DatasetStore, DPOPair, SFTSample
from systemone.training.lifecycle import StudentLifecycleOrchestrator

STATE = {"source": "suricata_eve", "src_ip": "192.168.1.150", "payload_snippet": "GET /../../../../etc/passwd HTTP/1.1"}
ACTION = {"confidence_score": 0.99, "verdict": "SUSPICIOUS", "immediate_action": "DROP_AND_BLACKLIST_IP",
          "target_ioc": ["192.168.1.150"], "supported_actions": ["DROP_AND_BLACKLIST_IP", "ESCALATE", "ALLOW"]}


async def test_dataset_dedup_counts_and_sft_export(redis_client, tmp_path):
    ds = DatasetStore(tmp_path / "datasets", JsonStore(redis_client, "t"))
    assert await ds.add_sft(SFTSample(domain="secops", state=STATE, action=ACTION, cot="traversal", source="oracle", judge_score=0.8))
    assert not await ds.add_sft(SFTSample(domain="secops", state=STATE, action={**ACTION, "confidence_score": 0.5}, source="oracle"))
    stats = await ds.stats("secops")
    assert stats["sft"] == 1 and stats["new_since_train"] == 1
    n = ds.export_sft(SECOPS, tmp_path / "train.jsonl")
    row = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert n == 1 and row["prompt"][0]["role"] == "system" and row["prompt"][-1]["role"] == "user"
    completion = json.loads(row["completion"][0]["content"])
    # judge caps the confidence target; supported_actions is not trained
    assert completion["confidence_score"] == 0.8 and "supported_actions" not in completion
    await ds.mark_trained("secops", 1)
    assert (await ds.stats("secops"))["new_since_train"] == 0


async def test_dpo_export(redis_client, tmp_path):
    ds = DatasetStore(tmp_path / "datasets", JsonStore(redis_client, "t"))
    rejected = {**ACTION, "verdict": "BENIGN", "immediate_action": "ALLOW", "target_ioc": []}
    await ds.add_dpo(DPOPair(domain="secops", state=STATE, rejected=rejected, chosen=ACTION, delta={"new_alerts": 3}))
    assert ds.export_dpo(SECOPS, tmp_path / "dpo.jsonl") == 1
    row = json.loads((tmp_path / "dpo.jsonl").read_text())
    assert json.loads(row["chosen"][0]["content"])["immediate_action"] == "DROP_AND_BLACKLIST_IP"
    assert json.loads(row["rejected"][0]["content"])["immediate_action"] == "ALLOW"


async def test_prepare_run_writes_container_paths(redis_client, tmp_path):
    s = Settings(data_dir=tmp_path / "s1", workspace_container_path="/workspace")
    s.ensure_dirs()
    store = JsonStore(redis_client, "t")
    ds = DatasetStore(s.workspace / "datasets", store)
    await ds.add_sft(SFTSample(domain="secops", state=STATE, action=ACTION, source="oracle"))
    lc = StudentLifecycleOrchestrator(s, gpu=None, docker=None, store=store, datasets=ds, domains=DomainRegistry(store), bus=None)
    cfg = lc.prepare_run("secops", "sft")
    assert cfg["train_file"].startswith("/workspace/runs/secops-sft-") and cfg["train_file"].endswith("/train.jsonl")
    assert cfg["rows"] == 1 and cfg["init_from"] is None and cfg["base_model"] == s.student_base_model
    assert (s.workspace / "runs" / cfg["run_id"] / "config.json").exists()
    assert lc.from_container(cfg["output_dir"]) == s.workspace / "runs" / cfg["run_id"]
    assert lc.read_pointer()["model"] == s.student_base_model
    # lifecycle gate
    assert lc.acquire() and lc.inflight == 1
    lc.release()
    assert lc.inflight == 0
