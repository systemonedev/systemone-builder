"""Distillation datasets on the shared workspace volume.

Layout (``<workspace>/datasets/<domain>/``)::

    sft.jsonl       (state -> CoT -> action) samples, one JSON object per line
    dpo.jsonl       (state, chosen, rejected, delta) preference pairs (Phase 5)
    heldout.jsonl   real, unseen samples for the evaluation sandbox (Phase 5)

Redis tracks dedup keys and counters so the factory, the DPO loop and the
auto-training trigger never have to scan the files.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field

from systemone_builder.datastore.store import JsonStore
from systemone_builder.domains.spec import DomainSpec
from systemone_builder.extraction.prompt import PromptBuilder, _sort

Split = Literal["sft", "dpo", "heldout"]


class SFTSample(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    domain: str
    state: dict[str, Any]
    cot: str | None = None
    action: dict[str, Any]
    source: str  # oracle | triage | human | synthetic | seed
    judge_score: float | None = None
    replay_seq: int | None = None
    created_at: float = Field(default_factory=time.time)


class DPOPair(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    domain: str
    state: dict[str, Any]
    rejected: dict[str, Any]
    chosen: dict[str, Any]
    delta: dict[str, Any] | None = None
    cot: str | None = None
    source: str = "oracle"  # oracle | human
    replay_seq: int | None = None
    created_at: float = Field(default_factory=time.time)


class HeldOutSample(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    domain: str
    state: dict[str, Any]
    expected: dict[str, Any]
    # Alternative correct actions (e.g. two buttons that both submit).
    acceptable: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def sample_key(state: dict[str, Any], action: dict[str, Any] | None = None) -> str:
    payload = json.dumps(_sort({"s": state, "a": {k: v for k, v in (action or {}).items() if k != "confidence_score"}}))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


class DatasetStore:
    def __init__(self, root: Path, store: JsonStore) -> None:
        self.root = root
        self.store = store
        self._locks: dict[str, asyncio.Lock] = {}

    def path(self, domain: str, split: Split) -> Path:
        return self.root / domain / f"{split}.jsonl"

    def _lock(self, domain: str, split: str) -> asyncio.Lock:
        return self._locks.setdefault(f"{domain}:{split}", asyncio.Lock())

    async def _append(self, domain: str, split: Split, obj: BaseModel, dedup: str | None) -> bool:
        if dedup is not None:
            added = await self.store.r.sadd(self.store.key("dataset", domain, split, "keys"), dedup)
            if not added:
                return False
        p = self.path(domain, split)
        async with self._lock(domain, split):
            p.parent.mkdir(parents=True, exist_ok=True)
            line = obj.model_dump_json() + "\n"
            await asyncio.to_thread(_append_line, p, line)
        await self.store.incr("dataset", domain, split, "count")
        return True

    @staticmethod
    def dedup_key(split: Split, row: dict[str, Any]) -> str:
        if split == "sft":
            return sample_key(row["state"], row["action"])
        if split == "dpo":
            return sample_key(row["state"], row["chosen"]) + sample_key(row["state"], row["rejected"])[:6]
        return sample_key(row["state"])

    async def add_sft(self, s: SFTSample) -> bool:
        return await self._append(s.domain, "sft", s, self.dedup_key("sft", s.model_dump()))

    async def add_dpo(self, p: DPOPair) -> bool:
        return await self._append(p.domain, "dpo", p, self.dedup_key("dpo", p.model_dump()))

    async def add_heldout(self, h: HeldOutSample) -> bool:
        return await self._append(h.domain, "heldout", h, self.dedup_key("heldout", h.model_dump()))

    def domains_on_disk(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []

    async def reconcile(self, domain: str) -> dict[str, dict[str, int]]:
        """Rebuild the Redis counters and dedup keys from the JSONL files.

        The files are the source of truth. Redis can drift from them when the
        workspace volume changes, files are edited or deleted by hand, or Redis
        persisted state from an older workspace. Without this, the counters
        report samples that no longer exist and the dedup keys reject
        re-imports of them.
        """
        report: dict[str, dict[str, int]] = {}
        for split in ("sft", "dpo", "heldout"):
            keys = [self.dedup_key(split, row) for row in self.iter(domain, split)]  # type: ignore[arg-type]
            before = await self.count(domain, split)  # type: ignore[arg-type]
            k_keys = self.store.key("dataset", domain, split, "keys")
            pipe = self.store.r.pipeline()
            pipe.delete(k_keys)
            if keys:
                pipe.sadd(k_keys, *keys)
            pipe.set(self.store.key("dataset", domain, split, "count"), len(keys))
            await pipe.execute()
            report[split] = {"before": before, "after": len(keys)}
        trained = await self.store.get("dataset", domain, "trained_upto", default=0)
        if trained > report["sft"]["after"]:
            await self.mark_trained(domain, report["sft"]["after"])
        dpo_trained = await self.store.get("dataset", domain, "dpo_trained_upto", default=0)
        if dpo_trained > report["dpo"]["after"]:
            await self.store.set(report["dpo"]["after"], "dataset", domain, "dpo_trained_upto")
        return report

    async def count(self, domain: str, split: Split) -> int:
        return await self.store.counter("dataset", domain, split, "count")

    def iter(self, domain: str, split: Split) -> Iterator[dict[str, Any]]:
        p = self.path(domain, split)
        if not p.exists():
            return
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def tail(self, domain: str, split: Split, n: int = 50) -> list[dict[str, Any]]:
        items = list(self.iter(domain, split))
        return items[-n:]

    async def stats(self, domain: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for split in ("sft", "dpo", "heldout"):
            out[split] = await self.count(domain, split)  # type: ignore[arg-type]
        out["trained_upto"] = await self.store.get("dataset", domain, "trained_upto", default=0)
        out["new_since_train"] = max(out["sft"] - out["trained_upto"], 0)
        return out

    async def mark_trained(self, domain: str, count: int) -> None:
        await self.store.set(count, "dataset", domain, "trained_upto")

    # ------------------------------------------------------------ exports
    def export_sft(self, domain: DomainSpec, out: Path, min_judge: float | None = None) -> int:
        """Conversational prompt/completion rows for TRL's SFTTrainer.

        The student is a System-1 model: it learns to emit the action JSON
        directly from the (static prefix + state) prompt. The teacher's CoT is
        kept in the dataset for audit and for users who opt into rationale
        distillation, but is not part of the completion.
        """
        pb = PromptBuilder(domain)
        n = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for s in self.iter(domain.id, "sft"):
                if min_judge is not None and s.get("judge_score") is not None and s["judge_score"] < min_judge:
                    continue
                messages, _ = pb.messages(s["state"])
                action = _training_action(s["action"])
                if s.get("judge_score") is not None and "confidence_score" in action:
                    # Supervise confidence with the judge's verdict so the
                    # student's self-report stays calibrated.
                    action["confidence_score"] = round(min(float(action["confidence_score"]), float(s["judge_score"])), 2)
                completion = json.dumps(_sort(action), separators=(",", ":"))
                f.write(json.dumps({"prompt": messages, "completion": [{"role": "assistant", "content": completion}]}) + "\n")
                n += 1
        return n

    def export_dpo(self, domain: DomainSpec, out: Path) -> int:
        pb = PromptBuilder(domain)
        n = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for p in self.iter(domain.id, "dpo"):
                messages, _ = pb.messages(p["state"])
                row = {
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": json.dumps(_sort(_training_action(p["chosen"])), separators=(",", ":"))}],
                    "rejected": [{"role": "assistant", "content": json.dumps(_sort(_training_action(p["rejected"])), separators=(",", ":"))}],
                }
                f.write(json.dumps(row) + "\n")
                n += 1
        return n


def _training_action(action: dict[str, Any]) -> dict[str, Any]:
    # supported_actions is part of the prompt contract; don't train the model
    # to regurgitate it on every step (it costs tokens on the hot path).
    return {k: v for k, v in action.items() if k != "supported_actions" and v is not None}


def _append_line(path: Path, line: str) -> None:
    with path.open("a") as f:
        f.write(line)
