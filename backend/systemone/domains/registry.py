"""Workspace domain registry: built-ins + generated / user domains in Redis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from systemone.datastore.store import JsonStore
from systemone.domains.builtin import BUILTIN_DOMAINS
from systemone.domains.spec import DomainSpec


class DomainRegistry:
    def __init__(self, store: JsonStore, export_dir: Path | None = None) -> None:
        self.store = store
        self.export_dir = export_dir
        self._cache: dict[str, DomainSpec] = dict(BUILTIN_DOMAINS)

    async def load(self) -> None:
        for did, raw in (await self.store.hgetall(("domains",))).items():
            self._cache[did] = DomainSpec.model_validate(raw)

    def get(self, domain_id: str) -> DomainSpec:
        try:
            return self._cache[domain_id]
        except KeyError:
            raise KeyError(f"unknown domain {domain_id!r}") from None

    def all(self) -> list[DomainSpec]:
        return list(self._cache.values())

    async def put(self, spec: DomainSpec) -> DomainSpec:
        self._cache[spec.id] = spec
        await self.store.hset(("domains",), spec.id, spec.model_dump(mode="json"))
        if self.export_dir is not None:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            (self.export_dir / f"{spec.id}.json").write_text(json.dumps(spec.model_dump(mode="json"), indent=2))
        return spec

    async def delete(self, domain_id: str) -> None:
        if domain_id in BUILTIN_DOMAINS and self._cache.get(domain_id) is BUILTIN_DOMAINS[domain_id]:
            raise ValueError("built-in domains cannot be deleted")
        self._cache.pop(domain_id, None)
        await self.store.hdel(("domains",), domain_id)
        if domain_id in BUILTIN_DOMAINS:  # restore pristine built-in after an override
            self._cache[domain_id] = BUILTIN_DOMAINS[domain_id]

    async def update(self, domain_id: str, **changes: Any) -> DomainSpec:
        spec = self.get(domain_id)
        merged = spec.model_copy(update=changes)
        spec = DomainSpec.model_validate(merged.model_dump())
        return await self.put(spec)
