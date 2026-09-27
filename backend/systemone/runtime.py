"""Runtime container: builds and owns every long-lived component.

The FastAPI app creates exactly one :class:`Runtime` in its lifespan and the
route handlers reach components through ``request.app.state.rt``.
"""

from __future__ import annotations

import logging
from typing import Any

from systemone.config import Settings
from systemone.datastore.replay_buffer import RedisReplayBuffer
from systemone.datastore.store import JsonStore, create_redis
from systemone.domains.registry import DomainRegistry
from systemone.extraction.pipeline import StateExtractor
from systemone.extraction.prompt import PrefixTracker
from systemone.orchestrator.docker_ctl import DockerController
from systemone.orchestrator.gpu import create_gpu_monitor
from systemone.orchestrator.hardware import HardwareOrchestrator
from systemone.telemetry.bus import EventBus

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.bus = EventBus()

        self.redis = create_redis(settings.redis_url)
        self.store = JsonStore(self.redis, settings.redis_namespace)
        self.replay = RedisReplayBuffer(self.redis, settings.replay_capacity, settings.redis_namespace)

        roles = {settings.student_gpu: "student", settings.triage_gpu: "triage"}
        self.gpu = create_gpu_monitor(roles)
        self.docker = DockerController()
        self.hardware = HardwareOrchestrator(settings, self.gpu, self.docker)

        self.domains = DomainRegistry(self.store, settings.data_dir / "domains")
        self.prefix = PrefixTracker()
        self._extractors: dict[str, tuple[int, StateExtractor]] = {}

    def extractor(self, domain_id: str) -> StateExtractor:
        """Cached per-domain extractor, rebuilt when the domain spec changes."""
        spec = self.domains.get(domain_id)
        key = hash(spec.model_dump_json())
        cached = self._extractors.get(domain_id)
        if cached is None or cached[0] != key:
            cached = (key, StateExtractor(spec))
            self._extractors[domain_id] = cached
        return cached[1]

    async def startup(self) -> None:
        await self.redis.ping()
        await self.domains.load()

    async def shutdown(self) -> None:
        await self.redis.aclose()
        self.gpu.close()

    async def health(self) -> dict[str, Any]:
        try:
            redis_ok = bool(await self.redis.ping())
        except Exception:
            redis_ok = False
        return {"redis": redis_ok}
