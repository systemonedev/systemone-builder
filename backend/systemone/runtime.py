"""Runtime container: builds and owns every long-lived component.

The FastAPI app creates exactly one :class:`Runtime` in its lifespan and the
route handlers reach components through ``request.app.state.rt``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from systemone.adapters.factory import build_adapter
from systemone.config import Settings
from systemone.datastore.replay_buffer import RedisReplayBuffer
from systemone.datastore.store import JsonStore, create_redis
from systemone.domains.registry import DomainRegistry
from systemone.extraction.pipeline import StateExtractor
from systemone.extraction.prompt import PrefixTracker
from systemone.orchestrator.docker_ctl import DockerController
from systemone.orchestrator.gpu import create_gpu_monitor
from systemone.orchestrator.hardware import HardwareOrchestrator
from systemone.routing.confidence import ConfidenceScorer
from systemone.routing.oracle import OracleEscalationService
from systemone.dpo.loop import DPOLoop
from systemone.evaluation.sandbox import EvaluationSandbox
from systemone.factory.dataset import DatasetStore
from systemone.factory.synthetic import SyntheticFactory, VisionParser
from systemone.routing.router import FastSlowRouter
from systemone.training.lifecycle import LifecycleError, StudentLifecycleOrchestrator
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

        # ---- BYOM model endpoints
        self.student = build_adapter(settings.student_adapter, settings.student_url, settings.student_served_name, settings.request_timeout_s)
        self.triage = build_adapter(settings.triage_adapter, settings.triage_url, settings.triage_model, settings.request_timeout_s)
        self.oracle = build_adapter(settings.oracle_adapter, settings.oracle_url, settings.oracle_model, settings.oracle_timeout_s)

        # ---- datasets, lifecycle, synthetic factory
        self.datasets = DatasetStore(settings.workspace / "datasets", self.store)
        self.lifecycle = StudentLifecycleOrchestrator(
            settings, self.gpu, self.docker, self.store, self.datasets, self.domains, self.bus
        )
        self.factory = SyntheticFactory(
            self.oracle, self.replay, self.store, self.datasets, self.domains, self.bus,
            batch_size=settings.factory_batch_size, poll_interval_s=settings.factory_poll_interval_s,
            use_judge=settings.factory_judge, on_new_samples=self._maybe_auto_train,
        )
        self.vision = VisionParser(self.oracle, settings.oracle_vision_model) if settings.oracle_vision_model else None

        # ---- Fast-Slow routing
        self.scorer = ConfidenceScorer(settings.confidence_logprob_weight)
        self.escalations = OracleEscalationService(
            self.oracle, self.store, self.replay, self.domains, self.bus, concurrency=settings.oracle_concurrency
        )
        self.router = FastSlowRouter(
            student=self.student,
            triage=self.triage,
            oracle=self.escalations,
            scorer=self.scorer,
            replay=self.replay,
            store=self.store,
            bus=self.bus,
            prefix=self.prefix,
            extractor_for=self.extractor,
            domain_for=self.domains.get,
            student_model_name=self.student_model_name,
            student_max_tokens=settings.student_max_tokens,
            student_gate=self.lifecycle,
            vision=self.vision,
        )
        self._build_phase5()

    def _build_phase5(self) -> None:
        s = self.settings
        self.dpo = DPOLoop(
            self.replay, self.store, self.datasets, self.domains, self.factory, self.bus, self.extractor,
            auto_approve_min_judge=s.dpo_auto_approve_min_judge if s.dpo_auto_approve_min_judge <= 1 else None,
            on_new_pairs=self._maybe_auto_dpo,
        )
        self.evaluation = EvaluationSandbox(
            {"student": self.student, "triage": self.triage, "oracle": self.oracle},
            self.datasets, self.domains, self.store, self.bus, self.scorer,
            s.data_dir / "eval_results", self.student_model_name, s.student_max_tokens,
        )

    async def _maybe_auto_dpo(self, domain_id: str) -> None:
        if not self.settings.auto_train or self.lifecycle.busy:
            return
        pairs = await self.datasets.count(domain_id, "dpo")
        done = await self.store.get("dataset", domain_id, "dpo_trained_upto", default=0)
        if pairs - done >= self.settings.dpo_auto_train_min_pairs:
            try:
                cfg = self.lifecycle.start_cycle(domain_id, "dpo")
                await self.store.set(pairs, "dataset", domain_id, "dpo_trained_upto")
                self.bus.publish("training", "auto_triggered", domain=domain_id, run_id=cfg["run_id"], rows=cfg["rows"], mode="dpo")
            except LifecycleError as exc:
                log.warning("auto-DPO skipped: %s", exc)

    async def _maybe_auto_train(self, domain_ids: set[str]) -> None:
        if not self.settings.auto_train or self.lifecycle.busy:
            return
        for did in sorted(domain_ids):
            stats = await self.datasets.stats(did)
            if stats["new_since_train"] >= self.settings.auto_train_min_samples:
                try:
                    cfg = self.lifecycle.start_cycle(did, "sft")
                    self.bus.publish("training", "auto_triggered", domain=did, run_id=cfg["run_id"], rows=cfg["rows"])
                except LifecycleError as exc:
                    log.warning("auto-train skipped: %s", exc)
                return

    async def student_model_name(self) -> str:
        """Model id the student vLLM currently serves (base, LoRA or merged)."""
        return await self.store.get("lifecycle", "served_model", default=self.settings.student_served_name)

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
        await self._load_calibrations()
        self.escalations.start()
        await self._restore_lifecycle()
        if self.settings.auto_factory:
            self.factory.start()

    async def _restore_lifecycle(self) -> None:
        """After an API restart mid-cycle, make sure the student is serving."""
        state = await self.store.get("lifecycle", "state", default={})
        if state.get("phase") not in (None, "serving"):
            log.warning("lifecycle was %s at shutdown; recovering student", state.get("phase"))
            asyncio.create_task(self.lifecycle.recover())
            return
        # Container B has no restart policy (the orchestrator owns it).
        info = await self.docker.status(self.settings.student_container)
        if info.status in ("exited", "created"):
            log.info("starting student container %s", self.settings.student_container)
            await self.docker.start(self.settings.student_container)

    async def _load_calibrations(self) -> None:
        from systemone.routing.confidence import Calibration

        for did, c in (await self.store.hgetall(("routing", "calibration"))).items():
            self.scorer.calibrations[did] = Calibration(**c)

    async def shutdown(self) -> None:
        await self.factory.stop()
        await self.escalations.stop()
        for a in (self.student, self.triage, self.oracle):
            await a.aclose()
        await self.redis.aclose()
        self.gpu.close()

    async def health(self) -> dict[str, Any]:
        try:
            redis_ok = bool(await self.redis.ping())
        except Exception:
            redis_ok = False
        return {"redis": redis_ok}

    async def deep_health(self) -> dict[str, Any]:
        return {
            **await self.health(),
            "student": await self.student.health(),
            "triage": await self.triage.health(),
            "oracle": await self.oracle.health(),
        }
