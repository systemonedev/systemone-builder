"""Strict GPU 0 lifecycle orchestrator (Module C, Phase 4).

A single 24GB RTX 3090 cannot hold the vLLM student (~20GB with KV cache)
and an Unsloth QLoRA run at the same time. Every training cycle therefore
walks a strict state machine, never skipping a step::

    SERVING -> DRAINING -> PAUSED -> FLUSHING -> TRAINING -> FLUSHING
            -> RELOADING -> SERVING

* DRAINING  - the router stops sending traffic to the student (requests are
              served by the GPU 1 triage tier) and in-flight requests finish.
* PAUSED    - Container B (vLLM student) is stopped.
* FLUSHING  - NVML must report GPU 0 below ``vram_flush_threshold_mb``;
              otherwise the cycle aborts instead of risking an OOM.
* TRAINING  - Container A (Unsloth) runs one-shot on GPU 0 over the exported
              replay-distilled dataset; loss is streamed to the dashboard.
* RELOADING - the new merged weights (or LoRA adapter) are written to the
              student pointer file and Container B is restarted and health
              checked before traffic returns.

Any failure rolls back to the previous weights and restarts the student.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from systemone.config import Settings
from systemone.datastore.store import JsonStore
from systemone.domains.registry import DomainRegistry
from systemone.factory.dataset import DatasetStore
from systemone.orchestrator.docker_ctl import ContainerController, RunSpec
from systemone.orchestrator.gpu import GpuMonitor
from systemone.telemetry.bus import EventBus

log = logging.getLogger(__name__)


class Phase(str, Enum):
    SERVING = "serving"
    DRAINING = "draining"
    PAUSED = "paused"
    FLUSHING = "flushing"
    TRAINING = "training"
    RELOADING = "reloading"
    FAILED = "failed"


class LifecycleError(RuntimeError):
    pass


class StudentLifecycleOrchestrator:
    def __init__(
        self,
        settings: Settings,
        gpu: GpuMonitor,
        docker: ContainerController,
        store: JsonStore,
        datasets: DatasetStore,
        domains: DomainRegistry,
        bus: EventBus,
    ) -> None:
        self.s = settings
        self.gpu = gpu
        self.docker = docker
        self.store = store
        self.datasets = datasets
        self.domains = domains
        self.bus = bus
        self.phase = Phase.SERVING
        self._lock = asyncio.Lock()
        self.inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self.current_run: str | None = None
        self._task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------ paths
    @property
    def ws(self) -> Path:
        return self.s.workspace

    def to_container(self, p: Path) -> str:
        """Translate an API-side workspace path to the path seen by A/B."""
        rel = p.resolve().relative_to(self.ws.resolve())
        return f"{self.s.workspace_container_path.rstrip('/')}/{rel.as_posix()}"

    def from_container(self, p: str) -> Path:
        prefix = self.s.workspace_container_path.rstrip("/") + "/"
        return self.ws / p[len(prefix):] if p.startswith(prefix) else Path(p)

    @property
    def pointer_file(self) -> Path:
        return self.ws / "state" / "student.json"

    def read_pointer(self) -> dict[str, Any]:
        if self.pointer_file.exists():
            return json.loads(self.pointer_file.read_text())
        return {"model": self.s.student_base_model, "served_name": self.s.student_served_name, "lora": None}

    def _write_pointer(self, ptr: dict[str, Any]) -> None:
        self.pointer_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(ptr, indent=2))
        tmp.replace(self.pointer_file)

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ------------------------------------------------- router integration
    @property
    def student_available(self) -> bool:
        return self.phase == Phase.SERVING

    def acquire(self) -> bool:
        """Router calls before a student request; False => skip the student."""
        if not self.student_available:
            return False
        self.inflight += 1
        self._idle.clear()
        return True

    def release(self) -> None:
        self.inflight = max(self.inflight - 1, 0)
        if self.inflight == 0:
            self._idle.set()

    # --------------------------------------------------------- state
    async def _set_phase(self, phase: Phase, **info: Any) -> None:
        self.phase = phase
        await self.store.set({"phase": phase.value, "run_id": self.current_run, "ts": time.time(), **info}, "lifecycle", "state")
        self.bus.publish("lifecycle", "phase", phase=phase.value, run_id=self.current_run, **info)
        log.info("lifecycle -> %s %s", phase.value, info or "")

    async def status(self) -> dict[str, Any]:
        st = await self.store.get("lifecycle", "state", default={"phase": self.phase.value})
        gpu0 = self.gpu.get(self.s.student_gpu)
        return {
            **st,
            "phase": self.phase.value,
            "inflight": self.inflight,
            "pointer": self.read_pointer(),
            "served_model": await self.store.get("lifecycle", "served_model", default=self.s.student_served_name),
            "gpu0": gpu0.to_dict() if gpu0 else None,
            "busy": self.busy,
        }

    async def _flush(self, stage: str) -> Any:
        """wait_for_flush with a diagnosis of who holds GPU 0 on failure."""
        s = self.s
        try:
            return await self.gpu.wait_for_flush(s.student_gpu, s.vram_flush_threshold_mb, s.vram_flush_timeout_s,
                                                 s.vram_required_free_mb)
        except TimeoutError as exc:
            from systemone.orchestrator.gpu import placement_warnings

            gpus = self.gpu.snapshot()
            usage = ", ".join(f"GPU {g.index}: {g.memory_used_mb}/{g.memory_total_mb} MiB" for g in gpus)
            triage = await self.docker.status(s.triage_container)
            hints = placement_warnings(gpus, {"triage": s.triage_gpu}, {"triage"} if triage.status == "running" else set())
            raise LifecycleError(f"{stage}: {exc} [{usage}] {' '.join(hints)}".strip()) from exc

    # ------------------------------------------------------------ runs
    async def _save_run(self, run: dict[str, Any]) -> None:
        await self.store.hset(("training", "runs"), run["run_id"], run)

    async def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list((await self.store.hgetall(("training", "runs"))).values())
        items.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return items[:limit]

    async def run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return await self.store.peek(("training", "metrics", run_id))

    def prepare_run(self, domain_id: str, mode: str) -> dict[str, Any]:
        domain = self.domains.get(domain_id)
        run_id = f"{domain_id}-{mode}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        run_dir = self.ws / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        train_file = run_dir / "train.jsonl"
        rows = self.datasets.export_dpo(domain, train_file) if mode == "dpo" else self.datasets.export_sft(domain, train_file)
        if rows == 0:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise LifecycleError(f"no {mode} samples for domain {domain_id!r}")
        ptr = self.read_pointer()
        # SFT retrains from the base on the cumulative dataset (no drift);
        # DPO refines the currently served weights.
        init_from = None
        if mode == "dpo" and ptr.get("model") and ptr["model"] != self.s.student_base_model:
            init_from = ptr["model"]
        cfg = {
            "run_id": run_id,
            "mode": mode,
            "domain": domain_id,
            "base_model": self.s.student_base_model,
            "init_from": init_from,
            "train_file": self.to_container(train_file),
            "output_dir": self.to_container(run_dir),
            "max_seq_length": self.s.train_max_seq_length,
            "lora_rank": self.s.lora_rank,
            "lora_alpha": self.s.lora_alpha,
            "epochs": self.s.train_epochs,
            "batch_size": self.s.train_batch_size,
            "grad_accum": 4,
            "learning_rate": self.s.train_learning_rate if mode == "sft" else self.s.train_learning_rate / 40,
            "dpo_beta": 0.1,
            "merge": self.s.reload_mode == "merged",
            "seed": 3407,
            "rows": rows,
        }
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        return cfg

    # ----------------------------------------------------------- cycle
    def start_cycle(self, domain_id: str, mode: str = "sft") -> dict[str, Any]:
        """Prepare a run and execute the cycle in the background."""
        if self.busy or (self._task is not None and not self._task.done()):
            raise LifecycleError("a training cycle is already running")
        cfg = self.prepare_run(domain_id, mode)
        self._task = asyncio.create_task(self.run_cycle(cfg), name=f"lifecycle-{cfg['run_id']}")
        return cfg

    async def run_cycle(self, cfg: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self.current_run = cfg["run_id"]
            run = {"run_id": cfg["run_id"], "domain": cfg["domain"], "mode": cfg["mode"], "rows": cfg["rows"],
                   "status": "running", "created_at": time.time(), "phases": []}
            await self._save_run(run)
            previous = self.read_pointer()
            try:
                result = await self._cycle(cfg, run)
                run.update(status="succeeded", result=result, finished_at=time.time())
                if cfg["mode"] == "sft":
                    await self.datasets.mark_trained(cfg["domain"], await self.datasets.count(cfg["domain"], "sft"))
                return run
            except Exception as exc:
                log.exception("training cycle %s failed", cfg["run_id"])
                run.update(status="failed", error=str(exc), finished_at=time.time())
                await self._set_phase(Phase.FAILED, error=str(exc))
                await self._rollback(previous)
                return run
            finally:
                await self._save_run(run)
                self.current_run = None

    async def _mark(self, run: dict[str, Any], phase: Phase, **info: Any) -> None:
        run["phases"].append({"phase": phase.value, "ts": time.time(), **info})
        await self._save_run(run)
        await self._set_phase(phase, **info)

    async def _cycle(self, cfg: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        s = self.s
        # 1. DRAINING -------------------------------------------------------
        await self._mark(run, Phase.DRAINING, inflight=self.inflight)
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=30)
        except asyncio.TimeoutError:
            log.warning("drain timeout with %d in-flight requests; proceeding", self.inflight)

        # 2. PAUSED ---------------------------------------------------------
        await self.docker.stop(s.student_container, timeout_s=30)
        await self._mark(run, Phase.PAUSED)

        # 3. FLUSHING -------------------------------------------------------
        st = await self._flush("before training")
        await self._mark(run, Phase.FLUSHING, vram_used_mb=st.memory_used_mb)

        # 4. TRAINING -------------------------------------------------------
        await self._mark(run, Phase.TRAINING, rows=cfg["rows"])
        metrics_key = ("training", "metrics", cfg["run_id"])
        result_payload: dict[str, Any] = {}
        error_payload: dict[str, Any] = {}
        pending: list[asyncio.Task[Any]] = []

        def handle(line: str) -> None:  # runs on the event loop
            if not line.startswith("S1_METRIC "):
                return
            try:
                m = json.loads(line[len("S1_METRIC "):])
            except json.JSONDecodeError:
                return
            if m.get("kind") == "result":
                result_payload.update(m)
            elif m.get("kind") == "error":
                error_payload.update(m)
            # The trainer's own messages may already carry run_id: merge, don't
            # pass it twice.
            data = {k: v for k, v in m.items() if k != "kind"}
            data["run_id"] = cfg["run_id"]
            self.bus.publish("training", m.get("kind", "log"), **data)
            if m.get("kind") == "log":
                pending.append(asyncio.ensure_future(self.store.push(metrics_key, m)))

        def on_log(line: str) -> None:  # called from the docker log thread
            loop.call_soon_threadsafe(handle, line)

        loop = asyncio.get_running_loop()
        spec = RunSpec(
            name=s.trainer_container,
            image=s.trainer_image,
            command=[f"{cfg['output_dir']}/config.json"],
            gpu=s.student_gpu,
            environment={"PYTHONUNBUFFERED": "1", "HF_HOME": f"{s.workspace_container_path}/hf_cache",
                         "S1_GPU_INDEX": str(s.student_gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                         "NVIDIA_VISIBLE_DEVICES": str(s.student_gpu),
                         **({"HF_TOKEN": _env("HF_TOKEN")} if _env("HF_TOKEN") else {})},
            volumes={s.workspace_host_path: {"bind": s.workspace_container_path, "mode": "rw"}},
            shm_size=s.trainer_shm_size,
            mem_limit=s.trainer_mem_limit,
        )
        # Hard guard: never put the trainer on GPU 0 next to a live student.
        student = await self.docker.status(s.student_container)
        if student.status in ("running", "restarting"):
            raise LifecycleError(f"refusing to start the trainer: {s.student_container} is {student.status}")
        code = await self.docker.run_to_completion(spec, on_log=on_log, timeout_s=6 * 3600)
        await asyncio.sleep(0.1)  # let queued log callbacks run
        await asyncio.gather(*pending, return_exceptions=True)
        result_file = self.ws / "runs" / cfg["run_id"] / "result.json"
        if code != 0 or not result_file.exists():
            raise LifecycleError(f"trainer exited with code {code}: {error_payload.get('error', 'no result.json')}")
        result = json.loads(result_file.read_text())

        # trainer has exited: VRAM must be free again before vLLM returns
        st = await self._flush("after training")
        await self._mark(run, Phase.FLUSHING, vram_used_mb=st.memory_used_mb, after="training")

        # 5. RELOADING ------------------------------------------------------
        if s.reload_mode == "merged" and result.get("merged_dir"):
            ptr = {"model": result["merged_dir"], "served_name": s.student_served_name, "lora": None,
                   "run_id": cfg["run_id"], "domain": cfg["domain"]}
            served = s.student_served_name
        else:
            name = f"student-{cfg['run_id']}"
            base = self.read_pointer().get("model") if cfg["mode"] == "dpo" and cfg.get("init_from") else s.student_base_model
            ptr = {"model": base, "served_name": s.student_served_name, "lora": {"name": name, "path": result["adapter_dir"]},
                   "run_id": cfg["run_id"], "domain": cfg["domain"]}
            served = name
        await self._mark(run, Phase.RELOADING, model=ptr["model"], lora=ptr["lora"])
        self._write_pointer(ptr)
        await self._start_student_and_wait(served)
        await self.store.set(served, "lifecycle", "served_model")
        await self._mark(run, Phase.SERVING, served_model=served)
        return result

    async def _start_student_and_wait(self, served: str, timeout_s: float | None = None) -> None:
        timeout_s = timeout_s or self.s.student_start_timeout_s
        trainer = await self.docker.status(self.s.trainer_container)
        if trainer.status in ("running", "restarting"):
            raise LifecycleError(f"refusing to start the student: {self.s.trainer_container} is still {trainer.status}")
        await self.docker.start(self.s.student_container)
        root = self.s.student_url[:-3] if self.s.student_url.endswith("/v1") else self.s.student_url
        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                info = await self.docker.status(self.s.student_container)
                if info.status not in ("running", "restarting", "created"):
                    raise LifecycleError(f"student container exited during reload ({info.status}, code {info.exit_code})")
                try:
                    h = await client.get(f"{root}/health")
                    if h.status_code == 200:
                        m = await client.get(f"{root}/v1/models")
                        if served in [x["id"] for x in m.json().get("data", [])]:
                            return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        raise LifecycleError(f"student did not become healthy serving {served!r} within {timeout_s}s")

    async def _rollback(self, previous: dict[str, Any]) -> None:
        try:
            self._write_pointer(previous)
            served = previous["lora"]["name"] if previous.get("lora") else previous.get("served_name", self.s.student_served_name)
            await self.docker.stop(self.s.trainer_container)
            try:
                await self.gpu.wait_for_flush(self.s.student_gpu, self.s.vram_flush_threshold_mb,
                                              self.s.vram_flush_timeout_s, self.s.vram_required_free_mb)
            except TimeoutError as exc:
                # Never leave the stack without a student: try anyway, vLLM
                # reports precisely what it could not allocate.
                log.warning("rollback: %s - starting the student anyway", exc)
            await self._start_student_and_wait(served)
            await self.store.set(served, "lifecycle", "served_model")
            await self._set_phase(Phase.SERVING, rolled_back=True)
        except Exception as exc:
            log.exception("rollback failed")
            await self._set_phase(Phase.FAILED, error=f"rollback failed: {exc}")

    async def recover(self) -> dict[str, Any]:
        """Manual recovery from FAILED: restart the student on the pointer."""
        if self._lock.locked():
            raise LifecycleError("cycle in progress")
        await self._rollback(self.read_pointer())
        return await self.status()

    async def rollback_to(self, run_id: str | None) -> dict[str, Any]:
        """Serve the weights of a previous successful run (or the base model)."""
        if self._lock.locked():
            raise LifecycleError("cycle in progress")
        async with self._lock:
            if run_id is None:
                ptr = {"model": self.s.student_base_model, "served_name": self.s.student_served_name, "lora": None}
            else:
                run = await self.store.hget(("training", "runs"), run_id)
                if not run or run.get("status") != "succeeded":
                    raise LifecycleError(f"run {run_id} is not a successful run")
                res = run["result"]
                if res.get("merged_dir"):
                    ptr = {"model": res["merged_dir"], "served_name": self.s.student_served_name, "lora": None, "run_id": run_id}
                else:
                    ptr = {"model": self.s.student_base_model, "served_name": self.s.student_served_name,
                           "lora": {"name": f"student-{run_id}", "path": res["adapter_dir"]}, "run_id": run_id}
            await self._set_phase(Phase.DRAINING)
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
            await self.docker.stop(self.s.student_container)
            await self._flush("rollback")
            await self._set_phase(Phase.RELOADING, model=ptr["model"])
            self._write_pointer(ptr)
            served = ptr["lora"]["name"] if ptr.get("lora") else ptr["served_name"]
            await self._start_student_and_wait(served)
            await self.store.set(served, "lifecycle", "served_model")
            await self._set_phase(Phase.SERVING, served_model=served)
        return await self.status()


def _env(name: str) -> str | None:
    return os.environ.get(name) or None
