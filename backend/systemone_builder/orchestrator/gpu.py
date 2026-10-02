"""NVML-backed GPU telemetry and VRAM flush verification.

:class:`NvmlGpuMonitor` talks to the NVIDIA driver through ``pynvml``
(``nvidia-ml-py``). The API container needs access to the host GPUs
(``NVIDIA_VISIBLE_DEVICES=all`` + ``pid: host`` so per-process usage is visible).
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any

MIB = 1024 * 1024


@dataclass
class GpuProcess:
    pid: int
    used_mb: int
    name: str = ""


@dataclass
class GpuStatus:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int
    temperature_c: int | None = None
    power_w: float | None = None
    processes: list[GpuProcess] = field(default_factory=list)
    role: str | None = None

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["memory_free_mb"] = self.memory_free_mb
        return d


class GpuMonitor(abc.ABC):
    roles: dict[int, str]

    @abc.abstractmethod
    def snapshot(self) -> list[GpuStatus]: ...

    def get(self, index: int) -> GpuStatus | None:
        return next((g for g in self.snapshot() if g.index == index), None)

    async def wait_for_flush(
        self,
        index: int,
        threshold_mb: int,
        timeout_s: float,
        required_free_mb: int | None = None,
        poll_s: float = 0.5,
        stable_s: float = 3.0,
    ) -> GpuStatus:
        """Block until GPU ``index`` has been released by our containers.

        Flushed means either:

        * used VRAM is below ``threshold_mb`` (a dedicated Linux GPU), or
        * at least ``required_free_mb`` is free and usage has stopped falling
          for ``stable_s`` seconds. This covers GPUs that also drive a display
          or, on WSL2, where NVML counts memory held by Windows itself
          (desktop compositor, browsers) that never frees.

        Raises :class:`TimeoutError` otherwise; the lifecycle treats that as a
        hard stop rather than risk an OOM.
        """
        deadline = time.monotonic() + timeout_s
        stable_since: float | None = None
        last_used: int | None = None
        while True:
            status = self.get(index)
            if status is None:
                raise RuntimeError(f"GPU {index} not found")
            if status.memory_used_mb < threshold_mb:
                return status
            now = time.monotonic()
            if last_used is None or abs(status.memory_used_mb - last_used) > 64:
                stable_since = now
            last_used = status.memory_used_mb
            if required_free_mb is not None and status.memory_free_mb >= required_free_mb and now - (stable_since or now) >= stable_s:
                return status
            if now >= deadline:
                need = f"{required_free_mb} MiB free" if required_free_mb is not None else f"< {threshold_mb} MiB used"
                raise TimeoutError(
                    f"GPU {index} VRAM not released: {status.memory_used_mb} MiB used / {status.memory_free_mb} MiB free "
                    f"(need {need}). Memory held outside systemone - another process, or on WSL2 the Windows desktop "
                    f"and apps using this GPU - counts against it; close them or lower S1_VRAM_REQUIRED_FREE_MB."
                )
            await asyncio.sleep(poll_s)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class NvmlGpuMonitor(GpuMonitor):
    def __init__(self, roles: dict[int, str] | None = None) -> None:
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self.roles = roles or {}

    def snapshot(self) -> list[GpuStatus]:
        nv = self._nvml
        out: list[GpuStatus] = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            name = nv.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            try:
                temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            except nv.NVMLError:
                temp = None
            try:
                power = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except nv.NVMLError:
                power = None
            procs: list[GpuProcess] = []
            try:
                for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
                    used = (p.usedGpuMemory or 0) // MIB
                    procs.append(GpuProcess(pid=p.pid, used_mb=used))
            except nv.NVMLError:
                pass
            out.append(
                GpuStatus(
                    index=i,
                    name=name,
                    memory_total_mb=mem.total // MIB,
                    memory_used_mb=mem.used // MIB,
                    utilization_pct=util.gpu,
                    temperature_c=temp,
                    power_w=power,
                    processes=procs,
                    role=self.roles.get(i),
                )
            )
        return out

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


def create_gpu_monitor(roles: dict[int, str]) -> GpuMonitor:
    return NvmlGpuMonitor(roles)


def placement_warnings(gpus: list[GpuStatus], expected: dict[str, int], running: set[str], min_used_mb: int = 3000) -> list[str]:
    """Services that are up while their assigned GPU is nearly idle.

    A vLLM server claims most of its GPU's memory, so an up service whose GPU
    shows almost nothing used is running somewhere else - typically because
    the container runtime exposed every GPU (WSL2) and CUDA picked GPU 0.
    """
    by_index = {g.index: g for g in gpus}
    out = []
    for name, idx in expected.items():
        g = by_index.get(idx)
        if name in running and g is not None and g.memory_used_mb < min_used_mb:
            others = ", ".join(f"GPU {o.index}: {o.memory_used_mb} MiB" for o in gpus if o.index != idx)
            out.append(
                f"{name} is running but its GPU {idx} shows only {g.memory_used_mb} MiB used ({others}); "
                f"it is probably on another GPU. Rebuild the launcher image and recreate the container "
                f"so it pins itself (S1_GPU_INDEX)."
            )
    return out

