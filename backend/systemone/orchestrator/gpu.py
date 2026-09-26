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

    async def wait_for_flush(self, index: int, threshold_mb: int, timeout_s: float, poll_s: float = 0.5) -> GpuStatus:
        """Block until GPU ``index`` uses less than ``threshold_mb`` MiB.

        Raises :class:`TimeoutError` when VRAM is not released in time, which
        the lifecycle orchestrator treats as a hard stop (never start training
        on a card that could OOM).
        """
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.get(index)
            if status is None:
                raise RuntimeError(f"GPU {index} not found")
            if status.memory_used_mb < threshold_mb:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"GPU {index} VRAM not flushed: {status.memory_used_mb} MiB used (threshold {threshold_mb} MiB)"
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
