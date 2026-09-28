"""Flush criterion logic (NVML itself is exercised on the rig)."""

from __future__ import annotations

import pytest

from systemone.orchestrator.gpu import GpuMonitor, GpuStatus


class Scripted(GpuMonitor):
    roles: dict[int, str] = {}

    def __init__(self, used: list[int]) -> None:
        self.used = used
        self.i = 0

    def snapshot(self) -> list[GpuStatus]:
        u = self.used[min(self.i, len(self.used) - 1)]
        self.i += 1
        return [GpuStatus(0, "RTX 3090", 24576, u, 0)]


async def test_dedicated_gpu_flushes_below_threshold():
    st = await Scripted([20800, 900]).wait_for_flush(0, 1024, 1.0, poll_s=0.01)
    assert st.memory_used_mb == 900


async def test_wsl2_desktop_baseline_passes_on_free_memory_once_stable():
    # vLLM releases memory, then Windows keeps holding ~2.3 GB forever
    mon = Scripted([20800, 12000, 2300, 2300, 2300, 2300, 2300])
    st = await mon.wait_for_flush(0, 1024, 2.0, required_free_mb=21000, poll_s=0.01, stable_s=0.03)
    assert st.memory_used_mb == 2300


async def test_times_out_when_not_enough_free():
    with pytest.raises(TimeoutError, match="need 21000 MiB free"):
        await Scripted([6000]).wait_for_flush(0, 1024, 0.1, required_free_mb=21000, poll_s=0.01, stable_s=0.01)


def test_placement_warning_when_triage_gpu_idle():
    from systemone.orchestrator.gpu import placement_warnings

    gpus = [GpuStatus(0, "a", 24576, 23522, 0), GpuStatus(1, "b", 24576, 450, 0)]
    w = placement_warnings(gpus, {"triage": 1}, {"triage"})
    assert len(w) == 1 and "GPU 1 shows only 450 MiB" in w[0]
    assert placement_warnings(gpus, {"triage": 1}, set()) == []
    ok = [GpuStatus(0, "a", 24576, 2300, 0), GpuStatus(1, "b", 24576, 22400, 0)]
    assert placement_warnings(ok, {"triage": 1}, {"triage"}) == []
