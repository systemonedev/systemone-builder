"""Hardware topology and process-isolation auditing (Phase 1).

The reference topology pins every container to exactly one GPU:

========  =========================================  ==================
GPU       Containers                                 Role
========  =========================================  ==================
0         A: Unsloth trainer, B: vLLM student         Sub-100ms Student
1         C: vLLM triage server                      Synchronous triage
========  =========================================  ==================

Container A and B share GPU 0 and must never be resident at the same time -
that invariant is what :func:`HardwareOrchestrator.audit` verifies and what
the lifecycle orchestrator (Phase 4) enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from systemone.config import Settings
from systemone.orchestrator.docker_ctl import ContainerController
from systemone.orchestrator.gpu import GpuMonitor


@dataclass
class Placement:
    container: str
    label: str
    gpu: int
    exclusive_group: str | None = None  # containers in a group may not co-reside


class HardwareOrchestrator:
    def __init__(self, settings: Settings, gpu: GpuMonitor, docker: ContainerController) -> None:
        self.settings = settings
        self.gpu = gpu
        self.docker = docker
        self.placements = [
            Placement(settings.trainer_container, "A: Unsloth QLoRA Trainer", settings.student_gpu, "gpu0-lifecycle"),
            Placement(settings.student_container, "B: vLLM Student Runner", settings.student_gpu, "gpu0-lifecycle"),
            Placement(settings.triage_container, "C: vLLM Triage Server", settings.triage_gpu),
        ]

    async def containers(self) -> list[dict[str, Any]]:
        out = []
        for p in self.placements:
            info = await self.docker.status(p.container)
            d = info.to_dict()
            d.update(label=p.label, assigned_gpu=p.gpu)
            out.append(d)
        return out

    async def audit(self) -> dict[str, Any]:
        """Return isolation violations (empty list == healthy)."""
        violations: list[str] = []
        running: dict[str, list[str]] = {}
        for p in self.placements:
            info = await self.docker.status(p.container)
            if info.status != "running":
                continue
            if info.gpus and p.gpu not in info.gpus:
                violations.append(f"{p.container} runs on GPU {info.gpus}, expected GPU {p.gpu}")
            if len(info.gpus) > 1:
                violations.append(f"{p.container} spans multiple GPUs {info.gpus}")
            if p.exclusive_group:
                running.setdefault(p.exclusive_group, []).append(p.container)
        for group, names in running.items():
            if len(names) > 1:
                violations.append(f"exclusive group {group} has co-resident containers: {names}")
        for g in self.gpu.snapshot():
            if g.memory_total_mb and g.memory_used_mb / g.memory_total_mb > 0.97:
                violations.append(f"GPU {g.index} VRAM at {g.memory_used_mb}/{g.memory_total_mb} MiB (OOM risk)")
        return {"healthy": not violations, "violations": violations}

    async def snapshot(self) -> dict[str, Any]:
        return {
            "gpus": [g.to_dict() for g in self.gpu.snapshot()],
            "containers": await self.containers(),
            "audit": await self.audit(),
        }
