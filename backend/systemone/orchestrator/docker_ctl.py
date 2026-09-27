"""Docker Engine control for the strict container lifecycle.

The API process talks to the host Docker daemon (``/var/run/docker.sock`` is
mounted into the API container) to pause/resume the vLLM student runner,
launch the one-shot Unsloth trainer and inspect container health.

All Docker SDK calls are blocking, so they are pushed to a worker thread.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    name: str
    status: str  # running | exited | created | paused | missing
    image: str | None = None
    exit_code: int | None = None
    started_at: str | None = None
    gpus: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RunSpec:
    """One-shot container specification (used for the trainer)."""

    name: str
    image: str
    command: list[str]
    gpu: int
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    shm_size: str = "16g"
    network: str | None = None


class ContainerController(abc.ABC):
    @abc.abstractmethod
    async def status(self, name: str) -> ContainerInfo: ...

    @abc.abstractmethod
    async def start(self, name: str) -> ContainerInfo: ...

    @abc.abstractmethod
    async def stop(self, name: str, timeout_s: int = 30) -> ContainerInfo: ...

    @abc.abstractmethod
    async def run_to_completion(
        self, spec: RunSpec, on_log: Callable[[str], None] | None = None, timeout_s: float | None = None
    ) -> int:
        """Run a one-shot container, stream its logs, remove it, return exit code."""

    @abc.abstractmethod
    async def logs(self, name: str, tail: int = 200) -> str: ...

    async def restart(self, name: str) -> ContainerInfo:
        await self.stop(name)
        return await self.start(name)


class DockerController(ContainerController):
    def __init__(self, client: Any | None = None) -> None:
        import docker

        self._docker = docker
        self.client = client or docker.from_env()

    async def _call(self, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        return await asyncio.to_thread(fn, *a, **kw)

    def _info(self, c: Any) -> ContainerInfo:
        attrs = c.attrs or {}
        state = attrs.get("State", {})
        gpus: list[int] = []
        for req in (attrs.get("HostConfig", {}) or {}).get("DeviceRequests") or []:
            for d in req.get("DeviceIDs") or []:
                if str(d).isdigit():
                    gpus.append(int(d))
        return ContainerInfo(
            name=c.name,
            status=c.status,
            image=(c.image.tags[0] if c.image and c.image.tags else None),
            exit_code=state.get("ExitCode"),
            started_at=state.get("StartedAt"),
            gpus=gpus,
        )

    async def _get(self, name: str) -> Any | None:
        try:
            return await self._call(self.client.containers.get, name)
        except self._docker.errors.NotFound:
            return None

    async def status(self, name: str) -> ContainerInfo:
        c = await self._get(name)
        return self._info(c) if c else ContainerInfo(name=name, status="missing")

    async def start(self, name: str) -> ContainerInfo:
        c = await self._get(name)
        if c is None:
            raise RuntimeError(f"container {name!r} does not exist (did you run `docker compose up`?)")
        await self._call(c.start)
        await self._call(c.reload)
        return self._info(c)

    async def stop(self, name: str, timeout_s: int = 30) -> ContainerInfo:
        c = await self._get(name)
        if c is None:
            return ContainerInfo(name=name, status="missing")
        if c.status in ("running", "restarting", "paused"):
            await self._call(c.stop, timeout=timeout_s)
            await self._call(c.reload)
        return self._info(c)

    async def logs(self, name: str, tail: int = 200) -> str:
        c = await self._get(name)
        if c is None:
            return ""
        raw = await self._call(c.logs, tail=tail)
        return raw.decode(errors="replace")

    async def run_to_completion(
        self, spec: RunSpec, on_log: Callable[[str], None] | None = None, timeout_s: float | None = None
    ) -> int:
        old = await self._get(spec.name)
        if old is not None:
            await self._call(old.remove, force=True)
        device = self._docker.types.DeviceRequest(device_ids=[str(spec.gpu)], capabilities=[["gpu"]])
        c = await self._call(
            self.client.containers.run,
            spec.image,
            spec.command,
            name=spec.name,
            detach=True,
            environment=spec.environment,
            volumes=spec.volumes,
            device_requests=[device],
            shm_size=spec.shm_size,
            network=spec.network,
            ipc_mode="host",
        )

        def _stream() -> None:
            for line in c.logs(stream=True, follow=True):
                if on_log:
                    on_log(line.decode(errors="replace").rstrip())

        stream_task = asyncio.create_task(asyncio.to_thread(_stream))
        try:
            result = await asyncio.wait_for(self._call(c.wait), timeout=timeout_s)
            code = int(result.get("StatusCode", 1))
        except asyncio.TimeoutError:
            await self._call(c.kill)
            code = 124
        finally:
            await asyncio.wait([stream_task], timeout=5)
            try:
                await self._call(c.remove, force=True)
            except Exception:  # pragma: no cover
                log.warning("could not remove container %s", spec.name)
        return code
