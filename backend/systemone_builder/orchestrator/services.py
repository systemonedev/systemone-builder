"""Service status for the dashboard: is each part of the stack operational?

Combines three signals per model service:

* the Docker container state (running / exited + exit code / restarting)
* the model endpoint's own health check and served models
* for GPU 0, the lifecycle phase (the student is *expected* to be down while
  a training cycle runs)

While a vLLM container is running but not yet healthy, the latest meaningful
log line is parsed so the UI can show *why* it is not ready yet (downloading,
loading safetensors shards 2/3, compiling, capturing CUDA graphs ...).
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from systemone_builder.orchestrator.docker_ctl import ContainerController
from systemone_builder.orchestrator.gpu import placement_warnings

# Most specific first. Each maps a vLLM / HF log line to a short progress label.
_PROGRESS = [
    (re.compile(r"Loading safetensors checkpoint shards:\s+(\d+)% Completed \| (\d+)/(\d+)"), lambda m: f"loading weights {m.group(2)}/{m.group(3)} shards ({m.group(1)}%)"),
    (re.compile(r"Loading weights took ([\d.]+) seconds"), lambda m: f"weights loaded in {float(m.group(1)):.0f}s"),
    (re.compile(r"(Downloading|Fetching \d+ files).*?(\d+%)"), lambda m: f"downloading model {m.group(2)}"),
    (re.compile(r"Using cache directory: .*torch_compile_cache"), lambda m: "compiling (torch.compile)"),
    (re.compile(r"torch\.compile took ([\d.]+) s"), lambda m: f"compiled in {float(m.group(1)):.0f}s"),
    (re.compile(r"Capturing CUDA graph"), lambda m: "capturing CUDA graphs"),
    (re.compile(r"CPU KV offload probe: (.*)"), lambda m: f"KV offload probe: {m.group(1)}"),
    (re.compile(r"Available KV cache memory: ([\d.]+ GiB)"), lambda m: f"allocating KV cache ({m.group(1)})"),
    (re.compile(r"Starting vLLM API server|Application startup complete"), lambda m: "starting API server"),
    (re.compile(r"\[serve_\w+\] exec:"), lambda m: "launching vLLM"),
]
_ERROR = re.compile(r"(Error|Exception|error:)[^\n]*$")


def summarize_logs(text: str) -> tuple[str | None, str | None]:
    """(progress label, last error line) from the tail of a container log."""
    progress = error = None
    lines = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
    for line in reversed(lines):
        if progress is None:
            for rx, fmt in _PROGRESS:
                m = rx.search(line)
                if m:
                    progress = fmt(m)
                    break
        if error is None and _ERROR.search(line) and "WARNING" not in line:
            error = re.sub(r"^.*?\] ", "", line.strip())[-300:]
        if progress and error:
            break
    return progress, error


async def _health(handle: Any, timeout_s: float = 3.0) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(handle.health(), timeout_s)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"no response within {timeout_s:.0f}s"}


class ServiceMonitor:
    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self._cache: tuple[float, dict[str, Any]] | None = None

    async def _model_service(self, role: str, container: str) -> dict[str, Any]:
        handle = getattr(self.rt, role)
        info, health = await asyncio.gather(self.rt.docker.status(container), _health(handle))
        out: dict[str, Any] = {
            "name": role, "container": container, "container_status": info.status,
            "exit_code": info.exit_code, "started_at": info.started_at, "gpus": info.gpus,
            "url": handle.config.url, "model": handle.config.model,
        }
        if health.get("ok"):
            try:
                out["served_models"] = await asyncio.wait_for(handle.list_models(), 3.0)
            except Exception:
                out["served_models"] = []
            out.update(state="online", detail="operational")
        elif info.status == "missing":
            out.update(state="offline", detail="container not created (docker compose up)")
        elif info.status in ("exited", "dead"):
            _, err = summarize_logs(await self.rt.docker.logs(container, tail=200))
            crashed = info.exit_code not in (0, None)
            out.update(state="crashed" if crashed else "stopped",
                       detail=err or (f"exited with code {info.exit_code}" if crashed else "stopped"))
        elif info.status == "restarting":
            _, err = summarize_logs(await self.rt.docker.logs(container, tail=200))
            out.update(state="crashed", detail=f"restart loop: {err or 'see container logs'}")
        else:  # running but not answering yet
            progress, _ = summarize_logs(await self.rt.docker.logs(container, tail=80))
            out.update(state="starting", detail=progress or "starting")
        if role == "student":
            phase = self.rt.lifecycle.phase.value
            out["lifecycle_phase"] = phase
            if phase not in ("serving",):
                out.update(state="training" if phase != "failed" else "crashed",
                           detail=f"GPU 0 lifecycle: {phase}")
            out["served_model"] = await self.rt.student_model_name()
        return out

    async def _oracle(self) -> dict[str, Any]:
        h = self.rt.oracle
        health = await _health(h)
        out = {"name": "oracle", "url": h.config.url, "model": h.config.model}
        if not health.get("ok"):
            return {**out, "state": "offline", "detail": health.get("error", "unreachable")}
        try:
            models = await asyncio.wait_for(h.list_models(), 3.0)
        except Exception:
            models = []
        out["served_models"] = models
        if models and h.config.model not in models:
            return {**out, "state": "degraded", "detail": f"model not pulled: ollama pull {h.config.model}"}
        return {**out, "state": "online", "detail": "operational"}

    async def _kenning(self) -> dict[str, Any]:
        url = self.rt.settings.kenning_url
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{url}/health")
            h = r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError) as exc:
            return {"name": "kenning", "url": url, "state": "offline", "detail": f"unreachable: {type(exc).__name__}"}
        if not h:
            return {"name": "kenning", "url": url, "state": "degraded", "detail": f"HTTP {r.status_code}"}
        return {"name": "kenning", "url": url, "state": "online", "detail": "operational", "model": h.get("model"),
                "container": "systemone-kenning"}

    async def _redis(self) -> dict[str, Any]:
        try:
            await self.rt.redis.ping()
            mem = await self.rt.store.memory_info()
            return {"name": "redis", "state": "online", "detail": f"{mem.get('used_memory_human')} / {mem.get('maxmemory_human')}"}
        except Exception as exc:
            return {"name": "redis", "state": "offline", "detail": repr(exc)}

    async def snapshot(self, max_age_s: float = 2.0) -> dict[str, Any]:
        if self._cache and time.monotonic() - self._cache[0] < max_age_s:
            return self._cache[1]
        s = self.rt.settings
        api = {"name": "api", "state": "online", "detail": "operational"}
        if not s.pipeline:
            # Kenning-only stack: the student, triage and oracle are not part of it.
            kenning, redis = await asyncio.gather(self._kenning(), self._redis())
            services = [api, kenning, redis]
            snap = {"ts": time.time(), "services": services, "warnings": [], "pipeline": False,
                    "operational": all(x["state"] == "online" for x in services)}
            self._cache = (time.monotonic(), snap)
            return snap
        kenning, student, triage, oracle, redis = await asyncio.gather(
            self._kenning(),
            self._model_service("student", s.student_container),
            self._model_service("triage", s.triage_container),
            self._oracle(),
            self._redis(),
        )
        services = [api, kenning, redis, student, triage, oracle]
        up = {x["name"] for x in (student, triage) if x["state"] == "online"}
        try:
            warnings = placement_warnings(self.rt.gpu.snapshot(), {"student": s.student_gpu, "triage": s.triage_gpu}, up)
        except Exception:
            warnings = []
        snap = {"ts": time.time(), "services": services, "warnings": warnings, "pipeline": True,
                "operational": all(x["state"] == "online" for x in services) and not warnings}
        self._cache = (time.monotonic(), snap)
        return snap
