"""System-1 telemetry collector (Module F, Phase 6).

Samples once per ``interval_s`` and keeps a rolling history for the
dashboard's live graphs:

* TTFT / end-to-end latency of every routed decision (from the event bus)
* vLLM prefix-cache hit rate for the student and triage servers, scraped
  from their Prometheus ``/metrics`` endpoints
* KV-cache usage and running/waiting requests per vLLM server
* GPU memory / utilization / power from NVML
* Unsloth training loss (from the lifecycle's metric stream)

Each sample is published on the ``telemetry`` channel and the history is
served to newly connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from typing import Any

from systemone_builder.orchestrator.gpu import GpuMonitor
from systemone_builder.telemetry.bus import EventBus

log = logging.getLogger(__name__)

_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name (across label sets)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + val
    return out


def _first(metrics: dict[str, float], *names: str) -> float | None:
    for n in names:
        if n in metrics:
            return metrics[n]
    return None


class VllmScraper:
    """Derives an interval prefix-cache hit rate from vLLM counters."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._prev: tuple[float, float] | None = None

    async def sample(self) -> dict[str, Any] | None:
        try:
            m = parse_prometheus(await self.adapter.metrics_text())
        except Exception:
            self._prev = None
            return None
        hits = _first(m, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
        queries = _first(m, "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
        rate: float | None = None
        if hits is not None and queries is not None:
            if self._prev is not None:
                dh, dq = hits - self._prev[0], queries - self._prev[1]
                rate = (dh / dq) if dq > 0 else None
            self._prev = (hits, queries)
            lifetime = hits / queries if queries else None
        else:  # vLLM V0 exposes a gauge
            rate = _first(m, "vllm:gpu_prefix_cache_hit_rate")
            lifetime = rate
        return {
            "prefix_cache_hit_rate": rate,
            "prefix_cache_hit_rate_lifetime": lifetime,
            "kv_cache_usage": _first(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            "cpu_kv_cache_usage": _first(m, "vllm:cpu_cache_usage_perc"),
            "running": _first(m, "vllm:num_requests_running"),
            "waiting": _first(m, "vllm:num_requests_waiting"),
            "preemptions": _first(m, "vllm:num_preemptions_total", "vllm:num_preemptions"),
        }


class TelemetryCollector:
    def __init__(
        self,
        bus: EventBus,
        gpu: GpuMonitor,
        student: Any,
        triage: Any,
        interval_s: float = 1.0,
        history: int = 900,
    ) -> None:
        self.bus = bus
        self.gpu = gpu
        self.interval_s = interval_s
        # Adapters may be BYOM handles; non-vLLM backends simply yield no metrics.
        self.scrapers = {name: VllmScraper(a) for name, a in (("student", student), ("triage", triage))}
        self.samples: deque[dict[str, Any]] = deque(maxlen=history)
        self.decisions: deque[dict[str, Any]] = deque(maxlen=history * 4)
        self.losses: deque[dict[str, Any]] = deque(maxlen=5000)
        self._task: asyncio.Task[None] | None = None
        self._listener: asyncio.Task[None] | None = None

    async def _listen(self) -> None:
        q = self.bus.subscribe()
        try:
            while True:
                ev = await q.get()
                ch, typ, d = ev["channel"], ev["type"], ev["data"]
                if ch == "routing" and typ == "tier_result" and d.get("tier") == "student" and d.get("ttft_ms") is not None:
                    self.decisions.append({"ts": ev["ts"], "ttft_ms": d["ttft_ms"], "latency_ms": d.get("latency_ms"),
                                           "cached_tokens": d.get("cached_tokens"), "prompt_tokens": d.get("prompt_tokens")})
                elif ch == "routing" and typ == "decision":
                    self.decisions.append({"ts": ev["ts"], "e2e_ms": d.get("latency_ms"), "tier": d.get("tier"),
                                           "halted": d.get("halted")})
                elif ch == "training" and typ == "log" and "loss" in d:
                    self.losses.append({"ts": ev["ts"], "run_id": d.get("run_id"), "step": d.get("step"), "loss": d["loss"]})
        finally:
            self.bus.unsubscribe(q)

    async def sample_once(self) -> dict[str, Any]:
        now = time.time()
        window = [d for d in self.decisions if now - d["ts"] <= max(self.interval_s, 1.0) * 5]
        ttfts = [d["ttft_ms"] for d in window if "ttft_ms" in d]
        e2e = [d["e2e_ms"] for d in window if d.get("e2e_ms") is not None]
        tiers = [d.get("tier") for d in window if "tier" in d]
        cached = sum(d.get("cached_tokens") or 0 for d in window if "ttft_ms" in d)
        prompt = sum(d.get("prompt_tokens") or 0 for d in window if "ttft_ms" in d)
        vllm = {}
        for name, sc in self.scrapers.items():
            vllm[name] = await sc.sample()
        try:
            gpus = [g.to_dict() for g in self.gpu.snapshot()]
        except Exception:
            gpus = []
        sample = {
            "ts": now,
            "ttft_ms_p50": _pct(ttfts, 50),
            "ttft_ms_p95": _pct(ttfts, 95),
            "e2e_ms_p50": _pct(e2e, 50),
            "rps": len(tiers) / max(self.interval_s * 5, 1.0),
            "tier_mix": {t: tiers.count(t) for t in set(tiers)},
            "request_prefix_token_ratio": (cached / prompt) if prompt else None,
            "vllm": vllm,
            "gpus": [{k: g[k] for k in ("index", "memory_used_mb", "memory_total_mb", "utilization_pct", "power_w", "temperature_c", "role")} for g in gpus],
            "loss": self.losses[-1]["loss"] if self.losses else None,
        }
        self.samples.append(sample)
        self.bus.publish("telemetry", "tick", **sample)
        return sample

    async def _loop(self) -> None:
        while True:
            try:
                await self.sample_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("telemetry sample failed")
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self._listener = asyncio.create_task(self._listen(), name="telemetry-listener")
        self._task = asyncio.create_task(self._loop(), name="telemetry-collector")

    async def stop(self) -> None:
        for t in (self._task, self._listener):
            if t:
                t.cancel()
        await asyncio.gather(*(t for t in (self._task, self._listener) if t), return_exceptions=True)

    def snapshot(self, n: int = 300) -> dict[str, Any]:
        return {"samples": list(self.samples)[-n:], "losses": list(self.losses)[-2000:]}


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p / 100)))]
