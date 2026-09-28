"use client";

import { useState } from "react";
import { LineChart } from "@/components/LineChart";
import { ServiceTable, useServices } from "@/components/Services";
import { Card, PageTitle, Stat, StatusPill } from "@/components/ui";
import { fmtMb, fmtMs, fmtNum, fmtPct, fmtTime } from "@/utils/format";
import { useEvents, usePoll, type TelemetrySample } from "@/utils/hooks";

type Loss = { ts: number; run_id: string; step: number; loss: number };

const PHASE_STATUS: Record<string, "good" | "warning" | "serious" | "critical"> = {
  serving: "good",
  draining: "warning",
  paused: "warning",
  flushing: "warning",
  training: "serious",
  reloading: "serious",
  failed: "critical",
};

export default function TelemetryPage() {
  const [samples, setSamples] = useState<TelemetrySample[]>([]);
  const [losses, setLosses] = useState<Loss[]>([]);
  useEvents(
    ["telemetry", "training"],
    (e) => {
      if (e.channel === "telemetry" && e.type === "tick") setSamples((s) => [...s.slice(-599), e.data as TelemetrySample]);
      if (e.channel === "training" && e.type === "log" && e.data.loss != null)
        setLosses((l) => [...l.slice(-4999), { ts: e.ts, run_id: e.data.run_id, step: e.data.step, loss: e.data.loss }]);
    },
    (hello) => {
      setSamples(hello.telemetry.samples ?? []);
      setLosses(hello.telemetry.losses ?? []);
    },
  );
  const lifecycle = usePoll<any>("/training/status", 3000);
  const replay = usePoll<any>("/replay/stats", 5000);
  const services = useServices(4000);

  const last = samples[samples.length - 1];
  const x = (s: TelemetrySample) => s.ts;
  const run = losses.length ? losses[losses.length - 1].run_id : null;
  const runLoss = losses.filter((l) => l.run_id === run);
  const phase = lifecycle.data?.phase ?? "unknown";

  return (
    <div>
      <PageTitle
        title="System-1 Telemetry"
        sub="Sub-100ms reflex path, prefix-cache health and training on GPU 0"
      />
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="TTFT p50 (student)" value={fmtMs(last?.ttft_ms_p50)} sub={`p95 ${fmtMs(last?.ttft_ms_p95)}`} accent="var(--tier-student)" />
        <Stat label="Prefix-cache hit rate" value={fmtPct(last?.vllm?.student?.prefix_cache_hit_rate)} sub={`lifetime ${fmtPct(last?.vllm?.student?.prefix_cache_hit_rate_lifetime)}`} />
        <Stat label="End-to-end p50" value={fmtMs(last?.e2e_ms_p50)} sub={`${fmtNum(last?.rps, 1)} req/s`} />
        <Stat label="Replay buffer" value={(replay.data?.size ?? 0).toLocaleString()} sub={`of ${(replay.data?.capacity ?? 10000).toLocaleString()} actions`} />
        <div className="rounded-lg border border-line bg-surface px-4 py-3">
          <div className="text-xs text-ink-3">GPU 0 lifecycle</div>
          <div className="mt-2">
            <StatusPill status={PHASE_STATUS[phase] ?? "warning"} label={phase} />
          </div>
          <div className="mt-1 truncate font-mono text-xs text-ink-3" title={lifecycle.data?.served_model}>
            {lifecycle.data?.served_model ?? "–"}
          </div>
        </div>
      </div>

      <Card
        title="Services"
        className="mb-4"
        actions={
          services.data && (
            <StatusPill
              status={services.data.operational ? "good" : "warning"}
              label={services.data.operational ? "all systems operational" : "not fully operational"}
            />
          )
        }
      >
        <ServiceTable snap={services.data} />
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Time-To-First-Token (student, ms)">
          <LineChart
            series={[
              { key: "p50", label: "p50", color: "var(--tier-student)", points: samples.map((s) => ({ x: x(s), y: s.ttft_ms_p50 })) },
              { key: "p95", label: "p95", color: "var(--tier-triage)", points: samples.map((s) => ({ x: x(s), y: s.ttft_ms_p95 })) },
            ]}
            yFormat={(v) => `${Math.round(v)}`}
            xFormat={fmtTime}
            yMin={0}
            reference={{ y: 100, label: "100 ms budget" }}
            empty="Waiting for student traffic"
          />
        </Card>
        <Card title="vLLM prefix-cache hit rate">
          <LineChart
            series={[
              { key: "student", label: "student", color: "var(--tier-student)", points: samples.map((s) => ({ x: x(s), y: s.vllm?.student?.prefix_cache_hit_rate ?? null })) },
              { key: "triage", label: "triage", color: "var(--tier-triage)", points: samples.map((s) => ({ x: x(s), y: s.vllm?.triage?.prefix_cache_hit_rate ?? null })) },
            ]}
            yFormat={(v) => `${Math.round(v * 100)}%`}
            xFormat={fmtTime}
            yMin={0}
            yMax={1}
            empty="Waiting for vLLM /metrics"
          />
        </Card>
        <Card title={`Unsloth training loss${run ? ` · ${run}` : ""}`}>
          <LineChart
            series={[{ key: "loss", label: "loss", color: "var(--tier-student)", points: runLoss.map((l) => ({ x: l.step, y: l.loss })) }]}
            yFormat={(v) => v.toFixed(2)}
            xFormat={(v) => `step ${v}`}
            empty="No training run yet"
          />
        </Card>
        <Card title="GPUs (NVML)">
          <div className="grid gap-3 sm:grid-cols-2">
            {(last?.gpus ?? []).map((g: any) => {
              const frac = g.memory_used_mb / g.memory_total_mb;
              return (
                <div key={g.index} className="rounded-md border border-line bg-surface-2 p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-ink">GPU {g.index}</span>
                    <span className="text-ink-3">{g.role === "student" ? "Student · A/B" : g.role === "triage" ? "Triage · C" : g.role}</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded bg-surface">
                    <div className="h-full rounded" style={{ width: `${frac * 100}%`, background: g.role === "triage" ? "var(--tier-triage)" : "var(--tier-student)" }} />
                  </div>
                  <div className="tabular mt-1.5 flex justify-between text-xs text-ink-2">
                    <span>
                      {fmtMb(g.memory_used_mb)} / {fmtMb(g.memory_total_mb)}
                    </span>
                    <span>{g.utilization_pct}% util</span>
                  </div>
                  <div className="tabular mt-0.5 flex justify-between text-xs text-ink-3">
                    <span>{g.temperature_c ?? "–"}°C</span>
                    <span>{g.power_w != null ? `${Math.round(g.power_w)} W` : "–"}</span>
                  </div>
                  {frac > 0.97 && (
                    <div className="mt-1.5">
                      <StatusPill status="critical" label="OOM risk" />
                    </div>
                  )}
                </div>
              );
            })}
            {!last?.gpus?.length && <div className="text-xs text-ink-3">Waiting for NVML samples</div>}
          </div>
          {last?.vllm && (
            <div className="tabular mt-3 grid grid-cols-2 gap-3 text-xs text-ink-2">
              {(["student", "triage"] as const).map((k) => (
                <div key={k}>
                  <div className="text-ink-3">{k} vLLM</div>
                  {last.vllm[k] ? (
                    <>
                      KV cache {fmtPct(last.vllm[k].kv_cache_usage)} · running {last.vllm[k].running ?? "–"} · waiting {last.vllm[k].waiting ?? "–"}
                    </>
                  ) : (
                    <span style={{ color: "var(--critical)" }}>✕ unreachable</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
