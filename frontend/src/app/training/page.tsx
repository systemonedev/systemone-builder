"use client";

import { useState } from "react";
import { Button, Card, ErrorNote, Field, inputCls, Json, PageTitle, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { notify } from "@/utils/notify";
import { fmtMb, fmtTime } from "@/utils/format";
import { useEvents, usePoll } from "@/utils/hooks";

const PHASES = ["serving", "draining", "paused", "flushing", "training", "reloading"];

function Stepper({ phase }: { phase: string }) {
  const idx = PHASES.indexOf(phase);
  return (
    <ol className="flex flex-wrap items-center gap-1 text-xs">
      {PHASES.map((p, i) => {
        const cur = p === phase;
        const done = idx > 0 && i < idx && i > 0;
        return (
          <li key={p} className="flex items-center gap-1">
            <span
              className={`rounded-full border px-2.5 py-1 ${cur ? "border-accent font-medium text-ink" : done ? "border-line text-ink-2" : "border-line text-ink-3"}`}
            >
              {done ? "✓ " : cur ? "● " : ""}
              {p}
            </span>
            {i < PHASES.length - 1 && <span className="text-ink-3">→</span>}
          </li>
        );
      })}
      {phase === "failed" && <StatusPill status="critical" label="failed" />}
    </ol>
  );
}

export default function TrainingPage() {
  const status = usePoll<any>("/training/status", 2000);
  const runs = usePoll<any[]>("/training/runs", 5000);
  const factory = usePoll<any>("/factory/status", 4000);
  const domains = usePoll<any[]>("/domains", 0);
  const [domain, setDomain] = useState("computer_use");
  const [mode, setMode] = useState("sft");
  const [scenarios, setScenarios] = useState("");
  const [per, setPer] = useState(10);
  const [err, setErr] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  useEvents(["lifecycle", "training", "factory"], (e) => {
    const d = e.data;
    const line =
      e.channel === "lifecycle"
        ? `lifecycle → ${d.phase}${d.vram_used_mb != null ? ` (GPU0 ${d.vram_used_mb} MiB)` : ""}${d.error ? ` ✕ ${d.error}` : ""}`
        : e.channel === "training" && e.type === "log"
          ? `step ${d.step}/${d.max_steps ?? "?"} loss ${d.loss?.toFixed?.(4) ?? "–"}`
          : e.channel === "training"
            ? `training ${e.type} ${d.phase ?? d.error ?? ""}`
            : `factory ${e.type} ${d.domain ?? ""} ${d.source ?? ""}${d.judge_score != null ? ` judge ${d.judge_score.toFixed(2)}` : ""}`;
    setLog((l) => [`${fmtTime(e.ts)}  ${line}`, ...l].slice(0, 200));
    if (e.channel === "lifecycle") status.reload();
  });

  const call = async (fn: () => Promise<any>, ok?: string | ((r: any) => string)) => {
    try {
      setErr(null);
      const r = await fn();
      if (ok) notify(typeof ok === "function" ? ok(r) : ok, "ok");
      status.reload();
      runs.reload();
      factory.reload();
    } catch (e: any) {
      setErr(e.message);
      notify(e.message, "error");
    }
  };

  const s = status.data;
  // busy while a cycle holds the lock or GPU 0 is anywhere between serving states
  const cycleActive = !!s && (s.busy || !["serving", "failed", undefined].includes(s.phase));
  return (
    <div>
      <PageTitle title="Synthetic Factory & Training" sub="Strict GPU 0 lifecycle: pause vLLM → flush VRAM → Unsloth QLoRA → hot-reload" />
      <ErrorNote error={err} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="GPU 0 lifecycle">
          <Stepper phase={s?.phase ?? "serving"} />
          <div className="tabular mt-4 grid grid-cols-2 gap-3 text-xs text-ink-2">
            <div>
              <div className="text-ink-3">served model</div>
              <div className="truncate font-mono text-ink" title={s?.served_model}>{s?.served_model ?? "–"}</div>
            </div>
            <div>
              <div className="text-ink-3">weights</div>
              <div className="truncate font-mono" title={s?.pointer?.model}>{s?.pointer?.model ?? "–"}</div>
            </div>
            <div>
              <div className="text-ink-3">GPU 0 VRAM</div>
              {fmtMb(s?.gpu0?.memory_used_mb)} / {fmtMb(s?.gpu0?.memory_total_mb)}
            </div>
            <div>
              <div className="text-ink-3">in-flight student requests</div>
              {s?.inflight ?? 0}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-end gap-2">
            <Field label="Domain">
              <select className={inputCls} value={domain} onChange={(e) => setDomain(e.target.value)}>
                {(domains.data ?? []).map((d) => (
                  <option key={d.id}>{d.id}</option>
                ))}
              </select>
            </Field>
            <Field label="Mode">
              <select className={inputCls} value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="sft">SFT (distill)</option>
                <option value="dpo">DPO (preferences)</option>
              </select>
            </Field>
            <Button
              variant="primary"
              disabled={cycleActive}
              title={cycleActive ? "A training cycle is already running - follow it in the banner above" : undefined}
              onClick={() =>
                call(
                  () => api("/training/run", { method: "POST", json: { domain, mode } }),
                  (r) => `Training cycle ${r.run_id} started on ${r.rows} samples - progress is shown at the top of the page`,
                )
              }
            >
              {cycleActive ? "Training in progress…" : "Run training cycle"}
            </Button>
            {s?.phase === "failed" && <Button onClick={() => call(() => api("/training/recover", { method: "POST" }), "Student restarted")}>Recover student</Button>}
          </div>
        </Card>

        <Card title="Live log">
          <pre className="h-64 overflow-auto rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed text-ink-2">{log.join("\n") || "waiting for events…"}</pre>
        </Card>
      </div>

      <Card title="Training runs" className="mt-4">
        <div className="overflow-auto">
          <table className="tabular w-full text-xs">
            <thead className="text-ink-3">
              <tr>
                <th className="py-1 text-left font-normal">run</th>
                <th className="text-left font-normal">mode</th>
                <th className="text-right font-normal">rows</th>
                <th className="text-right font-normal">loss</th>
                <th className="text-right font-normal">runtime</th>
                <th className="pl-3 text-left font-normal">status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(runs.data ?? []).map((r) => (
                <tr key={r.run_id} className="border-t border-line text-ink-2">
                  <td className="py-1.5 font-mono text-ink">{r.run_id}</td>
                  <td>{r.mode}</td>
                  <td className="text-right">{r.rows}</td>
                  <td className="text-right">{r.result?.train_loss?.toFixed(4) ?? "–"}</td>
                  <td className="text-right">{r.result?.runtime_s ? `${Math.round(r.result.runtime_s)}s` : "–"}</td>
                  <td className="pl-3">
                    <StatusPill status={r.status === "succeeded" ? "good" : r.status === "failed" ? "critical" : "warning"} label={r.status} />
                    {r.error && <span className="ml-2 text-ink-3">{r.error}</span>}
                  </td>
                  <td className="text-right">
                    {r.status === "succeeded" && (
                      <Button variant="ghost" disabled={cycleActive} onClick={() => call(() => api("/training/rollback", { method: "POST", json: { run_id: r.run_id } }), `Now serving ${r.run_id}`)}>
                        serve
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!runs.data?.length && <div className="py-4 text-center text-xs text-ink-3">No runs yet</div>}
          <div className="mt-2">
            <Button variant="ghost" disabled={cycleActive} onClick={() => call(() => api("/training/rollback", { method: "POST", json: { run_id: null } }), "Now serving the base model")}>
              Roll back to base model
            </Button>
          </div>
        </div>
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card
          title="Synthetic factory (Mac oracle)"
          actions={
            factory.data?.running ? (
              <Button onClick={() => call(() => api("/factory/stop", { method: "POST" }), "Replay distillation stopped")}>Stop replay distillation</Button>
            ) : (
              <Button variant="primary" onClick={() => call(() => api("/factory/start", { method: "POST" }), "Replay distillation started")}>Start replay distillation</Button>
            )
          }
        >
          <div className="mb-3 text-xs text-ink-2">
            <StatusPill status={factory.data?.running ? "good" : "neutral"} label={factory.data?.running ? "pulling replay buffer" : "stopped"} />
            <span className="ml-2 text-ink-3">cursor seq {factory.data?.cursor ?? 0}</span>
          </div>
          <table className="tabular w-full text-xs">
            <thead className="text-ink-3">
              <tr>
                <th className="text-left font-normal">domain</th>
                <th className="text-right font-normal">SFT</th>
                <th className="text-right font-normal">new</th>
                <th className="text-right font-normal">DPO</th>
                <th className="text-right font-normal">held-out</th>
                <th className="text-right font-normal">accepted</th>
                <th className="text-right font-normal">rejected</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(factory.data?.domains ?? {}).map(([d, v]: [string, any]) => (
                <tr key={d} className="border-t border-line text-ink-2">
                  <td className="py-1 font-mono text-ink">{d}</td>
                  <td className="text-right">{v.sft}</td>
                  <td className="text-right">{v.new_since_train}</td>
                  <td className="text-right">{v.dpo}</td>
                  <td className="text-right">{v.heldout}</td>
                  <td className="text-right">{v.accepted}</td>
                  <td className="text-right">{v.rejected}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Seed synthesis (batch CoT generation)">
          <div className="grid gap-3">
            <Field label="Scenarios (one per line; empty = domain defaults)">
              <textarea className={`${inputCls} h-24`} value={scenarios} onChange={(e) => setScenarios(e.target.value)} />
            </Field>
            <div className="flex items-end gap-2">
              <Field label="States per scenario">
                <input type="number" min={1} max={500} className={inputCls} value={per} onChange={(e) => setPer(+e.target.value)} />
              </Field>
              <Button
                variant="primary"
                onClick={() =>
                  call(() =>
                    api("/factory/synthesize", {
                      method: "POST",
                      json: { domain, per_scenario: per, scenarios: scenarios.trim() ? scenarios.split("\n").map((x) => x.trim()).filter(Boolean) : null },
                    }),
                    (r) => `Seed synthesis ${r.job_id} started on the Mac oracle`,
                  )
                }
              >
                Synthesize for {domain}
              </Button>
            </div>
            {(factory.data?.jobs ?? []).slice(-5).reverse().map((j: any) => (
              <div key={j.id} className="flex items-center justify-between text-xs text-ink-2">
                <span className="font-mono">{j.id}</span>
                <span className="tabular">
                  {j.accepted != null ? `${j.accepted} accepted / ${j.rejected} rejected` : ""}
                </span>
                <StatusPill status={j.status === "done" ? "good" : j.status === "failed" ? "critical" : "warning"} label={j.status} />
              </div>
            ))}
          </div>
        </Card>
      </div>
      {s?.error && <Json value={{ error: s.error }} className="mt-4" />}
    </div>
  );
}
