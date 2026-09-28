"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/BarChart";
import { Button, Card, Empty, ErrorNote, Field, inputCls, PageTitle, Stat, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { notify } from "@/utils/notify";
import { fmtMs, fmtPct, fmtTime } from "@/utils/format";
import { useEvents, usePoll } from "@/utils/hooks";

export default function EvalPage() {
  const domains = usePoll<any[]>("/domains", 0);
  const reports = usePoll<any[]>("/eval/reports", 5000);
  const [domain, setDomain] = useState("computer_use");
  const [target, setTarget] = useState("student");
  const [limit, setLimit] = useState(50);
  const [applyCal, setApplyCal] = useState(false);
  const [gates, setGates] = useState({ min_accuracy: 0.9, min_success_rate: 0.97, max_hallucination_ratio: 0.02, max_p95_latency_ms: 100, max_ece: 0.1, min_samples: 30 });
  const [importText, setImportText] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [rep, setRep] = useState<any>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const ds = usePoll<any>(`/datasets/${domain}`, 5000);

  useEvents(["eval"], (e) => {
    if (e.type === "progress") setProgress(`${e.data.done}/${e.data.total}`);
    if (e.type === "finished" || e.type === "failed") {
      setProgress(null);
      reports.reload();
      if (e.type === "finished") setSelId(e.data.id);
    }
  });

  useEffect(() => {
    if (!selId) return;
    api<any>(`/eval/reports/${selId}`).then(setRep).catch((e) => setErr(e.message));
  }, [selId]);

  const call = async (fn: () => Promise<any>, ok?: string | ((r: any) => string)) => {
    try {
      setErr(null);
      const r = await fn();
      if (ok) notify(typeof ok === "function" ? ok(r) : ok, "ok");
    } catch (e: any) {
      setErr(e.message);
      notify(e.message, "error");
    }
  };

  const doImport = () =>
    call(async () => {
      const text = importText.trim();
      const items = text.startsWith("[") ? JSON.parse(text) : text.split("\n").filter(Boolean).map((l) => JSON.parse(l));
      const r = await api<any>(`/eval/${domain}/heldout`, { method: "POST", json: items });
      setMsg(`Imported ${r.added} samples (${r.total} total)${r.errors.length ? `, ${r.errors.length} rejected` : ""}`);
      ds.reload();
    });

  const m = rep?.metrics;
  return (
    <div>
      <PageTitle title="Evaluation & Accuracy Suite" sub="Benchmark against real, unseen samples before deploying a reflex model" />
      <ErrorNote error={err} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Run benchmark" actions={progress && <StatusPill status="warning" label={`running ${progress}`} />}>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Domain">
              <select className={inputCls} value={domain} onChange={(e) => setDomain(e.target.value)}>
                {(domains.data ?? []).map((d) => (
                  <option key={d.id}>{d.id}</option>
                ))}
              </select>
            </Field>
            <Field label="Target">
              <select className={inputCls} value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="student">student (GPU 0)</option>
                <option value="triage">triage (GPU 1)</option>
                <option value="oracle">oracle (Mac)</option>
              </select>
            </Field>
            <Field label="Samples (max)">
              <input type="number" min={1} className={inputCls} value={limit} onChange={(e) => setLimit(+e.target.value)} />
            </Field>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Object.entries(gates).map(([k, v]) => (
              <Field key={k} label={k.replaceAll("_", " ")}>
                <input type="number" step="any" className={inputCls} value={v} onChange={(e) => setGates({ ...gates, [k]: +e.target.value })} />
              </Field>
            ))}
          </div>
          <label className="mt-3 flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={applyCal} onChange={(e) => setApplyCal(e.target.checked)} />
            push the fitted confidence calibration to the live router
          </label>
          <div className="mt-3 flex items-center gap-3">
            <Button
              variant="primary"
              onClick={() =>
                call(
                  () => api("/eval/run", { method: "POST", json: { domain, target, limit, gates, apply_calibration: applyCal } }),
                  (r) => `Evaluation ${r.report_id} started - progress is shown at the top of the page`,
                )
              }
            >
              Run evaluation
            </Button>
            <span className="text-xs text-ink-3">{ds.data?.heldout ?? 0} held-out samples for {domain}</span>
          </div>
        </Card>
        <Card title="Held-out data (real samples)">
          <Field label='JSON array or JSONL of {"observation"|"state", "expected", "acceptable"?, "tags"?}'>
            <textarea className={`${inputCls} h-32 font-mono text-[12px]`} value={importText} onChange={(e) => setImportText(e.target.value)} />
          </Field>
          <div className="mt-2 flex flex-wrap gap-2">
            <Button onClick={doImport} disabled={!importText.trim()}>Import</Button>
            <Button
              variant="ghost"
              onClick={() =>
                call(async () => {
                  const r = await api<any>(`/eval/${domain}/heldout/from-replay`, { method: "POST", json: { limit: 50 } });
                  setMsg(`Promoted ${r.added} verified replay records (${r.total} total)`);
                  ds.reload();
                })
              }
            >
              Promote verified replay traffic
            </Button>
          </div>
          {msg && <div className="mt-2 text-xs text-ink-2">{msg}</div>}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[320px_1fr]">
        <Card title="Reports">
          <div className="max-h-[70vh] space-y-1 overflow-auto">
            {(reports.data ?? []).map((r) => (
              <button
                key={r.id}
                onClick={() => setSelId(r.id)}
                className={`w-full rounded-md border px-3 py-2 text-left text-xs ${selId === r.id ? "border-accent bg-surface-2" : "border-line"}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-ink">{r.domain} · {r.target}</span>
                  {r.status === "done" ? (
                    <StatusPill status={r.ready_for_deployment ? "good" : "serious"} label={r.ready_for_deployment ? "ready" : "not ready"} />
                  ) : (
                    <StatusPill status={r.status === "failed" ? "critical" : "warning"} label={r.status} />
                  )}
                </div>
                <div className="tabular mt-1 text-ink-3">
                  {r.created_at ? fmtTime(r.created_at) : ""} {r.accuracy != null && `· acc ${fmtPct(r.accuracy, 1)} · p95 ${fmtMs(r.p95_latency_ms)}`}
                </div>
              </button>
            ))}
            {!reports.data?.length && <Empty>No reports yet.</Empty>}
          </div>
        </Card>
        {rep && m ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Stat label="Accuracy" value={fmtPct(m.accuracy, 1)} sub={`action label ${fmtPct(m.action_accuracy, 1)}`} />
              <Stat label="Success rate (autonomous)" value={fmtPct(m.success_rate, 1)} sub={`coverage ${fmtPct(m.coverage, 0)}`} />
              <Stat label="Hallucination ratio" value={fmtPct(m.hallucination_ratio, 1)} sub={`invalid ${fmtPct(m.invalid_output_ratio, 1)}`} />
              <Stat label="Latency p95" value={fmtMs(rep.latency_ms?.p95)} sub={`p50 ${fmtMs(rep.latency_ms?.p50)} · ${fmtPct(rep.latency_ms?.under_100ms)} < 100ms`} />
            </div>
            <Card
              title={`Deployment readiness · ${rep.n} samples · ${rep.model ?? rep.target}`}
              actions={<StatusPill status={rep.readiness.ready_for_deployment ? "good" : "serious"} label={rep.readiness.ready_for_deployment ? "ready for deployment" : "not ready"} />}
            >
              <div className="grid gap-1 sm:grid-cols-2">
                {Object.entries(rep.readiness.checks).map(([k, c]: [string, any]) => (
                  <div key={k} className="flex items-center gap-2 text-xs text-ink-2">
                    <span style={{ color: c.pass ? "var(--good)" : "var(--critical)" }}>{c.pass ? "●" : "✕"}</span>
                    <span className="w-36 text-ink">{k.replaceAll("_", " ")}</span>
                    <span className="tabular font-mono text-ink-3">{c.detail}</span>
                  </div>
                ))}
              </div>
            </Card>
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Latency distribution (ms)">
                <BarChart
                  bars={(rep.latency_ms?.histogram ?? []).map((h: any) => ({ label: h.le == null ? ">1000" : `≤${h.le}`, value: h.count }))}
                  yFormat={(v) => `${v} requests`}
                  highlight={(i) => (rep.latency_ms?.histogram?.[i]?.le ?? 1e9) <= 100}
                />
                <div className="tabular mt-2 text-xs text-ink-3">
                  TTFT p50 {fmtMs(rep.ttft_ms?.p50)} · p95 {fmtMs(rep.ttft_ms?.p95)} · prefix-cached tokens {fmtPct(m.prefix_cache_token_ratio)}
                </div>
              </Card>
              <Card title={`Calibration · ECE ${rep.calibration.ece?.toFixed(3) ?? "–"} · Brier ${rep.calibration.brier?.toFixed(3) ?? "–"}`}>
                <BarChart
                  bars={rep.calibration.bins.map((b: any) => ({
                    label: `${b.lo.toFixed(1)}–${b.hi.toFixed(1)}`,
                    value: b.accuracy ?? 0,
                    note: b.n ? `${b.n} samples, mean conf ${b.confidence.toFixed(2)}` : "no samples",
                  }))}
                  yFormat={(v) => `accuracy ${fmtPct(v)}`}
                />
                <div className="mt-2 text-xs text-ink-3">
                  bar = observed accuracy per confidence bin (well calibrated ≈ bin midpoint)
                  {rep.calibration.fitted && ` · fitted Platt a=${rep.calibration.fitted.a.toFixed(2)} b=${rep.calibration.fitted.b.toFixed(2)}`}
                  {rep.calibration.applied && " · applied to router"}
                </div>
              </Card>
            </div>
            <Card title="Confusion (expected → predicted)">
              <table className="tabular text-xs">
                <tbody>
                  {Object.entries(rep.confusion).map(([exp, row]: [string, any]) => (
                    <tr key={exp} className="border-t border-line">
                      <td className="py-1 pr-4 font-mono text-ink">{exp}</td>
                      {Object.entries(row).map(([p, n]: [string, any]) => (
                        <td key={p} className="px-2 text-ink-2">
                          <span className={p === exp ? "font-medium text-ink" : ""}>{p}</span>: {n}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        ) : (
          <Empty>Select a report.</Empty>
        )}
      </div>
    </div>
  );
}
