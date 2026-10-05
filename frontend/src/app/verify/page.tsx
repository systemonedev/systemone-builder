"use client";

import { useCallback, useMemo, useState } from "react";
import { JobsPanel } from "@/components/Jobs";
import { Button, Card, Empty, ErrorNote, Field, PageTitle, StatusPill, inputCls } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtMs, fmtNum, fmtPct } from "@/utils/format";
import { usePoll } from "@/utils/hooks";
import { SUITES, reportLabel, type BenchSummary, type EngineReport, type KenningStatus, type QuestionMetrics } from "@/utils/kenning";
import { notify } from "@/utils/notify";

type EngineInfo = { id: string; name: string; available: boolean; note?: string };
type RunDetail = {
  suite: { name: string; gate?: string; questions: Record<string, { type: string; instructions: string }>; items: { id: string; state: unknown; labels: Record<string, unknown> }[] };
  reports: EngineReport[];
  results: Record<string, { id: string; answers: Record<string, any> | null; error: string | null }[]>;
};

/** Accuracy of a report's questions: one value, a range for layout variants, or a list. */
function accuracies(r: EngineReport): { text: string; title: string } {
  const qs = Object.entries(r.questions).filter(([, q]) => q.n > 0 && q.accuracy != null);
  const title = qs.map(([k, q]) => `${k}: ${fmtNum(q.accuracy, 3)}`).join("\n");
  if (qs.length === 0) return { text: "–", title };
  if (r.macro_accuracy != null) return { text: `${fmtNum(r.macro_accuracy, 3)} macro, ${qs.length} tasks`, title };
  const variants = qs.every(([k]) => k.includes("@"));
  if (variants && qs.length > 1) {
    const v = qs.map(([, q]) => q.accuracy as number);
    return { text: `${fmtNum(Math.min(...v), 2)}–${fmtNum(Math.max(...v), 2)}`, title };
  }
  return { text: qs.map(([, q]) => fmtNum(q.accuracy, 2)).join(" / "), title };
}

function gateOf(r: EngineReport, gate?: string): QuestionMetrics | undefined {
  if (gate && r.questions[gate]?.gate) return r.questions[gate];
  return Object.values(r.questions).find((q) => q.gate);
}

function Cell({ r, gate }: { r: EngineReport; gate?: string }) {
  const a = accuracies(r);
  const g = gateOf(r, gate)?.gate;
  return (
    <div>
      <div className="tabular font-medium text-ink" title={a.title}>
        {a.text}
      </div>
      {g && (
        <div className="tabular text-xs text-ink-3">
          auto {fmtPct(g.automation_rate)}
          {g.false_negatives_closed > 0 && (
            <span style={{ color: "var(--critical)" }}> · {g.false_negatives_closed} threat{g.false_negatives_closed > 1 ? "s" : ""} auto-closed</span>
          )}
          {g.false_positives_acted > 0 && <span style={{ color: "var(--warning)" }}> · {g.false_positives_acted} false alarm acted</span>}
        </div>
      )}
      <div className="tabular text-xs text-ink-3">p50 {fmtMs(r.latency_ms.p50)}{r.errors ? ` · ${r.errors} errors` : ""}</div>
    </div>
  );
}

function Scoreboard({ runs }: { runs: BenchSummary[] }) {
  // latest report per (suite, engine/model)
  const { cols, cells, gates } = useMemo(() => {
    const cells: Record<string, Record<string, EngineReport>> = {};
    const gates: Record<string, string | undefined> = {};
    const cols: string[] = [];
    for (const run of [...runs].sort((a, b) => a.ts - b.ts)) {
      for (const r of run.reports) {
        const label = reportLabel(r);
        if (!cols.includes(label)) cols.push(label);
        (cells[run.suite] ??= {})[label] = r;
        gates[run.suite] = run.gate;
      }
    }
    return { cols, cells, gates };
  }, [runs]);
  const suites = SUITES.filter((s) => cells[s.name]).concat(
    Object.keys(cells)
      .filter((name) => !SUITES.find((s) => s.name === name))
      .map((name) => ({ id: name, name, label: name, note: "" })),
  );
  if (suites.length === 0) return <Empty>No benchmark runs yet.</Empty>;
  return (
    <div className="overflow-auto">
      <table className="w-full text-left text-[13px]">
        <thead className="text-xs text-ink-3">
          <tr>
            <th className="pb-2 pr-4 font-normal">suite</th>
            {cols.map((c) => (
              <th key={c} className="pb-2 pr-4 font-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {suites.map((s) => (
            <tr key={s.name} className="border-t border-line align-top">
              <td className="py-2 pr-4">
                <div className="font-medium text-ink">{s.label}</div>
                <div className="max-w-[16rem] text-xs text-ink-3">{s.note}</div>
              </td>
              {cols.map((c) => (
                <td key={c} className="py-2 pr-4">
                  {cells[s.name]?.[c] ? <Cell r={cells[s.name][c]} gate={gates[s.name]} /> : <span className="text-ink-3">–</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">
        Latest run per suite and model. Accuracy at 0.5; &quot;auto&quot; is the share of items decided without a person (p ≥ 0.9 or ≤ 0.1). A threat
        auto-closed is a positive item the model was sure was negative: the costly mistake.
      </p>
    </div>
  );
}

function truthBit(v: unknown): number | null {
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v === 0 || v === 1) return v;
  if (typeof v === "string" && ["yes", "true", "1", "no", "false", "0"].includes(v.toLowerCase())) return ["yes", "true", "1"].includes(v.toLowerCase()) ? 1 : 0;
  return null;
}

function predicted(a: any): { text: string; value: unknown } {
  if (!a) return { text: "–", value: null };
  if (a.type === "noul") return { text: fmtNum(a.noul, 2), value: a.noul };
  if (a.type === "choice") return { text: `${a.choice} (${fmtNum(a.confidence, 2)})`, value: a.choice };
  return { text: fmtNum(a.score, 2), value: Math.round(a.score) };
}

function correct(label: unknown, a: any): boolean | null {
  if (!a || label == null) return null;
  if (a.type === "noul") {
    const t = truthBit(label);
    return t == null ? null : (a.noul >= 0.5 ? 1 : 0) === t;
  }
  if (a.type === "choice") return String(a.choice) === String(label);
  return Math.round(a.score) === Number(label);
}

function RunExplorer({ file }: { file: string }) {
  const run = usePoll<RunDetail>(`/kenning/bench/results/${encodeURIComponent(file)}`, 0);
  const [onlyWrong, setOnlyWrong] = useState(true);
  const d = run.data;
  const qids = d ? Object.keys(d.suite.questions) : [];
  const [q, setQ] = useState<string>("");
  const qid = q || d?.suite.gate || qids[0] || "";
  if (!d) return run.error ? <ErrorNote error={run.error} /> : <Empty>Loading…</Empty>;
  // runs written before per-item results were stored by engine have nothing to show here
  const engines = d.results && !Array.isArray(d.results) ? Object.keys(d.results) : [];
  const byEngine = Object.fromEntries(engines.map((e) => [e, Object.fromEntries(d.results[e].map((r) => [r.id, r]))]));
  const rows = d.suite.items
    .filter((it) => it.labels[qid] !== undefined)
    .map((it) => ({ it, marks: engines.map((e) => correct(it.labels[qid], byEngine[e][it.id]?.answers?.[qid])) }))
    .filter((r) => !onlyWrong || r.marks.some((m) => m === false));
  const gate = d.suite.questions[qid]?.type === "noul";
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-ink-3">
        <select className={`${inputCls} w-auto`} value={qid} onChange={(e) => setQ(e.target.value)}>
          {qids.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={onlyWrong} onChange={(e) => setOnlyWrong(e.target.checked)} /> only items some engine got wrong
        </label>
        <span>{rows.length} items</span>
      </div>
      <div className="max-h-[30rem] overflow-auto">
        <table className="w-full text-left text-[12px]">
          <thead className="sticky top-0 bg-surface text-xs text-ink-3">
            <tr>
              <th className="pb-1 pr-3 font-normal">item</th>
              <th className="pb-1 pr-3 font-normal">label</th>
              {engines.map((e) => (
                <th key={e} className="pb-1 pr-3 font-normal">
                  {reportLabel(d.reports.find((r) => r.engine === e) ?? ({ engine: e } as EngineReport))}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ it, marks }) => (
              <tr key={it.id} className="border-t border-line align-top">
                <td className="max-w-[28rem] py-1.5 pr-3">
                  <div className="font-mono text-ink-2">{it.id}</div>
                  <div className="line-clamp-2 text-ink-3" title={JSON.stringify(it.state).slice(0, 2000)}>
                    {JSON.stringify(it.state).slice(0, 220)}
                  </div>
                </td>
                <td className="py-1.5 pr-3 text-ink-2">{String(it.labels[qid])}</td>
                {engines.map((e, i) => {
                  const a = byEngine[e][it.id]?.answers?.[qid];
                  const p = predicted(a);
                  // a sure "no" on a positive item is the dangerous miss
                  const closedThreat = gate && truthBit(it.labels[qid]) === 1 && a?.noul <= 0.1;
                  return (
                    <td key={e} className="tabular py-1.5 pr-3" style={marks[i] === false ? { color: closedThreat ? "var(--critical)" : "var(--warning)" } : undefined}>
                      {byEngine[e][it.id]?.error ? "error" : p.text}
                      {marks[i] === false && (closedThreat ? " ✕ auto-closed" : " ✕")}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RunCard({ engines, status }: { engines: EngineInfo[]; status: KenningStatus | null }) {
  const [suites, setSuites] = useState<string[]>(["general"]);
  const [picked, setPicked] = useState<string[]>(["kenning"]);
  const [n, setN] = useState(50);
  const [paid, setPaid] = useState(false);
  const clef = usePoll<{ online: boolean }>("/kenning/clef", 15000);
  const options: EngineInfo[] = [
    { id: "kenning", name: `Kenning (${status?.server.model ?? "offline"})`, available: !!status?.server.online, note: "the active model" },
    { id: "clef", name: "Cloudflare Clef (local)", available: !!clef.data?.online, note: clef.data?.online ? "~225 ms per item" : "docker compose --profile clef up -d clef" },
    ...engines.filter((e) => e.id === "jev"),
    ...(status?.pipeline ? [{ id: "local", name: "Local LLM read-out (triage)", available: true, note: "label-token probabilities" }] : []),
  ];
  const toggle = (list: string[], set: (x: string[]) => void, id: string) => set(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);
  const jev = picked.includes("jev");
  return (
    <Card title="Run a benchmark">
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-xs text-ink-3">Suites</div>
          <div className="grid gap-1">
            {SUITES.map((s) => (
              <label key={s.id} className="flex items-start gap-2 text-[13px] text-ink-2">
                <input type="checkbox" className="mt-1" checked={suites.includes(s.id)} onChange={() => toggle(suites, setSuites, s.id)} />
                <span>
                  {s.label} <span className="text-xs text-ink-3">{s.note}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1 text-xs text-ink-3">Engines</div>
          <div className="grid gap-1">
            {options.map((e) => (
              <label key={e.id} className={`flex items-start gap-2 text-[13px] ${e.available ? "text-ink-2" : "text-ink-3"}`}>
                <input type="checkbox" className="mt-1" disabled={!e.available} checked={picked.includes(e.id)} onChange={() => toggle(picked, setPicked, e.id)} />
                <span>
                  {e.name} <span className="text-xs text-ink-3">{e.note}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
        {jev && (
          <label className="flex items-start gap-2 text-xs" style={{ color: "var(--warning)" }}>
            <input type="checkbox" className="mt-0.5" checked={paid} onChange={(e) => setPaid(e.target.checked)} />
            Each item is a paid TypeSafe request under my TypeSafe agreement. Results are for comparison only and are never used for training.
          </label>
        )}
        <div className="flex items-end gap-3">
          <Field label="Items (phishing, layouts)">
            <input type="number" min={5} max={1000} className={`${inputCls} w-28`} value={n} onChange={(e) => setN(Number(e.target.value))} />
          </Field>
          <Button
            variant="primary"
            disabled={!suites.length || !picked.length || (jev && !paid)}
            onClick={async () => {
              try {
                await api("/kenning/jobs", { method: "POST", json: { kind: "bench", params: { suites, engines: picked, n, confirm_paid: jev && paid } } });
                notify("Benchmark started", "info");
              } catch (e: any) {
                notify(e.message, "error");
              }
            }}
          >
            Run
          </Button>
        </div>
        <p className="text-xs text-ink-3">
          Benchmarks the active Kenning model; switch models on the Models page. Suites are never used for training.
        </p>
      </div>
    </Card>
  );
}

export default function VerifyPage() {
  const runs = usePoll<BenchSummary[]>("/kenning/bench/results", 30000);
  const engines = usePoll<EngineInfo[]>("/systemone/engines", 30000);
  const status = usePoll<KenningStatus>("/kenning/status", 15000);
  const [sel, setSel] = useState<string | null>(null);
  const reloadRuns = runs.reload;
  const onFinished = useCallback(() => reloadRuns(), [reloadRuns]);
  const list = runs.data ?? [];
  return (
    <div>
      <PageTitle
        title="Verify"
        sub="Benchmark Kenning against other System One models on held-out suites: accuracy, calibration, and what it would decide on its own."
      />
      <ErrorNote error={runs.error} />
      <div className="space-y-4">
        <Card title="Scoreboard" actions={status.data?.server.model && <StatusPill status="good" label={`serving ${status.data.server.model}`} />}>
          {!runs.data ? <Empty>Loading…</Empty> : <Scoreboard runs={list} />}
        </Card>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
          <RunCard engines={engines.data ?? []} status={status.data} />
          <JobsPanel kinds={["bench"]} onFinished={onFinished} />
        </div>
        <Card title="Runs">
          {list.length === 0 ? (
            <Empty>No runs yet.</Empty>
          ) : (
            <div className="grid gap-4 xl:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
              <ul className="max-h-[34rem] space-y-1 overflow-auto">
                {list.map((r) => (
                  <li key={r.file}>
                    <button
                      onClick={() => setSel(r.file)}
                      className={`w-full rounded-md px-2 py-1.5 text-left text-[13px] ${sel === r.file ? "bg-surface-2" : "hover:bg-surface-2"}`}
                    >
                      <div className="flex justify-between gap-2">
                        <span className="font-medium text-ink">{SUITES.find((s) => s.name === r.suite)?.label ?? r.suite}</span>
                        <span className="text-xs text-ink-3">{new Date(r.ts * 1000).toLocaleString([], { hour12: false })}</span>
                      </div>
                      <div className="truncate text-xs text-ink-3">
                        {r.reports.map((x) => `${reportLabel(x)} ${accuracies(x).text}`).join(" · ")}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
              <div className="min-w-0">{sel ? <RunExplorer key={sel} file={sel} /> : <Empty>Pick a run to see its items and mistakes.</Empty>}</div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
