"use client";

import { useEffect, useState } from "react";
import { RoutingMap, type Flow } from "@/components/RoutingMap";
import { Card, ErrorNote, PageTitle, StatusPill, TierBadge } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtMs, fmtPct, fmtTime } from "@/utils/format";
import { useEvents, usePoll, type BusEvent } from "@/utils/hooks";

type Threshold = { domain: string; name: string; threshold: number; triage_threshold: number; triage_enabled: boolean };

function ThresholdRow({ t, onSaved }: { t: Threshold; onSaved: () => void }) {
  const [v, setV] = useState(t.threshold);
  const [tv, setTv] = useState(t.triage_threshold);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setV(t.threshold);
    setTv(t.triage_threshold);
  }, [t.threshold, t.triage_threshold]);
  const save = async (patch: Partial<Threshold>) => {
    try {
      await api(`/routing/thresholds/${t.domain}`, { method: "PUT", json: patch });
      setErr(null);
      onSaved();
    } catch (e: any) {
      setErr(e.message);
    }
  };
  return (
    <div className="border-b border-line py-3 last:border-0">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-medium text-ink">{t.name}</div>
          <div className="font-mono text-xs text-ink-3">{t.domain}</div>
        </div>
        <label className="flex items-center gap-1.5 text-xs text-ink-2">
          <input type="checkbox" checked={t.triage_enabled} onChange={(e) => save({ triage_enabled: e.target.checked })} />
          triage tier
        </label>
      </div>
      <ErrorNote error={err} />
      {[
        { label: "Student threshold", val: v, set: setV, key: "threshold" as const, color: "var(--tier-student)" },
        { label: "Triage threshold", val: tv, set: setTv, key: "triage_threshold" as const, color: "var(--tier-triage)" },
      ].map((s) => (
        <div key={s.key} className="mt-1 grid grid-cols-[130px_1fr_48px] items-center gap-3">
          <span className="flex items-center gap-1.5 text-xs text-ink-2">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
          <input
            type="range"
            min={0.5}
            max={0.99}
            step={0.01}
            value={s.val}
            onChange={(e) => s.set(+e.target.value)}
            onMouseUp={() => save({ [s.key]: s.val })}
            onTouchEnd={() => save({ [s.key]: s.val })}
            onKeyUp={() => save({ [s.key]: s.val })}
            aria-label={`${t.name} ${s.label}`}
          />
          <span className="tabular text-right text-[13px] font-medium text-ink">{s.val.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function RoutingPage() {
  const thresholds = usePoll<Threshold[]>("/routing/thresholds", 0);
  const stats = usePoll<Record<string, Record<string, number>>>("/routing/stats", 5000);
  const escalations = usePoll<any[]>("/routing/escalations?limit=15", 4000);
  const [flows, setFlows] = useState<Flow[]>([]);
  const [feed, setFeed] = useState<BusEvent[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});

  const connected = useEvents(["routing"], (e) => {
    if (e.type !== "decision") return;
    const path: string[] = (e.data.route ?? []).map((r: any) => r.tier);
    if (e.data.halted) path.push("oracle");
    else if (e.data.tier === "oracle" && !path.includes("oracle")) path.push("oracle");
    setFlows((f) => [...f.slice(-30), { id: e.data.decision_id, path, halted: e.data.halted }]);
    setFeed((f) => [e, ...f].slice(0, 60));
    setCounts((c) => {
      const n = { ...c };
      const seq = ["obs", ...path, ...(e.data.halted ? [] : ["exec"])];
      for (let i = 0; i < seq.length - 1; i++) n[`${seq[i]}>${seq[i + 1]}`] = (n[`${seq[i]}>${seq[i + 1]}`] ?? 0) + 1;
      return n;
    });
  });

  return (
    <div>
      <PageTitle
        title="Fast-Slow Routing"
        sub="Dynamic confidence routing: Student → Triage → Oracle"
        actions={<StatusPill status={connected ? "good" : "critical"} label={connected ? "live" : "reconnecting"} />}
      />
      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <Card title="Live routing map" actions={<span className="text-xs text-ink-3">edge labels = decisions this session</span>}>
          <RoutingMap flows={flows} counts={counts} />
        </Card>
        <Card title="Confidence thresholds">
          <ErrorNote error={thresholds.error} />
          {(thresholds.data ?? []).map((t) => (
            <ThresholdRow key={t.domain} t={t} onSaved={thresholds.reload} />
          ))}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Decision feed">
          <div className="max-h-[420px] overflow-auto">
            <table className="tabular w-full text-xs">
              <thead className="sticky top-0 bg-surface text-ink-3">
                <tr>
                  <th className="py-1 text-left font-normal">time</th>
                  <th className="py-1 text-left font-normal">domain</th>
                  <th className="py-1 text-left font-normal">final tier</th>
                  <th className="py-1 text-left font-normal">action</th>
                  <th className="py-1 text-right font-normal">conf</th>
                  <th className="py-1 text-right font-normal">latency</th>
                </tr>
              </thead>
              <tbody>
                {feed.map((e) => (
                  <tr key={e.data.decision_id} className="border-t border-line text-ink-2">
                    <td className="py-1">{fmtTime(e.ts)}</td>
                    <td className="py-1 font-mono">{e.data.domain}</td>
                    <td className="py-1">{e.data.halted ? <StatusPill status="warning" label="halted → oracle" /> : <TierBadge tier={e.data.tier} />}</td>
                    <td className="py-1 font-mono text-ink">{e.data.action_label}</td>
                    <td className="py-1 text-right">{e.data.confidence?.toFixed(2)}</td>
                    <td className="py-1 text-right">{fmtMs(e.data.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!feed.length && <div className="py-6 text-center text-xs text-ink-3">Waiting for decisions (POST /api/v1/act/&lt;domain&gt;)</div>}
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="Routing totals">
            <div className="grid gap-2 sm:grid-cols-2">
              {Object.entries(stats.data ?? {}).map(([d, s]) => (
                <div key={d} className="tabular rounded-md border border-line bg-surface-2 p-3 text-xs text-ink-2">
                  <div className="mb-1 font-mono text-ink">{d}</div>
                  <div>total {s.total ?? 0}</div>
                  <div>escalation rate {fmtPct(s.total ? (s.escalations ?? 0) / s.total : null)}</div>
                  <div className="mt-1 flex gap-3">
                    {["student", "triage", "oracle"].map((t) => (
                      <span key={t} className="inline-flex items-center gap-1">
                        <span className="inline-block h-2 w-2 rounded-sm" style={{ background: `var(--tier-${t})` }} />
                        {s[`final_${t}`] ?? 0}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Oracle escalation queue (Mac)">
            <div className="max-h-[260px] space-y-1 overflow-auto text-xs">
              {(escalations.data ?? []).map((j) => (
                <div key={j.id} className="flex items-center justify-between gap-2 border-b border-line py-1 text-ink-2 last:border-0">
                  <span className="font-mono">{j.id}</span>
                  <span className="font-mono text-ink-3">{j.domain}</span>
                  <span className="truncate text-ink-3">{(j.reason ?? []).join(", ")}</span>
                  <StatusPill
                    status={j.status === "done" ? "good" : j.status === "failed" ? "critical" : "warning"}
                    label={j.status + (j.latency_ms ? ` · ${(j.latency_ms / 1000).toFixed(1)}s` : "")}
                  />
                </div>
              ))}
              {!escalations.data?.length && <div className="py-4 text-center text-ink-3">No escalations</div>}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
