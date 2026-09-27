"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Card, Empty, ErrorNote, Json, PageTitle, StatusPill, TierBadge } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtMs, fmtTime } from "@/utils/format";
import { usePoll } from "@/utils/hooks";

type Rec = {
  seq: number;
  ts: number;
  domain: string;
  session_id: string;
  state: any;
  action: any;
  confidence: number | null;
  tier: string;
  route_path: string[];
  latency_ms: number | null;
  ttft_ms: number | null;
  outcome: string;
  delta: any;
  meta: any;
};

const OUTCOME: Record<string, "good" | "critical" | "warning" | "neutral"> = { success: "good", failure: "critical", escalated: "warning", pending: "neutral" };

/** Thin timeline strip of the window, one column per record, colored by tier. */
function Timeline({ recs, size, offset, onSeek }: { recs: Rec[]; size: number; offset: number; onSeek: (o: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !recs.length) return;
    const w = c.clientWidth;
    c.width = w * devicePixelRatio;
    c.height = 40 * devicePixelRatio;
    const ctx = c.getContext("2d")!;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const css = getComputedStyle(document.documentElement);
    const col = (t: string) => css.getPropertyValue(`--tier-${t}`) || "#888";
    const crit = css.getPropertyValue("--critical");
    const n = recs.length;
    // recs are newest-first; draw oldest on the left
    for (let i = 0; i < n; i++) {
      const r = recs[n - 1 - i];
      const x = (i / n) * w;
      const bw = Math.max(w / n, 1);
      ctx.fillStyle = col(r.tier);
      ctx.fillRect(x, 12, bw, 22);
      if (r.outcome === "failure") {
        ctx.fillStyle = crit;
        ctx.fillRect(x, 2, bw, 6);
      }
    }
  }, [recs]);
  const pos = size > 1 ? 1 - offset / (size - 1) : 1;
  return (
    <div className="relative">
      <canvas
        ref={ref}
        className="h-10 w-full cursor-pointer rounded"
        onClick={(e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const frac = (e.clientX - r.left) / r.width;
          onSeek(Math.round((1 - frac) * (size - 1)));
        }}
      />
      <div className="pointer-events-none absolute top-0 h-10 w-0.5 bg-[var(--text-primary)]" style={{ left: `${pos * 100}%` }} />
    </div>
  );
}

export default function ReplayPage() {
  const stats = usePoll<any>("/replay/stats", 5000);
  const [recs, setRecs] = useState<Rec[]>([]);
  const [offset, setOffset] = useState(0);
  const [current, setCurrent] = useState<Rec | null>(null);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const size: number = stats.data?.size ?? 0;

  const loadStrip = useCallback(async () => {
    try {
      setRecs(await api<Rec[]>(`/replay?limit=1000&reverse=true`));
    } catch (e: any) {
      setErr(e.message);
    }
  }, []);
  useEffect(() => {
    loadStrip();
  }, [loadStrip]);

  useEffect(() => {
    if (!size) return;
    const t = setTimeout(async () => {
      try {
        const r = await api<any>(`/replay/scrub?offset=${offset}`);
        setCurrent(r.records[0] ?? null);
      } catch (e: any) {
        setErr(e.message);
      }
    }, 60);
    return () => clearTimeout(t);
  }, [offset, size]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setOffset((o) => (o <= 0 ? (setPlaying(false), 0) : o - 1)), 250);
    return () => clearInterval(t);
  }, [playing]);

  // The strip covers the newest 1000 records; outside that the scrubber still works.
  const stripRecs = recs;

  return (
    <div>
      <PageTitle
        title="Replay Explorer"
        sub={`Temporal scrubber over the rolling ${(stats.data?.capacity ?? 10000).toLocaleString()}-action RAM buffer`}
        actions={<Button onClick={() => { stats.reload(); loadStrip(); }}>Refresh</Button>}
      />
      <ErrorNote error={err} />
      {!size ? (
        <Empty>The replay buffer is empty. Route traffic through /api/v1/act to fill it.</Empty>
      ) : (
        <>
          <Card title="Timeline" actions={<span className="tabular text-xs text-ink-3">seq {stats.data?.oldest_seq} → {stats.data?.newest_seq}</span>}>
            <Timeline recs={stripRecs} size={Math.min(size, stripRecs.length || size)} offset={Math.min(offset, (stripRecs.length || size) - 1)} onSeek={setOffset} />
            <div className="mt-1 flex justify-between text-[11px] text-ink-3">
              <span>older</span>
              <span className="flex gap-3">
                {["student", "triage", "oracle"].map((t) => (
                  <span key={t} className="inline-flex items-center gap-1">
                    <span className="inline-block h-2 w-2 rounded-sm" style={{ background: `var(--tier-${t})` }} />
                    {t}
                  </span>
                ))}
                <span className="inline-flex items-center gap-1">
                  <span style={{ color: "var(--critical)" }}>▀</span> failure
                </span>
              </span>
              <span>newest</span>
            </div>
            <div className="mt-4 flex items-center gap-3">
              <Button onClick={() => setOffset((o) => Math.min(o + 1, size - 1))}>◀</Button>
              <Button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</Button>
              <Button onClick={() => setOffset((o) => Math.max(o - 1, 0))}>▶</Button>
              <input
                type="range"
                className="flex-1"
                min={0}
                max={size - 1}
                value={size - 1 - offset}
                onChange={(e) => setOffset(size - 1 - +e.target.value)}
                aria-label="Replay position"
              />
              <span className="tabular w-40 text-right text-xs text-ink-2">
                {offset === 0 ? "newest" : `${offset.toLocaleString()} actions ago`}
              </span>
            </div>
          </Card>

          {current && (
            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              <Card
                title={`seq ${current.seq} · ${current.domain}`}
                actions={<StatusPill status={OUTCOME[current.outcome] ?? "neutral"} label={current.outcome} />}
              >
                <div className="tabular mb-3 grid grid-cols-2 gap-2 text-xs text-ink-2 sm:grid-cols-4">
                  <div>
                    <div className="text-ink-3">time</div>
                    {fmtTime(current.ts)}
                  </div>
                  <div>
                    <div className="text-ink-3">final tier</div>
                    <TierBadge tier={current.tier} />
                  </div>
                  <div>
                    <div className="text-ink-3">confidence</div>
                    {current.confidence?.toFixed(3) ?? "–"}
                  </div>
                  <div>
                    <div className="text-ink-3">latency / TTFT</div>
                    {fmtMs(current.latency_ms)} / {fmtMs(current.ttft_ms)}
                  </div>
                </div>
                <div className="mb-2 text-xs text-ink-3">route: {current.route_path.join(" → ") || "–"} · session {current.session_id}</div>
                <div className="mb-1 text-xs font-medium text-ink-2">action_output</div>
                <Json value={current.action} className="max-h-48" />
                {current.meta?.student_action && current.tier !== "student" && (
                  <>
                    <div className="mb-1 mt-3 text-xs font-medium text-ink-2">student proposal (rejected by router)</div>
                    <Json value={current.meta.student_action} className="max-h-40" />
                  </>
                )}
                {current.meta?.oracle_action && (
                  <>
                    <div className="mb-1 mt-3 text-xs font-medium text-ink-2">oracle label</div>
                    <Json value={current.meta.oracle_action} className="max-h-40" />
                  </>
                )}
              </Card>
              <Card title="State & delta">
                <div className="mb-1 text-xs font-medium text-ink-2">state_input (scrubbed)</div>
                <Json value={current.state} className="max-h-72" />
                {current.delta && (
                  <>
                    <div className="mb-1 mt-3 text-xs font-medium text-ink-2">state delta after action</div>
                    <Json value={{ ...current.delta, post_state: undefined }} className="max-h-56" />
                  </>
                )}
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}
