"use client";

import { useEffect, useState } from "react";
import { dismiss, subscribeToasts, type Toast } from "@/utils/notify";
import type { LiveStatus } from "@/utils/hooks";

type Item = {
  kind: string;
  state: string;
  title: string;
  phase?: string;
  domain?: string;
  mode?: string;
  started_at?: number;
  finished_at?: number;
  step?: number;
  max_steps?: number;
  loss?: number;
  error?: string;
  done?: number;
  total?: number;
  accepted?: number;
  rejected?: number;
  target?: string;
};
export type Activity = { ts: number; busy: boolean; items: Item[]; factory_running: boolean };

const PHASE_TEXT: Record<string, string> = {
  draining: "draining in-flight requests",
  paused: "stopping the vLLM student",
  flushing: "waiting for GPU 0 VRAM to be released",
  training: "Unsloth QLoRA training on GPU 0",
  reloading: "loading the new weights into vLLM",
  serving: "serving",
  failed: "failed (rolled back to the previous weights)",
};

function elapsed(from?: number, to?: number) {
  if (!from) return "";
  const s = Math.max(0, Math.round((to ?? Date.now() / 1000) - from));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function describe(i: Item): string {
  if (i.kind === "training") {
    if (i.state === "running") {
      const step = i.step != null ? ` · step ${i.step}${i.max_steps ? `/${i.max_steps}` : ""}` : "";
      const loss = i.loss != null ? ` · loss ${i.loss.toFixed(3)}` : "";
      return `${PHASE_TEXT[i.phase ?? ""] ?? i.phase}${step}${loss} · ${elapsed(i.started_at)}`;
    }
    if (i.state === "succeeded") return `finished in ${elapsed(i.started_at, i.finished_at)} · loss ${i.loss?.toFixed(3) ?? "–"} · new model is live`;
    return i.error ?? "failed";
  }
  if (i.kind === "evaluation") return `${i.target} on ${i.domain} · ${i.done}/${i.total} samples · ${elapsed(i.started_at)}`;
  if (i.kind === "synthesis") return `${i.domain} · ${i.accepted ?? 0} accepted / ${i.rejected ?? 0} rejected so far`;
  return "";
}

const STYLE: Record<string, [string, string]> = {
  running: ["var(--warning)", "▲"],
  failed: ["var(--critical)", "✕"],
  succeeded: ["var(--good)", "●"],
};

/** Global "what is happening right now" strip shown on every page. */
export function ActivityBanner({ activity }: { activity: Activity | null }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((x) => x + 1), 1000); // keep elapsed timers moving
    return () => clearInterval(t);
  }, []);
  if (!activity) return null;
  if (!activity.items.length) {
    return <div className="mb-4 text-xs text-ink-3">○ Idle - no training, synthesis or evaluation running.</div>;
  }
  return (
    <div className="mb-4 space-y-2">
      {activity.items.map((i) => {
        const [color, icon] = STYLE[i.state] ?? STYLE.running;
        const pct = i.kind === "evaluation" && i.total ? (i.done ?? 0) / i.total : i.max_steps && i.step ? i.step / i.max_steps : null;
        return (
          <div key={i.title} className="rounded-lg border border-line bg-surface px-4 py-2.5" role="status">
            <div className="flex flex-wrap items-center gap-2 text-[13px]">
              <span style={{ color }} aria-hidden className={i.state === "running" ? "animate-pulse" : ""}>
                {icon}
              </span>
              <span className="font-medium text-ink">{i.title}</span>
              {i.domain && <span className="font-mono text-xs text-ink-3">{i.domain}{i.mode ? ` · ${i.mode}` : ""}</span>}
              <span className="text-xs text-ink-2">{describe(i)}</span>
            </div>
            {pct != null && i.state === "running" && (
              <div className="mt-2 h-1.5 overflow-hidden rounded bg-surface-2">
                <div className="h-full rounded bg-accent transition-all" style={{ width: `${Math.round(pct * 100)}%` }} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Explains the live stream: what it is, and why it is down. */
export function LiveIndicator({ live, pollSeconds }: { live: LiveStatus; pollSeconds: number }) {
  const tip = live.connected
    ? "Live stream (WebSocket) connected: charts, routing map and training logs update in real time."
    : `Live stream (WebSocket) not connected: ${live.reason}. Pages still auto-refresh every ${pollSeconds}s.`;
  return (
    <div className="flex items-center gap-1.5 text-xs" title={tip}>
      <span style={{ color: live.connected ? "var(--good)" : "var(--warning)" }} aria-hidden>
        {live.connected ? "●" : "▲"}
      </span>
      <span className="text-ink-2">{live.connected ? "Live updates on" : "Live updates off"}</span>
      {!live.connected && live.reason && <span className="text-ink-3">- {live.reason}</span>}
    </div>
  );
}

export function Toasts() {
  const [items, setItems] = useState<Toast[]>([]);
  useEffect(() => subscribeToasts(setItems), []);
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[360px] max-w-[calc(100vw-32px)] flex-col gap-2">
      {items.map((t) => (
        <div
          key={t.id}
          role="status"
          className="pointer-events-auto flex items-start gap-2 rounded-lg border border-line bg-surface px-3 py-2.5 text-[13px] text-ink shadow-lg"
        >
          <span style={{ color: t.kind === "error" ? "var(--critical)" : t.kind === "info" ? "var(--accent)" : "var(--good)" }} aria-hidden>
            {t.kind === "error" ? "✕" : t.kind === "info" ? "ℹ" : "●"}
          </span>
          <span className="flex-1">{t.text}</span>
          <button className="text-ink-3 hover:text-ink" onClick={() => dismiss(t.id)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
