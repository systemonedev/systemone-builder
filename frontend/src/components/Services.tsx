"use client";

import { StatusPill } from "@/components/ui";
import { usePoll } from "@/utils/hooks";

export type Service = {
  name: string;
  state: "online" | "starting" | "training" | "degraded" | "crashed" | "offline" | "stopped";
  detail?: string;
  model?: string;
  served_model?: string;
  served_models?: string[];
  container?: string;
  container_status?: string;
  exit_code?: number | null;
  url?: string;
  lifecycle_phase?: string;
};
export type ServicesSnapshot = {
  ts: number;
  operational: boolean;
  services: Service[];
  warnings?: string[];
  /** false: Kenning-only stack (S1_PIPELINE off), no student / triage / oracle */
  pipeline?: boolean;
};

const STATE: Record<string, { status: "good" | "warning" | "serious" | "critical" | "neutral"; label: string }> = {
  online: { status: "good", label: "online" },
  starting: { status: "warning", label: "starting" },
  training: { status: "serious", label: "training" },
  degraded: { status: "warning", label: "degraded" },
  crashed: { status: "critical", label: "crashed" },
  offline: { status: "critical", label: "offline" },
  stopped: { status: "neutral", label: "stopped" },
};

const LABEL: Record<string, string> = {
  api: "API",
  kenning: "Kenning · System One model",
  redis: "Redis (RAM datastore)",
  student: "Student · GPU 0",
  triage: "Triage · GPU 1",
  oracle: "Oracle · Mac",
};

export function useServices(intervalMs = 4000) {
  return usePoll<ServicesSnapshot>("/services", intervalMs);
}

const SHORT: Record<string, string> = { api: "API", kenning: "Kenning", redis: "Redis", student: "Student", triage: "Triage", oracle: "Oracle" };

const ICON: Record<string, [string, string]> = {
  good: ["●", "var(--good)"],
  warning: ["▲", "var(--warning)"],
  serious: ["◆", "var(--serious)"],
  critical: ["✕", "var(--critical)"],
  neutral: ["○", "var(--text-muted)"],
};

/** Compact list for the sidebar. */
export function ServiceList({ snap, apiUp }: { snap: ServicesSnapshot | null; apiUp: boolean }) {
  const rows: Service[] = snap?.services ?? [{ name: "api", state: apiUp ? "online" : "offline" }];
  return (
    <ul className="space-y-1">
      {rows.map((s) => {
        const st = STATE[s.state] ?? STATE.offline;
        const [icon, color] = ICON[st.status];
        return (
          <li key={s.name} className="flex items-center gap-1.5" title={s.detail}>
            <span style={{ color }} aria-hidden>
              {icon}
            </span>
            <span className="text-ink-2">{SHORT[s.name] ?? s.name}</span>
            <span className="ml-auto text-ink-3">{st.label}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Detailed card rows for the telemetry page. */
export function ServiceTable({ snap }: { snap: ServicesSnapshot | null }) {
  if (!snap) return <div className="text-xs text-ink-3">Waiting for the API…</div>;
  return (
    <div className="overflow-auto">
      <table className="w-full text-xs">
        <thead className="text-ink-3">
          <tr>
            <th className="py-1 text-left font-normal">service</th>
            <th className="py-1 text-left font-normal">status</th>
            <th className="py-1 text-left font-normal">detail</th>
            <th className="py-1 text-left font-normal">model</th>
            <th className="py-1 text-left font-normal">container</th>
          </tr>
        </thead>
        <tbody>
          {snap.services.map((s) => {
            const st = STATE[s.state] ?? STATE.offline;
            return (
              <tr key={s.name} className="border-t border-line align-top text-ink-2">
                <td className="py-1.5 pr-3 font-medium text-ink">{LABEL[s.name] ?? s.name}</td>
                <td className="py-1.5 pr-3">
                  <StatusPill status={st.status} label={st.label} />
                </td>
                <td className="max-w-[420px] py-1.5 pr-3 break-words">{s.detail ?? "–"}</td>
                <td className="max-w-[240px] truncate py-1.5 pr-3 font-mono" title={s.served_model ?? s.model}>
                  {s.served_model ?? s.model ?? "–"}
                </td>
                <td className="py-1.5 font-mono text-ink-3">
                  {s.container ? `${s.container_status}${s.exit_code ? ` (exit ${s.exit_code})` : ""}` : "–"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
