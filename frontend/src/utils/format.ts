export const fmtMs = (v: number | null | undefined) => (v == null ? "–" : v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`);
export const fmtPct = (v: number | null | undefined, digits = 0) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);
export const fmtNum = (v: number | null | undefined, digits = 2) => (v == null ? "–" : v.toFixed(digits));
export const fmtInt = (v: number | null | undefined) => (v == null ? "–" : Math.round(v).toLocaleString());
export const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
export const fmtMb = (mb: number | null | undefined) => (mb == null ? "–" : mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`);

export const TIER_COLOR: Record<string, string> = {
  student: "var(--tier-student)",
  triage: "var(--tier-triage)",
  oracle: "var(--tier-oracle)",
  human: "var(--text-secondary)",
};
