"use client";

import { useEffect, useRef, useState } from "react";

export type Flow = { id: string; path: string[]; halted: boolean };

const NODES: Record<string, { x: number; y: number; label: string; sub: string; color: string }> = {
  obs: { x: 60, y: 110, label: "Observation", sub: "extract + scrub", color: "var(--text-muted)" },
  student: { x: 250, y: 110, label: "Student", sub: "GPU 0 · vLLM", color: "var(--tier-student)" },
  triage: { x: 450, y: 110, label: "Triage", sub: "GPU 1 · vLLM", color: "var(--tier-triage)" },
  oracle: { x: 650, y: 110, label: "Oracle", sub: "Mac · Ollama", color: "var(--tier-oracle)" },
  exec: { x: 450, y: 250, label: "Execute", sub: "action out", color: "var(--text-secondary)" },
};

const EDGES: [string, string][] = [
  ["obs", "student"],
  ["student", "triage"],
  ["triage", "oracle"],
  ["student", "exec"],
  ["triage", "exec"],
  ["oracle", "exec"],
];

const edgePath = (a: string, b: string) => {
  const p = NODES[a];
  const q = NODES[b];
  if (q.y !== p.y) {
    const my = (p.y + q.y) / 2 + 20;
    return `M${p.x},${p.y + 26} C${p.x},${my} ${q.x},${my} ${q.x + (p.x < q.x ? -60 : p.x > q.x ? 60 : 0)},${q.y - 22}`;
  }
  return `M${p.x + 62},${p.y} L${q.x - 62},${q.y}`;
};

/** Live confidence-routing map: every decision animates along its route. */
export function RoutingMap({ flows, counts }: { flows: Flow[]; counts: Record<string, number> }) {
  const [active, setActive] = useState<Flow[]>([]);
  const seen = useRef(new Set<string>());
  useEffect(() => {
    const fresh = flows.filter((f) => !seen.current.has(f.id));
    if (!fresh.length) return;
    fresh.forEach((f) => seen.current.add(f.id));
    setActive((a) => [...a, ...fresh].slice(-40));
    const t = setTimeout(() => setActive((a) => a.filter((f) => !fresh.includes(f))), 1600);
    return () => clearTimeout(t);
  }, [flows]);

  const hops = (f: Flow) => {
    const seq = ["obs", ...f.path];
    if (!f.halted) seq.push("exec");
    const out: [string, string][] = [];
    for (let i = 0; i < seq.length - 1; i++) out.push([seq[i], seq[i + 1]]);
    return out;
  };

  return (
    <svg viewBox="0 0 720 300" className="h-auto w-full" role="img" aria-label="Fast-slow routing network map">
      {EDGES.map(([a, b]) => {
        const n = counts[`${a}>${b}`] ?? 0;
        const d = edgePath(a, b);
        return (
          <g key={`${a}-${b}`}>
            <path d={d} fill="none" stroke="var(--border)" strokeWidth={2} />
            <path id={`edge-${a}-${b}`} d={d} fill="none" stroke="none" />
            {n > 0 && (
              <text fontSize={10} fill="var(--text-muted)" className="tabular">
                <textPath href={`#edge-${a}-${b}`} startOffset="50%" textAnchor="middle" dy={-6}>
                  {n.toLocaleString()}
                </textPath>
              </text>
            )}
          </g>
        );
      })}
      {active.map((f) =>
        hops(f).map(([a, b], i) => (
          <circle key={`${f.id}-${i}`} r={5} fill={NODES[b === "exec" ? a : b].color} stroke="var(--surface-1)" strokeWidth={2} opacity={0}>
            <animateMotion dur="0.45s" begin={`${i * 0.4}s`} fill="freeze" path={edgePath(a, b)} />
            <animate attributeName="opacity" values="0;1;1;0" dur="0.45s" begin={`${i * 0.4}s`} fill="freeze" />
          </circle>
        )),
      )}
      {Object.entries(NODES).map(([k, n]) => (
        <g key={k}>
          <rect x={n.x - 60} y={n.y - 24} width={120} height={48} rx={8} fill="var(--surface-2)" stroke={n.color} strokeWidth={2} />
          <text x={n.x} y={n.y - 4} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-primary)">
            {n.label}
          </text>
          <text x={n.x} y={n.y + 12} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
            {n.sub}
          </text>
        </g>
      ))}
    </svg>
  );
}
