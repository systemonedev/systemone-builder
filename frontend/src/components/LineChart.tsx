"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type Series = { key: string; label: string; color: string; points: { x: number; y: number | null }[] };

type Props = {
  series: Series[];
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  yMin?: number;
  yMax?: number;
  /** horizontal reference line, e.g. the 100 ms budget */
  reference?: { y: number; label: string };
  empty?: string;
};

function niceTicks(min: number, max: number, count = 4): number[] {
  if (min === max) return [min];
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Multi-series line chart: 2px lines, recessive grid, crosshair tooltip,
 *  legend for >=2 series, direct end-labels, and a table view. */
export function LineChart({ series, height = 180, yFormat = (v) => v.toFixed(2), xFormat, yMin, yMax, reference, empty = "No data yet" }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const pad = { l: 48, r: series.length > 1 ? 72 : 16, t: 10, b: 22 };
  const all = series.flatMap((s) => s.points.filter((p) => p.y != null)) as { x: number; y: number }[];
  const xs = useMemo(() => Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b), [series]);

  if (!all.length) {
    return (
      <div ref={wrap} className="flex items-center justify-center text-xs text-ink-3" style={{ height }}>
        {empty}
      </div>
    );
  }
  const x0 = Math.min(...all.map((p) => p.x));
  const x1 = Math.max(...all.map((p) => p.x));
  let lo = yMin ?? Math.min(...all.map((p) => p.y), reference?.y ?? Infinity);
  let hi = yMax ?? Math.max(...all.map((p) => p.y), reference?.y ?? -Infinity);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;
  const sx = (x: number) => pad.l + (x1 === x0 ? iw / 2 : ((x - x0) / (x1 - x0)) * iw);
  const sy = (y: number) => pad.t + ih - ((y - lo) / (hi - lo)) * ih;
  const ticks = niceTicks(lo, hi);

  const path = (s: Series) => {
    let d = "";
    let pen = false;
    for (const p of s.points) {
      if (p.y == null) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.target as SVGRectElement).getBoundingClientRect();
    const px = e.clientX - r.left + pad.l;
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(sx(xs[i]) - px) < Math.abs(sx(xs[best]) - px)) best = i;
    setHover(xs[best]);
  };
  const hoverVals = hover == null ? [] : series.map((s) => ({ s, p: s.points.find((p) => p.x === hover) }));

  return (
    <div ref={wrap} className="relative">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
        {series.length > 1 &&
          series.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3.5 rounded" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        <button className="ml-auto text-ink-3 hover:text-ink" onClick={() => setTable((t) => !t)}>
          {table ? "chart" : "table"}
        </button>
      </div>
      {table ? (
        <div className="max-h-[220px] overflow-auto">
          <table className="tabular w-full text-xs">
            <thead className="text-ink-3">
              <tr>
                <th className="py-1 text-left font-normal">x</th>
                {series.map((s) => (
                  <th key={s.key} className="py-1 text-right font-normal">{s.label}</th>
                ))}
              </tr>
            </thead>
            <tbody className="text-ink-2">
              {xs.slice(-100).reverse().map((x) => (
                <tr key={x} className="border-t border-line">
                  <td className="py-0.5">{xFormat ? xFormat(x) : x}</td>
                  {series.map((s) => {
                    const p = s.points.find((q) => q.x === x);
                    return <td key={s.key} className="py-0.5 text-right">{p?.y == null ? "–" : yFormat(p.y)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <svg width={width} height={height} className="block overflow-visible">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth={1} />
              <text x={pad.l - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="tabular" fontSize={10} fill="var(--text-muted)">
                {yFormat(t)}
              </text>
            </g>
          ))}
          {reference && reference.y >= lo && reference.y <= hi && (
            <g>
              <line x1={pad.l} x2={width - pad.r} y1={sy(reference.y)} y2={sy(reference.y)} stroke="var(--text-muted)" strokeDasharray="4 4" strokeWidth={1} />
              <text x={width - pad.r} y={sy(reference.y) - 4} textAnchor="end" fontSize={10} fill="var(--text-muted)">
                {reference.label}
              </text>
            </g>
          )}
          {series.map((s) => (
            <path key={s.key} d={path(s)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {series.length > 1 &&
            series.map((s) => {
              const last = [...s.points].reverse().find((p) => p.y != null);
              if (!last) return null;
              return (
                <text key={s.key} x={sx(last.x) + 6} y={sy(last.y!)} dominantBaseline="middle" fontSize={10} fill="var(--text-secondary)">
                  {s.label}
                </text>
              );
            })}
          {hover != null && (
            <g pointerEvents="none">
              <line x1={sx(hover)} x2={sx(hover)} y1={pad.t} y2={pad.t + ih} stroke="var(--text-muted)" strokeWidth={1} />
              {hoverVals.map(({ s, p }) =>
                p?.y == null ? null : (
                  <circle key={s.key} cx={sx(hover)} cy={sy(p.y)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
                ),
              )}
            </g>
          )}
          <rect x={pad.l} y={pad.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
      )}
      {!table && hover != null && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-lg"
          style={{ left: Math.min(sx(hover) + 12, width - 150), top: 24 }}
        >
          <div className="mb-1 text-ink-3">{xFormat ? xFormat(hover) : hover}</div>
          {hoverVals.map(({ s, p }) => (
            <div key={s.key} className="tabular flex items-center gap-2 text-ink">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              <span className="text-ink-2">{s.label}</span>
              <span className="ml-auto pl-3 font-medium">{p?.y == null ? "–" : yFormat(p.y)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
