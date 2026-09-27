"use client";

import { useState } from "react";

/** Single-series bar chart: thin bars, rounded data-end anchored to the
 *  baseline, 2px gaps, hover tooltip per bar. */
export function BarChart({
  bars,
  height = 160,
  color = "var(--accent)",
  yFormat = (v: number) => String(v),
  highlight,
}: {
  bars: { label: string; value: number; note?: string }[];
  height?: number;
  color?: string;
  yFormat?: (v: number) => string;
  highlight?: (i: number) => boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1e-9, ...bars.map((b) => b.value));
  return (
    <div className="relative">
      <div className="flex items-end gap-[2px] border-b border-line" style={{ height }}>
        {bars.map((b, i) => (
          <div
            key={b.label}
            className="flex h-full flex-1 items-end"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          >
            <div
              className="w-full rounded-t"
              style={{
                height: `${(b.value / max) * 100}%`,
                minHeight: b.value > 0 ? 2 : 0,
                background: color,
                opacity: highlight && !highlight(i) ? 0.45 : hover === null || hover === i ? 1 : 0.7,
              }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex gap-[2px] text-[10px] text-ink-3">
        {bars.map((b, i) => (
          <div key={b.label} className="flex-1 truncate text-center">
            {i % Math.ceil(bars.length / 8) === 0 ? b.label : ""}
          </div>
        ))}
      </div>
      {hover !== null && (
        <div
          className="pointer-events-none absolute top-0 z-10 rounded-md border border-line bg-surface px-2 py-1 text-xs shadow-lg"
          style={{ left: `min(${((hover + 0.5) / bars.length) * 100}%, calc(100% - 120px))` }}
        >
          <div className="text-ink-3">{bars[hover].label}</div>
          <div className="tabular font-medium text-ink">{yFormat(bars[hover].value)}</div>
          {bars[hover].note && <div className="text-ink-3">{bars[hover].note}</div>}
        </div>
      )}
    </div>
  );
}
