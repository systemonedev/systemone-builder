"use client";

import type { ReactNode } from "react";

export function Card({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 border-b border-line px-4 py-2.5">
          <h2 className="text-[13px] font-semibold tracking-wide text-ink-2">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: ReactNode; accent?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="flex items-center gap-2 text-xs text-ink-3">
        {accent && <span className="inline-block h-2 w-2 rounded-full" style={{ background: accent }} />}
        {label}
      </div>
      <div className="tabular mt-1 text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className="tabular mt-0.5 text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

const STATUS: Record<string, { color: string; icon: string }> = {
  good: { color: "var(--good)", icon: "●" },
  warning: { color: "var(--warning)", icon: "▲" },
  serious: { color: "var(--serious)", icon: "◆" },
  critical: { color: "var(--critical)", icon: "✕" },
  neutral: { color: "var(--text-muted)", icon: "○" },
};

/** Status is never color-alone: icon + label. */
export function StatusPill({ status, label }: { status: keyof typeof STATUS; label: ReactNode }) {
  const s = STATUS[status] ?? STATUS.neutral;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-2 px-2 py-0.5 text-xs text-ink-2">
      <span style={{ color: s.color }} aria-hidden>
        {s.icon}
      </span>
      {label}
    </span>
  );
}

export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-2">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: `var(--tier-${tier}, var(--text-muted))` }} />
      {tier}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "default",
  disabled,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "danger" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    default: "border border-line bg-surface-2 text-ink hover:border-ink-3",
    primary: "bg-accent text-white hover:opacity-90",
    danger: "border border-line bg-surface-2 text-ink hover:border-[var(--critical)]",
    ghost: "text-ink-2 hover:text-ink",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-[13px] font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Json({ value, className = "" }: { value: unknown; className?: string }) {
  return (
    <pre className={`overflow-auto rounded-md border border-line bg-surface-2 p-3 font-mono text-[12px] leading-relaxed text-ink-2 ${className}`}>
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed border-line p-6 text-center text-sm text-ink-3">{children}</div>;
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="mb-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-xs text-ink-2">
      <span style={{ color: "var(--critical)" }}>✕</span> {error}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-xs text-ink-3">
      <span className="mb-1 block">{label}</span>
      {children}
    </label>
  );
}

export const inputCls =
  "w-full rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent";

export function PageTitle({ title, sub, actions }: { title: string; sub?: string; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-ink">{title}</h1>
        {sub && <p className="mt-0.5 text-sm text-ink-3">{sub}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}
