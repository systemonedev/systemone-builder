"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { apiBase, apiKey, setApiKey } from "@/utils/api";
import { usePoll } from "@/utils/hooks";

const NAV = [
  { href: "/", label: "Telemetry" },
  { href: "/routing", label: "Fast-Slow Routing" },
  { href: "/playground", label: "Playground" },
  { href: "/replay", label: "Replay Explorer" },
  { href: "/dpo", label: "DPO Corrections" },
  { href: "/training", label: "Factory & Training" },
  { href: "/eval", label: "Evaluation" },
  { href: "/workflows", label: "Prompt-to-Workflow" },
];

function ThemeToggle() {
  const [theme, setTheme] = useState<string>("system");
  useEffect(() => {
    try {
      const t = localStorage.getItem("s1.theme") ?? "system";
      setTheme(t);
    } catch {}
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("s1.theme", theme);
    } catch {}
  }, [theme]);
  return (
    <select value={theme} onChange={(e) => setTheme(e.target.value)} className="rounded border border-line bg-surface-2 px-1.5 py-1 text-xs text-ink-2">
      <option value="system">system</option>
      <option value="dark">dark</option>
      <option value="light">light</option>
    </select>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const health = usePoll<{ status: string; redis: boolean }>("/health", 5000);
  const [key, setKey] = useState("");
  const [open, setOpen] = useState(false);
  useEffect(() => setKey(apiKey() ?? ""), []);
  const ok = health.data?.status === "ok";
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="border-b border-line bg-surface md:sticky md:top-0 md:h-screen md:w-56 md:flex-none md:border-b-0 md:border-r">
        <div className="flex items-center justify-between px-4 py-4">
          <Link href="/" className="font-mono text-[15px] font-semibold text-ink">
            systemone<span className="text-accent">·</span>builder
          </Link>
          <button className="text-xs text-ink-3 md:hidden" onClick={() => setOpen((o) => !o)}>
            menu
          </button>
        </div>
        <nav className={`${open ? "block" : "hidden"} px-2 pb-3 md:block`}>
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                onClick={() => setOpen(false)}
                className={`block rounded-md px-3 py-1.5 text-[13px] ${active ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:text-ink"}`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className={`${open ? "block" : "hidden"} space-y-2 border-t border-line px-4 py-3 text-xs text-ink-3 md:block`}>
          <div className="flex items-center gap-1.5">
            <span style={{ color: ok ? "var(--good)" : "var(--critical)" }}>{ok ? "●" : "✕"}</span>
            API {ok ? "online" : "offline"}
          </div>
          <div className="truncate font-mono" title={apiBase()}>
            {apiBase().replace(/^https?:\/\//, "")}
          </div>
          <input
            type="password"
            placeholder="X-API-Key"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            onBlur={() => {
              setApiKey(key);
              health.reload();
            }}
            className="w-full rounded border border-line bg-surface-2 px-2 py-1 text-xs text-ink"
          />
          <ThemeToggle />
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-5 md:px-8">{children}</main>
    </div>
  );
}
