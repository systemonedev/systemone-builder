"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { authStatus, login, logout, type AuthStatus } from "@/utils/api";
import { useEvents, usePoll } from "@/utils/hooks";
import { notify } from "@/utils/notify";
import { ActivityBanner, LiveIndicator, Toasts, type Activity } from "@/components/Activity";
import { ServiceList, useServices } from "@/components/Services";

const NAV: { section: string; hint?: string; pipeline?: boolean; items: { href: string; label: string }[] }[] = [
  {
    section: "Kenning",
    items: [
      { href: "/", label: "Overview" },
      { href: "/models", label: "Models" },
      { href: "/train", label: "Train" },
      { href: "/verify", label: "Verify" },
      { href: "/try", label: "Try it" },
      { href: "/connect", label: "Connect" },
    ],
  },
  {
    section: "Legacy pipeline",
    hint: "The original generative path: vLLM student → triage → oracle",
    pipeline: true,
    items: [
      { href: "/telemetry", label: "Telemetry" },
      { href: "/routing", label: "Fast-Slow Routing" },
      { href: "/playground", label: "Playground" },
      { href: "/replay", label: "Replay Explorer" },
      { href: "/dpo", label: "DPO Corrections" },
      { href: "/training", label: "Factory & Training" },
      { href: "/eval", label: "Evaluation" },
      { href: "/workflows", label: "Prompt-to-Workflow" },
    ],
  },
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

/** Sign-in for a dashboard exposed beyond loopback. The key goes to the
 *  dashboard's gateway once and is answered with an HttpOnly session cookie;
 *  the browser never stores it. */
function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <form
      className="space-y-1.5"
      onSubmit={(e) => {
        e.preventDefault();
        setBusy(true);
        setError(null);
        login(key)
          .then(() => {
            setKey("");
            onSignedIn();
          })
          .catch((err) => setError(err.message || "sign-in failed"))
          .finally(() => setBusy(false));
      }}
    >
      <input
        type="password"
        autoComplete="current-password"
        placeholder="S1_API_KEY"
        value={key}
        onChange={(e) => setKey(e.target.value)}
        className="w-full rounded border border-line bg-surface-2 px-2 py-1 text-xs text-ink"
      />
      <button
        type="submit"
        disabled={busy || !key}
        className="w-full rounded border border-line bg-surface-2 px-2 py-1 text-xs text-ink disabled:opacity-50"
      >
        {busy ? "signing in…" : "sign in"}
      </button>
      {error && <div style={{ color: "var(--critical)" }}>{error}</div>}
    </form>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const health = usePoll<{ status: string; redis: boolean }>("/health", 5000);
  const services = useServices(5000);
  const activity = usePoll<Activity>("/activity", 3000);
  const live = useEvents(["lifecycle", "training", "eval", "factory", "workflow", "system"], (e) => {
    const d = e.data;
    if (e.channel === "lifecycle" && e.type === "phase") {
      activity.reload();
      if (d.phase === "failed") notify(`Training cycle failed: ${d.error ?? "see Factory & Training"}`, "error");
      else if (d.phase === "serving" && d.served_model) notify(`Training finished - ${d.served_model} is now serving`, "ok");
      else if (d.phase === "serving" && d.rolled_back) notify("Student restored to the previous weights", "info");
    }
    if (e.channel === "eval" && e.type === "finished")
      notify(`Evaluation finished: accuracy ${(d.accuracy * 100).toFixed(1)}% - ${d.ready_for_deployment ? "ready" : "not ready"}`, "ok");
    if (e.channel === "eval" && e.type === "failed") notify(`Evaluation failed: ${d.error}`, "error");
    if (e.channel === "training" && e.type === "auto_triggered") notify(`Automatic training started for ${d.domain}`, "info");
    if (e.channel === "workflow" && e.type === "activated") notify(`Domain ${d.domain} activated`, "ok");
    if (["eval", "factory"].includes(e.channel)) activity.reload();
  });
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const refreshAuth = () => authStatus().then(setAuth, () => setAuth(null));
  useEffect(() => {
    refreshAuth();
  }, []);
  const signedOut = auth?.required && !auth.authenticated;
  // A 401 while signed in (or with no sign-in needed) means the gateway's key
  // does not match the API's: both read S1_API_KEY from .env.
  const keyMismatch = !signedOut && [services.error, activity.error].some((x) => x && /401|API-Key/i.test(x));
  const [open, setOpen] = useState(false);
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
          {NAV.filter(
            // The legacy pages need the generative pipeline (S1_PIPELINE + compose profile "pipeline").
            (g) => !g.pipeline || services.data?.pipeline !== false || g.items.some((n) => path.startsWith(n.href)),
          ).map((group) => (
            <div key={group.section} className="mb-3">
              <div className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wider text-ink-3" title={group.hint}>
                {group.section}
              </div>
              {group.items.map((n) => {
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
            </div>
          ))}
        </nav>
        <div className={`${open ? "block" : "hidden"} space-y-2 border-t border-line px-4 py-3 text-xs text-ink-3 md:block`}>
          <ServiceList snap={ok ? services.data : null} apiUp={ok} />
          {signedOut && (
            <SignIn
              onSignedIn={() => {
                refreshAuth();
                health.reload();
                services.reload();
                activity.reload();
              }}
            />
          )}
          {auth?.required && auth.authenticated && (
            <button className="text-xs text-ink-3 underline hover:text-ink" onClick={() => logout().finally(refreshAuth)}>
              sign out
            </button>
          )}
          <ThemeToggle />
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-5 md:px-8">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <LiveIndicator live={live} pollSeconds={3} />
        </div>
        {signedOut && (
          <div className="mb-4 rounded-lg border border-line bg-surface px-4 py-2.5 text-[13px] text-ink" role="alert">
            <span style={{ color: "var(--warning)" }}>▲</span> This dashboard is reachable from the network, so it needs a
            sign-in: enter the S1_API_KEY from .env in the sidebar.
          </div>
        )}
        {keyMismatch && (
          <div className="mb-4 rounded-lg border border-line bg-surface px-4 py-2.5 text-[13px] text-ink" role="alert">
            <span style={{ color: "var(--critical)" }}>✕</span> The API rejected the dashboard&apos;s key. Both read
            S1_API_KEY from .env; after changing it, recreate both: docker compose up -d api dashboard
          </div>
        )}
        {(services.data?.warnings ?? []).map((w) => (
          <div key={w} className="mb-4 rounded-lg border border-line bg-surface px-4 py-2.5 text-[13px] text-ink" role="alert">
            <span style={{ color: "var(--warning)" }}>▲</span> {w}
          </div>
        ))}
        <ActivityBanner activity={activity.data} />
        {children}
      </main>
      <Toasts />
    </div>
  );
}
