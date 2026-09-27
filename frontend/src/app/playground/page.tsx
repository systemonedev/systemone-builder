"use client";

import { useEffect, useState } from "react";
import { Button, Card, ErrorNote, Field, inputCls, Json, PageTitle, StatusPill, TierBadge } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtMs } from "@/utils/format";
import { usePoll } from "@/utils/hooks";

const PRESETS: Record<string, { kind: string; url?: string; goal?: string; data: string }> = {
  computer_use: {
    kind: "html",
    url: "https://dashboard.internal/auth?ts=1727395200",
    goal: "Sign in as admin",
    data: `<html><body>
<h1>Sign in</h1>
<label for="u">Username</label><input id="u" value="admin">
<label for="p">Password</label><input id="p" type="password">
<button id="ember812">Sign in</button>
<a href="/forgot">Forgot password?</a>
</body></html>`,
  },
  secops: {
    kind: "suricata_eve",
    data: JSON.stringify(
      {
        timestamp: "2026-09-26T12:00:01.000000+0000",
        flow_id: 1234567890123456,
        event_type: "http",
        src_ip: "192.168.1.150",
        src_port: 51514,
        dest_ip: "10.0.0.5",
        dest_port: 80,
        proto: "TCP",
        http: { hostname: "intranet", url: "/../../../../etc/passwd", http_method: "GET", protocol: "HTTP/1.1" },
      },
      null,
      2,
    ),
  },
};

const KINDS = ["state", "html", "ax_tree", "elements", "screenshot", "suricata_eve", "syslog", "json", "text", "log"];

export default function PlaygroundPage() {
  const domains = usePoll<any[]>("/domains", 0);
  const [domain, setDomain] = useState("computer_use");
  const [kind, setKind] = useState("html");
  const [url, setUrl] = useState("");
  const [goal, setGoal] = useState("");
  const [data, setData] = useState("");
  const [session, setSession] = useState("playground");
  const [wait, setWait] = useState(false);
  const [decision, setDecision] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [post, setPost] = useState("");
  const [fb, setFb] = useState<any>(null);

  useEffect(() => {
    const p = PRESETS[domain];
    if (p) {
      setKind(p.kind);
      setUrl(p.url ?? "");
      setGoal(p.goal ?? "");
      setData(p.data);
    }
  }, [domain]);

  const parsedData = () => {
    if (["html", "syslog", "text", "screenshot", "log"].includes(kind)) {
      try {
        return kind === "log" ? JSON.parse(data) : data;
      } catch {
        return data;
      }
    }
    return JSON.parse(data);
  };

  const run = async () => {
    setBusy(true);
    setErr(null);
    setFb(null);
    try {
      const observation: any = { kind, data: parsedData() };
      if (url) observation.url = url;
      if (goal) observation.goal = goal;
      if (kind === "screenshot") observation.screenshot_b64 = data;
      setDecision(await api(`/act/${domain}`, { method: "POST", json: { observation, session_id: session, wait_for_oracle: wait } }));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const sendFeedback = async (outcome: "success" | "failure" | null) => {
    if (!decision) return;
    try {
      const body: any = { seq: decision.seq };
      if (outcome) body.outcome = outcome;
      if (post.trim()) body.post_observation = { kind, data: kind === "html" ? post : JSON.parse(post), url: url || undefined };
      setFb(await api("/feedback", { method: "POST", json: body }));
    } catch (e: any) {
      setErr(e.message);
    }
  };

  return (
    <div>
      <PageTitle title="Real-Time Playground" sub="Send an observation through extraction, scrubbing and fast-slow routing" />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Observation" actions={<Button variant="primary" onClick={run} disabled={busy}>{busy ? "Routing…" : "Act"}</Button>}>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Domain">
              <select className={inputCls} value={domain} onChange={(e) => setDomain(e.target.value)}>
                {(domains.data ?? [{ id: "computer_use" }, { id: "secops" }]).map((d) => (
                  <option key={d.id} value={d.id}>{d.id}</option>
                ))}
              </select>
            </Field>
            <Field label="Kind">
              <select className={inputCls} value={kind} onChange={(e) => setKind(e.target.value)}>
                {KINDS.map((k) => (
                  <option key={k}>{k}</option>
                ))}
              </select>
            </Field>
            <Field label="Session">
              <input className={inputCls} value={session} onChange={(e) => setSession(e.target.value)} />
            </Field>
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <Field label="URL (GUI domains)">
              <input className={inputCls} value={url} onChange={(e) => setUrl(e.target.value)} />
            </Field>
            <Field label="Goal (optional)">
              <input className={inputCls} value={goal} onChange={(e) => setGoal(e.target.value)} />
            </Field>
          </div>
          <div className="mt-3">
            <Field label={kind === "screenshot" ? "Base64 PNG screenshot" : "Data"}>
              <textarea className={`${inputCls} h-64 font-mono text-[12px]`} value={data} onChange={(e) => setData(e.target.value)} />
            </Field>
          </div>
          <label className="mt-2 flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={wait} onChange={(e) => setWait(e.target.checked)} />
            wait for the oracle if escalated (slow path)
          </label>
        </Card>

        <Card title="Decision">
          <ErrorNote error={err} />
          {!decision ? (
            <div className="text-xs text-ink-3">Press Act to route an observation.</div>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3">
                {decision.halted ? <StatusPill status="warning" label="halted — escalated to oracle" /> : <TierBadge tier={decision.tier} />}
                <span className="tabular text-xs text-ink-2">confidence {decision.confidence.toFixed(3)} / threshold {decision.threshold.toFixed(2)}</span>
                <span className="tabular text-xs text-ink-2">{fmtMs(decision.latency_ms)} end-to-end</span>
                <span className="tabular text-xs text-ink-2">prefix overlap {(decision.prefix_overlap * 100).toFixed(0)}%</span>
              </div>
              <Json value={decision.action} />
              {decision.execution && (
                <div className="text-xs text-ink-2">
                  execution handle: <span className="font-mono text-ink">{JSON.stringify(decision.execution)}</span>
                </div>
              )}
              <table className="tabular w-full text-xs">
                <thead className="text-ink-3">
                  <tr>
                    <th className="text-left font-normal">tier</th>
                    <th className="text-right font-normal">conf</th>
                    <th className="text-right font-normal">self</th>
                    <th className="text-right font-normal">tokens</th>
                    <th className="text-right font-normal">TTFT</th>
                    <th className="text-right font-normal">latency</th>
                    <th className="text-right font-normal">cached</th>
                    <th className="text-left font-normal pl-3">gates / errors</th>
                  </tr>
                </thead>
                <tbody>
                  {decision.route.map((r: any) => (
                    <tr key={r.tier} className="border-t border-line text-ink-2">
                      <td className="py-1"><TierBadge tier={r.tier} /> {r.accepted && "✓"}</td>
                      <td className="text-right">{r.confidence?.toFixed(3)}</td>
                      <td className="text-right">{r.self_reported?.toFixed(2) ?? "–"}</td>
                      <td className="text-right">{r.token_confidence?.toFixed(3) ?? "–"}</td>
                      <td className="text-right">{fmtMs(r.ttft_ms)}</td>
                      <td className="text-right">{fmtMs(r.latency_ms)}</td>
                      <td className="text-right">{r.cached_tokens ?? "–"}/{r.prompt_tokens ?? "–"}</td>
                      <td className="pl-3 text-ink-3">{[...(r.gates ?? []), ...(r.errors ?? []), r.error].filter(Boolean).join("; ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {decision.escalation_id && <div className="text-xs text-ink-2">escalation job <span className="font-mono">{decision.escalation_id}</span></div>}
              <details>
                <summary className="cursor-pointer text-xs text-ink-3">scrubbed state_input (what the student saw)</summary>
                <Json value={decision.state_input} className="mt-2 max-h-64" />
              </details>
              <div className="border-t border-line pt-3">
                <div className="mb-2 text-xs font-medium text-ink-2">Report outcome (state-delta DPO loop)</div>
                <textarea
                  className={`${inputCls} h-24 font-mono text-[12px]`}
                  placeholder="Optional post-action observation (same kind) — used to compute the state delta"
                  value={post}
                  onChange={(e) => setPost(e.target.value)}
                />
                <div className="mt-2 flex gap-2">
                  <Button onClick={() => sendFeedback("success")}>✓ Success</Button>
                  <Button variant="danger" onClick={() => sendFeedback("failure")}>✕ Failure</Button>
                  <Button variant="ghost" onClick={() => sendFeedback(null)}>Infer from delta</Button>
                </div>
                {fb && <Json value={fb} className="mt-2 max-h-56" />}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
