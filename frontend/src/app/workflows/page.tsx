"use client";

import { useState } from "react";
import { Button, Card, Empty, ErrorNote, Field, inputCls, PageTitle, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtTime } from "@/utils/format";
import { usePoll } from "@/utils/hooks";

const EXAMPLES = [
  "Watch sshd auth logs on our bastion hosts. Block IPs that brute-force root or service accounts, allow our monitoring subnet 10.20.0.0/16, escalate anything touching domain admins.",
  "Drive our internal expense tool: open pending reports, approve receipts under $50 that match policy, flag anything else for a human.",
  "Triage GitHub webhook events for our monorepo: label incoming issues as bug/feature/question and route security reports to the security team.",
];

export default function WorkflowsPage() {
  const drafts = usePoll<any[]>("/workflows/drafts", 0);
  const domains = usePoll<any[]>("/domains", 0);
  const [prompt, setPrompt] = useState("");
  const [kind, setKind] = useState("");
  const [draft, setDraft] = useState<any>(null);
  const [specText, setSpecText] = useState("");
  const [bootstrap, setBootstrap] = useState(true);
  const [per, setPer] = useState(20);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState<any>(null);

  const select = (d: any) => {
    setDraft(d);
    setSpecText(d?.spec ? JSON.stringify(d.spec, null, 2) : "");
    setDone(null);
  };

  const generate = async () => {
    setBusy(true);
    setErr(null);
    try {
      const d = await api<any>("/workflows/generate", { method: "POST", json: { prompt, kind: kind || null } });
      select(d);
      drafts.reload();
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const activate = async () => {
    setErr(null);
    try {
      const spec = JSON.parse(specText);
      const res = await api<any>("/workflows/activate", { method: "POST", json: { spec, bootstrap, per_scenario: per } });
      setDone(res);
      domains.reload();
    } catch (e: any) {
      setErr(e.message);
    }
  };

  let parsed: any = null;
  try {
    parsed = specText ? JSON.parse(specText) : null;
  } catch {
    parsed = null;
  }

  return (
    <div>
      <PageTitle title="Prompt-to-Workflow Engine" sub="Describe a specialization — the teacher designs extractors, schemas and factory parameters" />
      <ErrorNote error={err} />
      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <Card title="Specialization prompt" actions={<Button variant="primary" onClick={generate} disabled={busy || prompt.length < 10}>{busy ? "Teacher is designing…" : "Generate workspace"}</Button>}>
          <textarea
            className={`${inputCls} h-32`}
            placeholder="e.g. Triage Suricata alerts for our web tier and block scanners in under 100ms…"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <select className="rounded border border-line bg-surface-2 px-2 py-1 text-xs text-ink-2" value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="">auto-detect template</option>
              <option value="computer_use">computer_use template</option>
              <option value="secops">secops template</option>
              <option value="custom">custom (no template)</option>
            </select>
            {EXAMPLES.map((x, i) => (
              <button key={i} className="rounded-full border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink" onClick={() => setPrompt(x)}>
                example {i + 1}
              </button>
            ))}
          </div>
        </Card>
        <Card title="Active domains">
          <div className="space-y-1.5 text-xs">
            {(domains.data ?? []).map((d) => (
              <div key={d.id} className="flex items-center justify-between">
                <span className="font-mono text-ink">{d.id}</span>
                <span className="text-ink-3">
                  {d.source} · τ {d.threshold.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[280px_1fr]">
        <Card title="Drafts">
          <div className="max-h-[60vh] space-y-1 overflow-auto">
            {(drafts.data ?? []).map((d) => (
              <button
                key={d.id}
                onClick={() => select(d)}
                className={`w-full rounded-md border px-3 py-2 text-left text-xs ${draft?.id === d.id ? "border-accent bg-surface-2" : "border-line"}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-ink">{d.spec?.id ?? d.kind}</span>
                  <StatusPill status={d.status === "ready" ? "good" : "critical"} label={d.status} />
                </div>
                <div className="mt-1 line-clamp-2 text-ink-3">{d.prompt}</div>
                <div className="mt-0.5 text-ink-3">{fmtTime(d.created_at)}</div>
              </button>
            ))}
            {!drafts.data?.length && <Empty>No drafts yet.</Empty>}
          </div>
        </Card>
        {draft ? (
          <Card
            title={`Draft ${draft.id} · template ${draft.template ?? "none"}`}
            actions={<StatusPill status={draft.status === "ready" ? "good" : "critical"} label={draft.status} />}
          >
            {draft.attempts?.some((a: any) => a.errors.length) && (
              <details className="mb-3 text-xs text-ink-2">
                <summary className="cursor-pointer text-ink-3">validation / repair attempts ({draft.attempts.length})</summary>
                {draft.attempts.map((a: any) => (
                  <div key={a.attempt} className="mt-1">
                    attempt {a.attempt}: {a.errors.length ? a.errors.join("; ") : "valid"}
                  </div>
                ))}
              </details>
            )}
            {parsed && (
              <div className="mb-3 grid gap-3 text-xs text-ink-2 sm:grid-cols-3">
                <div>
                  <div className="text-ink-3">actions</div>
                  <div className="font-mono text-ink">{(parsed.supported_actions ?? []).join(", ")}</div>
                </div>
                <div>
                  <div className="text-ink-3">extractor · threshold</div>
                  {parsed.extractor?.type} · {parsed.threshold}
                </div>
                <div>
                  <div className="text-ink-3">factory scenarios</div>
                  {(parsed.factory?.scenarios ?? []).length}
                </div>
              </div>
            )}
            <Field label="Domain specification (editable)">
              <textarea className={`${inputCls} h-[420px] font-mono text-[12px]`} value={specText} onChange={(e) => setSpecText(e.target.value)} />
            </Field>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-ink-2">
                <input type="checkbox" checked={bootstrap} onChange={(e) => setBootstrap(e.target.checked)} />
                bootstrap: synthesize
                <input type="number" min={1} max={500} value={per} onChange={(e) => setPer(+e.target.value)} className="w-16 rounded border border-line bg-surface-2 px-1.5 py-0.5 text-ink" />
                seed trajectories per scenario
              </label>
              <Button variant="primary" onClick={activate} disabled={!parsed}>
                Activate domain
              </Button>
            </div>
            {done && (
              <div className="mt-3 text-xs text-ink-2">
                <span style={{ color: "var(--good)" }}>●</span> Domain <span className="font-mono text-ink">{done.domain}</span> is live.
                {done.bootstrap_job && (
                  <>
                    {" "}Factory job <span className="font-mono">{done.bootstrap_job}</span> is generating trajectories — follow it on Factory & Training.
                  </>
                )}
              </div>
            )}
          </Card>
        ) : (
          <Empty>Generate or select a draft.</Empty>
        )}
      </div>
    </div>
  );
}
