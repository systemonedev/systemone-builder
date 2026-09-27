"use client";

import { useEffect, useState } from "react";
import { Button, Card, Empty, ErrorNote, inputCls, Json, PageTitle, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtTime } from "@/utils/format";
import { usePoll } from "@/utils/hooks";

const STATUS: Record<string, "good" | "warning" | "critical" | "neutral" | "serious"> = {
  pending_teacher: "neutral",
  pending_review: "warning",
  approved: "good",
  rejected: "critical",
  teacher_failed: "serious",
};

export default function DpoPage() {
  const [filter, setFilter] = useState("pending_review");
  const list = usePoll<any[]>(`/dpo/candidates?limit=200${filter ? `&status=${filter}` : ""}`, 4000);
  const stats = usePoll<any>("/dpo/stats", 5000);
  const [sel, setSel] = useState<any>(null);
  const [edit, setEdit] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!sel && list.data?.length) setSel(list.data[0]);
  }, [list.data, sel]);
  useEffect(() => {
    setEdit(sel?.chosen ? JSON.stringify(sel.chosen, null, 2) : "");
    setNote("");
    setErr(null);
  }, [sel?.id, sel?.status]);

  const act = async (approve: boolean) => {
    if (!sel) return;
    try {
      let chosen: any = undefined;
      if (approve && edit.trim()) {
        const parsed = JSON.parse(edit);
        if (JSON.stringify(parsed) !== JSON.stringify(sel.chosen)) chosen = parsed;
      }
      const res = await api<any>(`/dpo/candidates/${sel.id}/review`, {
        method: "POST",
        json: { approve, chosen, reviewer: "dashboard", note: note || undefined },
      });
      setSel(res);
      list.reload();
      stats.reload();
    } catch (e: any) {
      setErr(e.message);
    }
  };
  const retry = async () => {
    if (!sel) return;
    try {
      await api(`/dpo/candidates/${sel.id}/retry`, { method: "POST" });
      setSel(await api(`/dpo/candidates/${sel.id}`));
    } catch (e: any) {
      setErr(e.message);
    }
  };

  const counts = stats.data?.candidates ?? {};
  const editable = sel && ["pending_review", "teacher_failed"].includes(sel.status);

  return (
    <div>
      <PageTitle title="DPO Corrections Studio" sub="Human-in-the-loop oversight of failed reflex actions" />
      <div className="mb-4 flex flex-wrap gap-2">
        {["pending_review", "teacher_failed", "pending_teacher", "approved", "rejected", ""].map((s) => (
          <button
            key={s || "all"}
            onClick={() => {
              setFilter(s);
              setSel(null);
            }}
            className={`rounded-full border px-3 py-1 text-xs ${filter === s ? "border-accent text-ink" : "border-line text-ink-2"}`}
          >
            {s || "all"} {s && counts[s] != null ? `(${counts[s]})` : ""}
          </button>
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-[300px_1fr]">
        <Card title="Candidates">
          <div className="max-h-[70vh] space-y-1 overflow-auto">
            {(list.data ?? []).map((c) => (
              <button
                key={c.id}
                onClick={() => setSel(c)}
                className={`w-full rounded-md border px-3 py-2 text-left text-xs ${sel?.id === c.id ? "border-accent bg-surface-2" : "border-line"}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-ink">{c.domain}</span>
                  <StatusPill status={STATUS[c.status] ?? "neutral"} label={c.status.replace("_", " ")} />
                </div>
                <div className="mt-1 text-ink-3">
                  seq {c.seq} · {fmtTime(c.created_at)}
                </div>
                <div className="mt-0.5 truncate text-ink-2">{(c.delta?.error_signals ?? []).join("; ") || "no error signal"}</div>
              </button>
            ))}
            {!list.data?.length && <Empty>No candidates in this state.</Empty>}
          </div>
        </Card>

        {sel ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Context: initial state + failure delta">
              <div className="mb-1 text-xs font-medium text-ink-2">state_input the student acted on</div>
              <Json value={sel.state} className="max-h-72" />
              <div className="mb-1 mt-3 text-xs font-medium text-ink-2">immediate consequence (state delta)</div>
              <Json value={sel.delta} className="max-h-56" />
              {sel.note && <div className="mt-2 text-xs text-ink-2">operator note: {sel.note}</div>}
            </Card>
            <Card
              title="Correction"
              actions={<StatusPill status={STATUS[sel.status] ?? "neutral"} label={sel.status.replace("_", " ")} />}
            >
              <ErrorNote error={err} />
              <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-ink-2">
                <span style={{ color: "var(--critical)" }}>✕</span> rejected (what the student did)
              </div>
              <Json value={sel.rejected} className="max-h-40" />
              <div className="mb-1 mt-3 flex items-center gap-1.5 text-xs font-medium text-ink-2">
                <span style={{ color: "var(--good)" }}>●</span> chosen (teacher proposal — editable)
              </div>
              <textarea
                className={`${inputCls} h-44 font-mono text-[12px]`}
                value={edit}
                onChange={(e) => setEdit(e.target.value)}
                disabled={!editable}
                placeholder={sel.status === "teacher_failed" ? "Teacher failed — write the correct action_output here" : ""}
              />
              {sel.judge && (
                <div className="mt-2 text-xs text-ink-2">
                  judge score <span className="tabular font-medium text-ink">{sel.judge.score?.toFixed(2)}</span>
                  {sel.judge.issues?.length ? ` · ${sel.judge.issues.join("; ")}` : ""}
                </div>
              )}
              {sel.teacher_errors?.length > 0 && <div className="mt-1 text-xs text-ink-3">teacher: {sel.teacher_errors.join("; ")}</div>}
              {sel.cot && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-ink-3">teacher reasoning (CoT)</summary>
                  <pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-surface-2 p-2 text-xs text-ink-2">{sel.cot}</pre>
                </details>
              )}
              {editable && (
                <>
                  <input className={`${inputCls} mt-3`} placeholder="Review note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button variant="primary" onClick={() => act(true)}>Approve → DPO pair</Button>
                    <Button variant="danger" onClick={() => act(false)}>Reject</Button>
                    <Button variant="ghost" onClick={retry}>Ask teacher again</Button>
                  </div>
                </>
              )}
              {sel.reviewer && (
                <div className="mt-3 text-xs text-ink-3">
                  reviewed by {sel.reviewer}
                  {sel.edited ? " (edited)" : ""}
                  {sel.review_note ? ` — ${sel.review_note}` : ""}
                </div>
              )}
            </Card>
          </div>
        ) : (
          <Empty>Select a candidate.</Empty>
        )}
      </div>
    </div>
  );
}
