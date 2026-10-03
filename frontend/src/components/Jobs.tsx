"use client";

import { useEffect, useRef, useState } from "react";
import { Button, Card, Empty, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { usePoll } from "@/utils/hooks";
import { notify } from "@/utils/notify";
import type { Job, JobKind } from "@/utils/kenning";

const STATUS: Record<string, "good" | "warning" | "serious" | "critical" | "neutral"> = {
  queued: "neutral",
  running: "warning",
  succeeded: "good",
  failed: "critical",
  cancelled: "neutral",
  interrupted: "serious",
};

const KIND: Record<JobKind, string> = { data: "Build data", label: "Label (Clef)", train: "Train", bench: "Benchmark" };

export function jobTitle(j: Job): string {
  const p = j.params;
  if (j.kind === "data") return `${KIND.data} · ${p.name}`;
  if (j.kind === "label") return `${KIND.label} · ${p.dataset} → ${p.name}`;
  if (j.kind === "train") return `${KIND.train} · ${p.name} on ${p.dataset}`;
  return `${KIND.bench} · ${(p.suites ?? []).join(", ")} · ${(p.engines ?? []).join(" vs ")}`;
}

function elapsed(j: Job): string {
  if (!j.started) return "";
  const s = Math.round((j.finished ?? Date.now() / 1000) - j.started);
  return s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
}

/** Jobs of the given kinds: the running one with a live log, and recent ones. */
export function JobsPanel({ kinds, onFinished }: { kinds: JobKind[]; onFinished?: (j: Job) => void }) {
  const list = usePoll<{ busy: boolean; jobs: Job[] }>("/kenning/jobs?limit=50", 3000);
  const jobs = (list.data?.jobs ?? []).filter((j) => kinds.includes(j.kind));
  const [sel, setSel] = useState<string | null>(null);
  const shown = jobs.find((j) => j.id === sel) ?? jobs.find((j) => j.status === "running") ?? jobs[0];
  const detail = usePoll<Job>(shown ? `/kenning/jobs/${shown.id}?tail=400` : null, shown?.status === "running" ? 2000 : 0);
  const job = detail.data?.id === shown?.id ? detail.data : shown;

  // tell the page once when a job it is watching finishes
  const seen = useRef<Record<string, string>>({});
  const openedAt = useRef(Date.now() / 1000);
  useEffect(() => {
    for (const j of jobs) {
      const before = seen.current[j.id];
      const active = (x?: string) => x === "running" || x === "queued";
      // finished since we last looked - including jobs fast enough to start and end between two polls
      if (!active(j.status) && (active(before) || (before === undefined && j.created >= openedAt.current - 1))) {
        notify(`${jobTitle(j)}: ${j.status}`, j.status === "succeeded" ? "ok" : "error");
        onFinished?.(j);
      }
      seen.current[j.id] = j.status;
    }
  }, [jobs, onFinished]);

  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job?.log?.length]);

  const cancel = async (id: string) => {
    if (!window.confirm("Cancel this job? Services it paused will be restarted.")) return;
    try {
      await api(`/kenning/jobs/${id}/cancel`, { method: "POST" });
      list.reload();
    } catch (e: any) {
      notify(e.message, "error");
    }
  };

  return (
    <Card title="Jobs" actions={list.data?.busy ? <StatusPill status="warning" label="GPU busy" /> : undefined}>
      {jobs.length === 0 ? (
        <Empty>No jobs yet.</Empty>
      ) : (
        <div className="grid gap-3 2xl:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
          <ul className="max-h-[12rem] space-y-1 overflow-auto 2xl:max-h-[28rem]">
            {jobs.map((j) => (
              <li key={j.id}>
                <button
                  onClick={() => setSel(j.id)}
                  className={`w-full rounded-md px-2 py-1.5 text-left text-[13px] ${job?.id === j.id ? "bg-surface-2" : "hover:bg-surface-2"}`}
                >
                  <div className="flex items-center gap-2">
                    <StatusPill status={STATUS[j.status] ?? "neutral"} label={j.status} />
                    <span className="text-xs text-ink-3">{new Date(j.created * 1000).toLocaleString([], { hour12: false })}</span>
                  </div>
                  <div className="mt-0.5 truncate text-ink-2" title={jobTitle(j)}>
                    {jobTitle(j)}
                  </div>
                </button>
              </li>
            ))}
          </ul>
          {job && (
            <div className="min-w-0">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="text-[13px] font-medium text-ink">{jobTitle(job)}</div>
                <div className="flex items-center gap-2 text-xs text-ink-3">
                  {elapsed(job)}
                  {job.status === "running" && (
                    <Button variant="danger" onClick={() => cancel(job.id)}>
                      Cancel
                    </Button>
                  )}
                </div>
              </div>
              {job.progress && job.progress.total > 0 && (
                <div className="mb-2">
                  <div className="h-1.5 overflow-hidden rounded bg-surface-2">
                    <div className="h-full bg-accent" style={{ width: `${Math.min(100, (100 * job.progress.done) / job.progress.total)}%` }} />
                  </div>
                  <div className="mt-1 text-xs tabular text-ink-3">
                    {job.progress.done.toLocaleString()} / {job.progress.total.toLocaleString()}
                  </div>
                </div>
              )}
              {job.paused?.length > 0 && (
                <p className="mb-2 text-xs text-ink-3">
                  Paused for this job (restarted when it ends): {job.paused.join(", ")}
                </p>
              )}
              {job.error && <p className="mb-2 text-xs" style={{ color: "var(--critical)" }}>{job.error}</p>}
              <pre ref={logRef} className="max-h-[22rem] overflow-auto rounded-md border border-line bg-surface-2 p-2 font-mono text-[11px] leading-relaxed text-ink-2">
                {(job.log ?? []).join("\n") || "…"}
              </pre>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
