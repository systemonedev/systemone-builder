"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { JobsPanel } from "@/components/Jobs";
import { Button, Card, Empty, ErrorNote, Field, PageTitle, StatusPill, inputCls } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtInt, fmtPct } from "@/utils/format";
import { usePoll } from "@/utils/hooks";
import type { Base, Dataset, ImportFile, Job, KenningStatus, Problem } from "@/utils/kenning";
import { notify } from "@/utils/notify";

// Public sources `systemone data` can sample (kenning/multitask.py SOURCES).
const SOURCES: { id: string; dataset: string; license: string; clean: boolean }[] = [
  { id: "nli", dataset: "nyu-mll/multi_nli (no fiction)", license: "OANC", clean: true },
  { id: "clinc", dataset: "clinc/clinc_oos", license: "CC-BY-3.0", clean: true },
  { id: "amazon", dataset: "fancyzhx/amazon_polarity", license: "Apache-2.0", clean: true },
  { id: "civil", dataset: "google/civil_comments", license: "CC0-1.0", clean: true },
  { id: "boolq", dataset: "google/boolq", license: "CC-BY-SA-3.0", clean: false },
  { id: "dbpedia", dataset: "fancyzhx/dbpedia_14", license: "CC-BY-SA-3.0", clean: false },
];
const CLEAN_ROWS = "nli=20000, clinc=3000, amazon=2000, civil=2000";

function parseRows(text: string, sources: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const part of text.split(",").map((x) => x.trim()).filter(Boolean)) {
    const [k, v] = part.split("=").map((x) => x.trim());
    if (sources.includes(k) && /^\d+$/.test(v ?? "")) out[k] = Number(v);
  }
  return out;
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+/, "").slice(0, 64);

async function startJob(kind: string, params: Record<string, unknown>): Promise<Job | null> {
  try {
    const job = await api<Job>("/kenning/jobs", { method: "POST", json: { kind, params } });
    notify("Job started", "info");
    return job;
  } catch (e: any) {
    notify(e.message, "error");
    return null;
  }
}

const SPEC_EXAMPLE = `{
  "name": "my_problem",
  "title": "What the model decides",
  "writes": "a support chat between a customer and an agent",
  "state": {"fields": {"chat": "the conversation so far", "account": "plan and tenure"}},
  "questions": [
    {"id": "resolved", "ask": "Was the customer's issue resolved?",
     "labels": {"yes": "the customer confirms it works", "no": "still open or unclear"}},
    {"id": "sentiment", "ask": "How does the customer feel?", "ordinal": true,
     "labels": {"negative": "", "neutral": "", "positive": ""}}
  ]
}`;

function ProblemSpecEditor({ onSaved }: { onSaved: () => void }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState(SPEC_EXAMPLE);
  if (!open)
    return (
      <button className="text-xs text-accent underline" onClick={() => setOpen(true)}>
        Define your own problem…
      </button>
    );
  return (
    <div className="space-y-2 rounded-md border border-line p-2">
      <p className="text-xs text-ink-3">
        Describe the state and every question with all its answers. A teacher LLM writes cases for randomly drawn answers, so every label is known.
        Optional <code>constraints</code> rule out impossible combinations: <code>{'{"if": {"risk": "high"}, "then": {"confirm": "yes"}}'}</code>.
      </p>
      <textarea className={`${inputCls} h-56 font-mono text-xs`} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      <div className="flex gap-2">
        <Button
          variant="primary"
          onClick={async () => {
            let spec: unknown;
            try {
              spec = JSON.parse(text);
            } catch {
              notify("Not valid JSON", "error");
              return;
            }
            try {
              const p = await api<Problem>("/kenning/problems", { method: "POST", json: spec });
              notify(`Saved problem ${p.name}`, "info");
              setOpen(false);
              onSaved();
            } catch (e: any) {
              notify(e.message, "error");
            }
          }}
        >
          Save problem
        </Button>
        <Button variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function ImportUpload({ onSaved }: { onSaved: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [license, setLicense] = useState("");
  const [content, setContent] = useState<string | null>(null);
  const [questions, setQuestions] = useState("");
  const [busy, setBusy] = useState(false);
  if (!open)
    return (
      <button className="text-xs text-accent underline" onClick={() => setOpen(true)}>
        Import your own labelled data…
      </button>
    );
  const labelled = content !== null && !content.slice(0, 2000).includes('"targets"');
  return (
    <div className="space-y-2 rounded-md border border-line p-2">
      <p className="text-xs text-ink-3">
        JSONL, one example per line: training rows <code>{'{"state", "questions", "targets"}'}</code>, or <code>{'{"state", "labels"}'}</code> with the
        questions below. Every line is checked before the file is kept.
      </p>
      <input
        type="file"
        accept=".jsonl,.json,.txt"
        className="text-xs"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          if (f.size > 50 * 2 ** 20) {
            notify("Files up to 50 MB; split larger ones", "error");
            return;
          }
          setContent(await f.text());
          if (!name) setName(slug(f.name.replace(/\.(jsonl|json|txt)$/, "")));
        }}
      />
      <div className="grid gap-2 sm:grid-cols-2">
        <Field label="Name">
          <input className={inputCls} value={name} onChange={(e) => setName(slug(e.target.value))} />
        </Field>
        <Field label="Licence (recorded in the model card)">
          <input className={inputCls} value={license} placeholder="e.g. CC-BY-4.0, internal, CC0" onChange={(e) => setLicense(e.target.value)} />
        </Field>
      </div>
      {labelled && (
        <Field label='Questions for "labels" lines: {"id": {"type", "instructions", "criteria"}}'>
          <textarea className={`${inputCls} h-24 font-mono text-xs`} value={questions} onChange={(e) => setQuestions(e.target.value)} spellCheck={false} />
        </Field>
      )}
      <div className="flex gap-2">
        <Button
          variant="primary"
          disabled={!name || !license.trim() || content === null || busy}
          onClick={async () => {
            let qs: unknown = null;
            if (labelled && questions.trim()) {
              try {
                qs = JSON.parse(questions);
              } catch {
                notify("Questions are not valid JSON", "error");
                return;
              }
            }
            setBusy(true);
            try {
              const r = await api<ImportFile>("/kenning/imports", { method: "POST", json: { name, license, content, questions: qs } });
              notify(`Imported ${fmtInt(r.rows ?? 0)} rows`, "info");
              setOpen(false);
              setContent(null);
              onSaved();
            } catch (e: any) {
              notify(e.message, "error");
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Checking…" : "Import"}
        </Button>
        <Button variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

type Preset = "problem_general" | "problem_only" | "general";
const PRESETS: { id: Preset; label: string; note: string }[] = [
  { id: "problem_general", label: "My problem + general", note: "Recommended: your cases plus general data, so the model stays good at everything else." },
  { id: "problem_only", label: "My problem only", note: "Fastest to train; expect the model to get worse at other decisions." },
  { id: "general", label: "General multi-task", note: "The clean-licence recipe behind kenning-large-v0.4." },
];

function DataCard({ datasets, pipeline, reload }: { datasets: Dataset[]; pipeline: boolean; reload: () => void }) {
  const problems = usePoll<Problem[]>("/kenning/problems", 0);
  const imports = usePoll<ImportFile[]>("/kenning/imports", 0);
  const [open, setOpen] = useState(false);
  const [preset, setPreset] = useState<Preset>("problem_general");
  const [name, setName] = useState("my-data-v1");
  const [picked, setPicked] = useState<Record<string, number>>({});
  const [picks, setPicks] = useState<string[]>([]);
  const [holdout, setHoldout] = useState(0.1);
  const [teacherPick, setTeacher] = useState<"triage" | "custom" | null>(null); // null: triage when the pipeline runs
  const teacher = teacherPick ?? (pipeline ? "triage" : "custom");
  const [teacherUrl, setTeacherUrl] = useState("");
  const [teacherModel, setTeacherModel] = useState("");
  const [check, setCheck] = useState(true);
  const [sources, setSources] = useState<string[]>(SOURCES.filter((s) => s.clean).map((s) => s.id));
  const [perSource, setPerSource] = useState(1200);
  const [rows, setRows] = useState(CLEAN_ROWS);
  const [phishing, setPhishing] = useState(0);
  const [synthEmail, setSynthEmail] = useState(0);
  const [synthTasks, setSynthTasks] = useState(0);
  const toggle = (id: string) => setSources((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const nonClean = sources.filter((id) => !SOURCES.find((s) => s.id === id)?.clean);
  const withProblem = preset !== "general";
  const withGeneral = preset !== "problem_only";
  const chosenProblems = withProblem ? picked : {};
  const chosenImports = withProblem ? picks : [];
  const needsTeacher = Object.keys(chosenProblems).length > 0;
  const teacherOk = !needsTeacher || (teacher === "triage" ? pipeline : !!teacherUrl && !!teacherModel);
  const hasData = (withGeneral && sources.length > 0) || needsTeacher || chosenImports.length > 0 || phishing > 0;

  return (
    <Card
      title="1 · Training data"
      actions={
        <Button variant={open ? "ghost" : "default"} onClick={() => setOpen((o) => !o)}>
          {open ? "Close" : "New dataset"}
        </Button>
      }
    >
      {open && (
        <div className="mb-4 space-y-4 rounded-md border border-line p-3">
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <Button key={p.id} variant={preset === p.id ? "primary" : "default"} onClick={() => setPreset(p.id)}>
                {p.label}
              </Button>
            ))}
          </div>
          <p className="text-xs text-ink-3">{PRESETS.find((p) => p.id === preset)?.note} Benchmark items are always excluded.</p>
          <Field label="Name">
            <input className={inputCls} value={name} onChange={(e) => setName(slug(e.target.value))} />
          </Field>

          {withProblem && (
            <section className="space-y-2">
              <div className="text-xs font-medium text-ink-2">Your problem</div>
              <div className="grid gap-1">
                {(problems.data ?? []).map((p) => (
                  <label key={p.name} className="flex items-start gap-2 text-[13px] text-ink-2">
                    <input
                      type="checkbox"
                      className="mt-1"
                      checked={p.name in picked}
                      onChange={() =>
                        setPicked((x) => {
                          const next = { ...x };
                          if (p.name in next) delete next[p.name];
                          else next[p.name] = 1000;
                          return next;
                        })
                      }
                    />
                    <span className="min-w-0 flex-1">
                      <span className="font-medium text-ink">{p.title}</span>
                      {!p.builtin && <span className="ml-1 text-xs text-ink-3">(yours)</span>}
                      <span className="block text-xs text-ink-3">
                        {p.description || `state: ${p.state.join(", ")}`} · asks {Object.keys(p.questions).join(", ")}
                      </span>
                    </span>
                    {p.name in picked && (
                      <span className="w-24 shrink-0">
                        <input
                          type="number"
                          min={20}
                          className={inputCls}
                          value={picked[p.name]}
                          title="cases to keep"
                          aria-label={`${p.title}: cases to keep`}
                          onChange={(e) => setPicked((x) => ({ ...x, [p.name]: Number(e.target.value) }))}
                        />
                      </span>
                    )}
                  </label>
                ))}
                {(imports.data ?? []).map((f) => (
                  <label key={f.name} className="flex items-start gap-2 text-[13px] text-ink-2">
                    <input
                      type="checkbox"
                      className="mt-1"
                      checked={picks.includes(f.name)}
                      onChange={() => setPicks((x) => (x.includes(f.name) ? x.filter((y) => y !== f.name) : [...x, f.name]))}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="font-medium text-ink">{f.name}</span> <span className="text-xs text-ink-3">(imported)</span>
                      <span className="block text-xs text-ink-3">
                        {fmtInt(f.rows ?? 0)} rows · asks {f.questions.join(", ")} · {f.license}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
              <div className="flex flex-wrap gap-4">
                <ProblemSpecEditor onSaved={problems.reload} />
                <ImportUpload onSaved={imports.reload} />
              </div>
              {needsTeacher && (
                <div className="space-y-2 rounded-md bg-surface p-2">
                  <div className="flex flex-wrap items-center gap-3 text-[13px] text-ink-2">
                    <span className="text-xs text-ink-3">Teacher that writes the cases:</span>
                    <label className="flex items-center gap-1">
                      <input type="radio" checked={teacher === "triage"} onChange={() => setTeacher("triage")} disabled={!pipeline} />
                      the pipeline&apos;s triage LLM{pipeline ? "" : " (not running)"}
                    </label>
                    <label className="flex items-center gap-1">
                      <input type="radio" checked={teacher === "custom"} onChange={() => setTeacher("custom")} />
                      another OpenAI-compatible server
                    </label>
                  </div>
                  {teacher === "custom" && (
                    <div className="grid gap-2 sm:grid-cols-2">
                      <Field label="Base URL">
                        <input className={inputCls} value={teacherUrl} placeholder="http://my-mac:11434/v1" onChange={(e) => setTeacherUrl(e.target.value.trim())} />
                      </Field>
                      <Field label="Model">
                        <input className={inputCls} value={teacherModel} placeholder="qwen3:32b" onChange={(e) => setTeacherModel(e.target.value.trim())} />
                      </Field>
                    </div>
                  )}
                  <label className="flex items-center gap-2 text-[13px] text-ink-2">
                    <input type="checkbox" checked={check} onChange={() => setCheck((c) => !c)} />
                    Blind check: the teacher answers each case without seeing the intended labels; keep it only if they match
                  </label>
                  <p className="text-xs text-ink-3">
                    A stronger teacher keeps more cases and writes better ones. Labelling the dataset with Clef afterwards (step 2) shows how much a second
                    model agrees.
                  </p>
                </div>
              )}
              {(needsTeacher || chosenImports.length > 0) && (
                <Field label={`Held out for benchmarking: ${Math.round(holdout * 100)}% of each problem and import (shows up on Verify)`}>
                  <input type="range" min={0} max={0.3} step={0.05} value={holdout} onChange={(e) => setHoldout(Number(e.target.value))} className="w-full" />
                </Field>
              )}
            </section>
          )}

          {withGeneral && (
            <section className="space-y-2">
              <div className="text-xs font-medium text-ink-2">General data</div>
              <div className="grid gap-1">
                {SOURCES.map((s) => (
                  <label key={s.id} className="flex items-center gap-2 text-[13px] text-ink-2">
                    <input type="checkbox" checked={sources.includes(s.id)} onChange={() => toggle(s.id)} />
                    <span>{s.dataset}</span>
                    <span className="ml-auto whitespace-nowrap text-xs" style={{ color: s.clean ? "var(--text-muted)" : "var(--warning)" }}>
                      {s.license}
                    </span>
                  </label>
                ))}
              </div>
              {nonClean.length > 0 && (
                <p className="text-xs" style={{ color: "var(--warning)" }}>
                  Share-alike sources ({nonClean.join(", ")}) are fine to train on, but list them in NOTICE.md when you share the weights.
                </p>
              )}
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Rows per source">
                  <input type="number" className={inputCls} value={perSource} min={10} onChange={(e) => setPerSource(Number(e.target.value))} />
                </Field>
                <Field label="Per-source overrides (name=rows, …)">
                  <input className={inputCls} value={rows} onChange={(e) => setRows(e.target.value)} />
                </Field>
                <Field label={`Synthetic tasks per task${pipeline ? "" : " (needs the pipeline's triage LLM)"}`}>
                  <input type="number" className={inputCls} value={synthTasks} min={0} disabled={!pipeline} onChange={(e) => setSynthTasks(Number(e.target.value))} />
                </Field>
              </div>
            </section>
          )}

          <details className="text-[13px] text-ink-2">
            <summary className="cursor-pointer text-xs text-ink-3">Email and phishing data</summary>
            <div className="mt-2 grid gap-3 sm:grid-cols-2">
              <Field label="Phishing rows (zefang-liu dataset, LGPL-3.0)">
                <input type="number" className={inputCls} value={phishing} min={0} onChange={(e) => setPhishing(Number(e.target.value))} />
              </Field>
              <Field label={`Synthetic modern emails per class${pipeline ? "" : " (needs the pipeline's triage LLM)"}`}>
                <input type="number" className={inputCls} value={synthEmail} min={0} disabled={!pipeline} onChange={(e) => setSynthEmail(Number(e.target.value))} />
              </Field>
            </div>
          </details>

          <Button
            variant="primary"
            disabled={!name || !hasData || !teacherOk}
            onClick={async () => {
              const job = await startJob("data", {
                name,
                sources: withGeneral ? sources : [],
                per_source: perSource,
                source_rows: withGeneral ? parseRows(rows, sources) : {},
                phishing_rows: phishing,
                synthetic_email: synthEmail,
                synthetic_tasks: withGeneral ? synthTasks : 0,
                problems: chosenProblems,
                imports: chosenImports,
                holdout,
                check,
                ...(needsTeacher && teacher === "custom" ? { teacher_url: teacherUrl, teacher_model: teacherModel } : {}),
              });
              if (job) {
                setOpen(false);
                reload();
              }
            }}
          >
            Build dataset
          </Button>
          {!teacherOk && <p className="text-xs" style={{ color: "var(--warning)" }}>Pick a teacher to write the problem cases.</p>}
        </div>
      )}
      {datasets.length === 0 ? (
        <Empty>No training sets yet.</Empty>
      ) : (
        <table className="w-full text-left text-[13px]">
          <thead className="text-xs text-ink-3">
            <tr>
              <th className="pb-1 pr-3 font-normal">dataset</th>
              <th className="pb-1 pr-3 font-normal">rows</th>
              <th className="pb-1 font-normal">sources</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.name} className="border-t border-line align-top">
                <td className="py-1.5 pr-3">
                  <div className="font-medium text-ink">{d.name}</div>
                  {d.teacher && (
                    <div className="text-xs text-ink-3">
                      labelled by {d.teacher.model} (alpha {d.teacher.alpha})
                    </div>
                  )}
                  {d.holdouts && Object.keys(d.holdouts).length > 0 && (
                    <div className="text-xs text-ink-3">
                      held out:{" "}
                      {Object.entries(d.holdouts)
                        .map(([k, n]) => `${k.replace(/^(problem|import):/, "")} ${n}`)
                        .join(", ")}
                    </div>
                  )}
                </td>
                <td className="tabular py-1.5 pr-3 text-ink-2">{fmtInt(d.rows)}</td>
                <td className="py-1.5 text-xs text-ink-3">
                  {Object.entries(d.sources)
                    .map(([k, s]) => `${k} ${fmtInt(s.rows)} (${s.license ?? "?"})`)
                    .join(" · ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

function LabelCard({ datasets }: { datasets: Dataset[] }) {
  const clef = usePoll<{ online: boolean; model?: string; hint?: string }>("/kenning/clef", 10000);
  const unlabelled = datasets.filter((d) => !d.teacher);
  const [ds, setDs] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const src = ds || unlabelled[0]?.name || "";
  const online = clef.data?.online;
  return (
    <Card
      title="2 · Label with a teacher (optional)"
      actions={clef.data && <StatusPill status={online ? "good" : "neutral"} label={online ? "Clef online" : "Clef off"} />}
    >
      <p className="mb-3 text-xs text-ink-3">
        Cloudflare&apos;s Clef (Apache-2.0) adds its probabilities to each row (distillation). Soft targets carry the teacher&apos;s uncertainty, which
        improves calibration. Clef needs most of a 24 GB GPU; the job pauses Kenning on that GPU while it labels (about 3 h for 30k rows).
      </p>
      {!online ? (
        <p className="text-[13px] text-ink-2">
          Start it with <code>docker compose --profile clef up -d clef</code> (first start downloads ~19 GB).
        </p>
      ) : unlabelled.length === 0 ? (
        <Empty>No unlabelled datasets.</Empty>
      ) : (
        <div className="grid items-end gap-3 sm:grid-cols-[1fr_10rem_auto]">
          <Field label="Dataset">
            <select className={inputCls} value={src} onChange={(e) => setDs(e.target.value)}>
              {unlabelled.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({fmtInt(d.rows)} rows)
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Teacher weight (alpha) ${alpha.toFixed(2)}`}>
            <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full" />
          </Field>
          <Button variant="primary" disabled={!src} onClick={() => startJob("label", { dataset: src, alpha, batch: 8 })}>
            Label → {src ? `${src}-clef` : "…"}
          </Button>
        </div>
      )}
    </Card>
  );
}

function TrainCard({ datasets, bases }: { datasets: Dataset[]; bases: Base[] }) {
  const [ds, setDs] = useState("");
  const [base, setBase] = useState("");
  const [typedName, setTypedName] = useState<string | null>(null); // null: follow the dataset
  const [lr, setLr] = useState("1e-5");
  const [epochs, setEpochs] = useState(1);
  const [maxLen, setMaxLen] = useState(512);
  const data = ds || datasets[0]?.name || "";
  const baseId = base || bases.find((b) => b.recommended)?.id || bases[0]?.id || "";
  const b = bases.find((x) => x.id === baseId);
  const name = typedName ?? (data ? slug(`kenning-${data}`) : "");
  return (
    <Card title="3 · Train">
      {datasets.length === 0 ? (
        <Empty>Build a dataset first.</Empty>
      ) : (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Dataset">
              <select className={inputCls} value={data} onChange={(e) => setDs(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name} ({fmtInt(d.rows)} rows{d.teacher ? `, ${d.teacher.model} labels` : ""})
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Base model">
              <select className={inputCls} value={baseId} onChange={(e) => setBase(e.target.value)}>
                {bases.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.id}
                    {x.recommended ? " (recommended)" : ""}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Model name">
              <input className={inputCls} value={name} onChange={(e) => setTypedName(slug(e.target.value))} />
            </Field>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Learning rate">
                <input className={inputCls} value={lr} onChange={(e) => setLr(e.target.value)} />
              </Field>
              <Field label="Epochs">
                <input type="number" step={0.5} min={0.5} className={inputCls} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
              </Field>
              <Field label="Max tokens">
                <input type="number" step={64} min={128} className={inputCls} value={maxLen} onChange={(e) => setMaxLen(Number(e.target.value))} />
              </Field>
            </div>
          </div>
          {b && (
            <p className="text-xs" style={b.apache_release ? { color: "var(--text-muted)" } : { color: "var(--warning)" }}>
              {b.license}.{" "}
              {b.apache_release ? "Models built on it are released under Apache-2.0." : "Models built on it are exported as research-only, not Apache-2.0."}
            </p>
          )}
          <p className="text-xs text-ink-3">
            Kenning (and the student, with the pipeline) is paused on its GPU while training runs: ~40 min to 2 h for 30k rows on an RTX 3090.
            Held-out accuracy and calibration are measured at the end; then benchmark it on <Link className="underline" href="/verify">Verify</Link>.
          </p>
          <Button
            variant="primary"
            disabled={!data || !name || !Number.isFinite(Number(lr))}
            onClick={() => startJob("train", { dataset: data, base: baseId, name, lr: Number(lr), epochs, max_length: maxLen })}
          >
            Train {name}
          </Button>
        </div>
      )}
    </Card>
  );
}

export default function TrainPage() {
  const datasets = usePoll<Dataset[]>("/kenning/datasets", 15000);
  const bases = usePoll<Base[]>("/kenning/bases", 0);
  const status = usePoll<KenningStatus>("/kenning/status", 30000);
  const [done, setDone] = useState<Job | null>(null);
  const reloadDatasets = datasets.reload;
  const onFinished = useCallback(
    (j: Job) => {
      reloadDatasets();
      if (j.kind === "train" && j.status === "succeeded") setDone(j);
    },
    [reloadDatasets],
  );
  const list = datasets.data ?? [];
  return (
    <div>
      <PageTitle
        title="Train"
        sub="Train Kenning for general decisions, for your own problem, or both: build a dataset, optionally distil a teacher's probabilities into it, and fine-tune. One job runs at a time."
      />
      <ErrorNote error={datasets.error} />
      {done && (
        <div className="mb-4 rounded-md border border-line bg-surface px-4 py-2.5 text-[13px] text-ink">
          <span style={{ color: "var(--good)" }}>●</span> {done.result.model} trained: held-out accuracy{" "}
          {fmtPct(done.result.heldout?.calibrated?.accuracy, 1)} (zero-shot {fmtPct(done.result.heldout?.zero_shot?.accuracy, 1)}).{" "}
          <Link className="font-medium text-accent underline" href="/models">
            Activate it
          </Link>{" "}
          and{" "}
          <Link className="font-medium text-accent underline" href="/verify">
            benchmark it
          </Link>
          .
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-4">
          <DataCard datasets={list} pipeline={!!status.data?.pipeline} reload={datasets.reload} />
          <LabelCard datasets={list} />
          <TrainCard datasets={list} bases={bases.data ?? []} />
        </div>
        <div>
          <JobsPanel kinds={["data", "label", "train"]} onFinished={onFinished} />
        </div>
      </div>
    </div>
  );
}
