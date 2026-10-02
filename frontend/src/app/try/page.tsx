"use client";

import { useState } from "react";
import { Button, Card, Empty, ErrorNote, Field, inputCls, Json, PageTitle, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtMs, fmtPct } from "@/utils/format";
import { usePoll } from "@/utils/hooks";
import { type Answer, type EngineResult, parseState, PRESETS, type QType, type QuestionDraft, toWireQuestions } from "@/utils/kenning";

type Engine = { id: string; name: string; available: boolean; note: string };

const CRITERIA_HINT: Record<QType, string> = {
  noul: "optional: what counts as yes",
  choice: "one option per line, optionally  option: description",
  score: "one level per line, lowest first",
};

function Bar({ label, p, strong }: { label: string; p: number; strong?: boolean }) {
  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className={`w-36 truncate ${strong ? "font-medium text-ink" : "text-ink-2"}`} title={label}>
        {label}
      </span>
      <span className="relative h-2 flex-1 overflow-hidden rounded bg-surface-2">
        <span className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.max(p * 100, 0.5)}%`, background: strong ? "var(--accent)" : "var(--text-muted)" }} />
      </span>
      <span className="tabular w-12 text-right text-ink-3">{fmtPct(p, 1)}</span>
    </div>
  );
}

function AnswerView({ a }: { a: Answer }) {
  if (a.type === "noul")
    return (
      <div>
        <div className="mb-1 text-[13px]">
          <span className="tabular font-semibold text-ink">{a.noul.toFixed(3)}</span>
          <span className="text-ink-3"> probability of yes · {a.noul >= 0.9 ? "act" : a.noul <= 0.1 ? "close" : "escalate"} at 0.9 / 0.1</span>
        </div>
        <Bar label="yes" p={a.noul} strong={a.noul >= 0.5} />
        <Bar label="no" p={1 - a.noul} strong={a.noul < 0.5} />
      </div>
    );
  const probs = Object.entries(a.probabilities);
  const top = a.type === "choice" ? a.choice : null;
  return (
    <div>
      <div className="mb-1 text-[13px]">
        <span className="font-semibold text-ink">{a.type === "choice" ? a.choice : a.score.toFixed(2)}</span>
        <span className="text-ink-3"> · confidence {a.confidence.toFixed(2)}</span>
        {a.type === "score" && <span className="text-ink-3"> · probability-weighted level</span>}
      </div>
      {probs.map(([k, p]) => (
        <Bar key={k} label={a.type === "score" ? `${k} · ${a.legend?.[k] ?? ""}` : k} p={p} strong={a.type === "choice" ? k === top : p === Math.max(...probs.map((x) => x[1]))} />
      ))}
    </div>
  );
}

export default function TryPage() {
  const engines = usePoll<Engine[]>("/systemone/engines", 15000);
  const [state, setState] = useState(PRESETS[0].state);
  const [questions, setQuestions] = useState<QuestionDraft[]>(PRESETS[0].questions);
  const [withJev, setWithJev] = useState(false);
  const [results, setResults] = useState<EngineResult[] | null>(null);
  const [repeat, setRepeat] = useState<{ runs: number; identical: number; latency: number[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const jev = engines.data?.find((e) => e.id === "jev");
  const kenning = engines.data?.find((e) => e.id === "kenning");

  const request = () => ({ state: parseState(state), questions: toWireQuestions(questions) });
  const setQ = (i: number, patch: Partial<QuestionDraft>) => setQuestions((qs) => qs.map((q, j) => (j === i ? { ...q, ...patch } : q)));

  const run = async () => {
    setErr(null);
    setRepeat(null);
    try {
      const r = await api<{ results: EngineResult[] }>("/systemone/compare", {
        method: "POST",
        json: { request: request(), engines: withJev && jev?.available ? ["kenning", "jev"] : ["kenning"] },
      });
      setResults(r.results);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  const runTen = async () => {
    setErr(null);
    try {
      const body = { request: request(), engines: ["kenning"] };
      const outs: EngineResult[] = [];
      for (let i = 0; i < 10; i++) outs.push((await api<{ results: EngineResult[] }>("/systemone/compare", { method: "POST", json: body })).results[0]);
      const first = JSON.stringify(outs[0].response?.answers);
      setRepeat({ runs: outs.length, identical: outs.filter((o) => JSON.stringify(o.response?.answers) === first).length, latency: outs.map((o) => o.latency_ms) });
      setResults([outs[0]]);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  return (
    <div>
      <PageTitle
        title="Try it"
        sub="Ask typed questions about any text or JSON. Answers are probabilities, not generated text."
        actions={
          <select className={`${inputCls} w-48`} onChange={(e) => { const p = PRESETS[+e.target.value]; setState(p.state); setQuestions(p.questions); setResults(null); setRepeat(null); }} defaultValue={0}>
            {PRESETS.map((p, i) => (
              <option key={p.name} value={i}>Example: {p.name}</option>
            ))}
          </select>
        }
      />
      <ErrorNote error={err} />
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-4">
          <Card title="State">
            <textarea className={`${inputCls} h-48 font-mono text-[12px]`} value={state} onChange={(e) => setState(e.target.value)} spellCheck={false} />
            <p className="mt-1 text-xs text-ink-3">Plain text, or a JSON object (sent as structured state).</p>
          </Card>
          <Card
            title={`Questions (${questions.length})`}
            actions={
              <>
                {(["noul", "choice", "score"] as QType[]).map((t) => (
                  <Button key={t} variant="ghost" onClick={() => setQuestions((qs) => [...qs, { id: `${t}_${qs.length + 1}`, type: t, instructions: "", criteria: "" }])}>
                    + {t}
                  </Button>
                ))}
              </>
            }
          >
            <div className="space-y-3">
              {questions.map((q, i) => (
                <div key={i} className="rounded-md border border-line p-3">
                  <div className="mb-2 flex gap-2">
                    <input className={`${inputCls} w-36 font-mono`} value={q.id} onChange={(e) => setQ(i, { id: e.target.value.replace(/\s/g, "_") })} aria-label="question id" />
                    <select className={`${inputCls} w-28`} value={q.type} onChange={(e) => setQ(i, { type: e.target.value as QType })} aria-label="question type">
                      <option value="noul">noul</option>
                      <option value="choice">choice</option>
                      <option value="score">score</option>
                    </select>
                    <Button variant="ghost" onClick={() => setQuestions((qs) => qs.filter((_, j) => j !== i))}>remove</Button>
                  </div>
                  <Field label="Instructions">
                    <input className={inputCls} value={q.instructions} onChange={(e) => setQ(i, { instructions: e.target.value })} />
                  </Field>
                  <div className="mt-2">
                    <Field label={`Criteria (${CRITERIA_HINT[q.type]})`}>
                      <textarea className={`${inputCls} h-20 font-mono text-[12px]`} value={q.criteria} onChange={(e) => setQ(i, { criteria: e.target.value })} spellCheck={false} />
                    </Field>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
        <div className="space-y-4">
          <Card
            title="Run"
            actions={
              <>
                <Button onClick={runTen} disabled={!kenning?.available} title="Ask the same request 10 times; Kenning should answer identically">Run 10×</Button>
                <Button variant="primary" onClick={run} disabled={!kenning?.available}>Run</Button>
              </>
            }
          >
            <div className="flex flex-wrap items-center gap-3 text-[13px]">
              <StatusPill status={kenning?.available ? "good" : "critical"} label={kenning?.name ?? "Kenning"} />
              <label className={`flex items-center gap-2 ${jev?.available ? "text-ink-2" : "text-ink-3"}`} title={jev?.note}>
                <input type="checkbox" checked={withJev && !!jev?.available} disabled={!jev?.available} onChange={(e) => setWithJev(e.target.checked)} />
                Compare with TypeSafe Jev
              </label>
            </div>
            {jev && <p className="mt-1 text-xs text-ink-3">{jev.note}</p>}
            {!kenning?.available && <p className="mt-2 text-xs text-ink-3">{kenning?.note ?? "Kenning server offline: docker compose up -d kenning"}</p>}
            {repeat && (
              <div className="mt-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-[13px]">
                <StatusPill status={repeat.identical === repeat.runs ? "good" : "warning"} label={`${repeat.identical}/${repeat.runs} identical`} />
                <span className="ml-2 text-ink-3">
                  latency {fmtMs(Math.min(...repeat.latency))}–{fmtMs(Math.max(...repeat.latency))}
                </span>
              </div>
            )}
          </Card>
          {!results ? (
            <Empty>Press Run to ask the questions.</Empty>
          ) : (
            <div className={`grid gap-4 ${results.length > 1 ? "lg:grid-cols-2" : ""}`}>
              {results.map((r) => (
                <Card key={r.engine} title={r.response?.model ?? r.engine} actions={<span className="tabular text-xs text-ink-3">{fmtMs(r.latency_ms)}</span>}>
                  {r.error ? (
                    <ErrorNote error={r.error} />
                  ) : (
                    <div className="space-y-4">
                      {Object.entries(r.response!.answers).map(([qid, a]) => (
                        <div key={qid}>
                          <div className="mb-1 font-mono text-xs text-ink-3">{qid}</div>
                          <AnswerView a={a} />
                        </div>
                      ))}
                      <div className="text-xs text-ink-3">
                        {r.response!.usage?.input_tokens ?? "–"} input tokens · {r.response!.usage?.output_tokens ?? 0} output tokens
                      </div>
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
          <Card title="Request (wire format)">
            <Json value={request()} className="max-h-72" />
            <p className="mt-1 text-xs text-ink-3">Send this to <code>POST /v1/systemone</code>; see Connect for code.</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
