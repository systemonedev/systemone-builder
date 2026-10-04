// Shared types and helpers for the Kenning pages (Overview, Models, Train, Verify, Try it, Connect).

export type Metrics = { n?: number; accuracy?: number; nll?: number; brier?: number; ece?: number } | null;

export type ModelSummary = {
  name: string;
  display_name: string;
  active: boolean;
  created?: string;
  base_model?: string;
  trained_on?: string;
  train_groups?: number;
  temperature?: Record<string, number>;
  max_length?: number;
  hyperparameters?: Record<string, unknown>;
  heldout: { zero_shot: Metrics; trained: Metrics; calibrated: Metrics };
  heldout_by_type?: Record<string, Metrics>;
  size_bytes: number;
  has_weights: boolean;
  manifest?: { sources?: Record<string, { dataset?: string; rows?: number; license?: string }> } | null;
};

export type KenningStatus = {
  server: { online: boolean; model?: string; path?: string; device?: string; temperature?: Record<string, number>; error?: string };
  active: string | null;
  pipeline?: boolean;
};

// ---------------------------------------------------------- Train / Verify
export type JobKind = "data" | "label" | "train" | "bench";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type Job = {
  id: string;
  kind: JobKind;
  params: Record<string, any>;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  progress: { done: number; total: number } | null;
  result: { model?: string; heldout?: ModelSummary["heldout"]; files?: string[] };
  error: string | null;
  paused: string[];
  log?: string[];
};
export type Dataset = {
  name: string;
  rows: number;
  size_bytes: number;
  modified: number;
  sources: Record<string, { rows?: number; license?: string; dataset?: string }>;
  teacher?: { model: string; alpha: number; agreement?: Record<string, { agreement: number }> } | null;
};
export type Base = { id: string; license: string; apache_release: boolean; recommended: boolean };

export type GateMetrics = {
  hi: number;
  lo: number;
  auto_positive: number;
  auto_negative: number;
  to_human: number;
  automation_rate: number;
  false_positives_acted: number;
  false_negatives_closed: number;
  automated_accuracy: number | null;
};
export type QuestionMetrics = {
  type: string;
  n: number;
  accuracy?: number;
  brier?: number;
  ece?: number;
  mean_top_probability?: number;
  gate?: GateMetrics;
};
export type EngineReport = {
  engine: string;
  model?: string | null;
  items: number;
  errors: number;
  error_examples?: string[];
  latency_ms: { p50: number | null; p95: number | null; mean: number | null };
  determinism: { checked: number; identical: number };
  questions: Record<string, QuestionMetrics>;
  /** multi-task suites: mean per-task accuracy */
  macro_accuracy?: number | null;
};
export type BenchSummary = { file: string; ts: number; suite: string; description?: string; gate?: string; items: number; reports: EngineReport[] };

/** id: the `--suite` value; name: the suite name stored in result files. */
export const SUITES: { id: string; name: string; label: string; note: string }[] = [
  { id: "multi", name: "multi", label: "Multi-task (14 tasks)", note: "~680 items over 14 decision tasks never trained on; headline: macro accuracy" },
  { id: "modern2", name: "modern_email_2", label: "Modern emails 2", note: "20 held-out modern emails, incl. calm credential lures" },
  { id: "modern", name: "modern_email", label: "Modern emails", note: "20 hand-written modern emails (training scenarios were designed after seeing it)" },
  { id: "phishing", name: "phishing", label: "Phishing dataset", note: "zefang-liu/phishing-email-dataset, balanced sample" },
  { id: "layouts", name: "layouts", label: "Layouts", note: "the phishing emails in 4 state layouts" },
  { id: "ood", name: "ood", label: "Out of domain", note: "SMS spam, emotion, news topic: tasks never trained on" },
];

/** "kenning-large-v0.4" for Kenning, "clef · <model>" for others. Runs from before the
 *  served model was recorded show as "kenning (model not recorded)". */
export const reportLabel = (r: EngineReport) =>
  r.engine === "kenning" ? (r.model ?? "kenning (model not recorded)") : r.model ? `${r.engine} · ${r.model}` : r.engine;

export type QType = "noul" | "choice" | "score";
export type QuestionDraft = {
  id: string;
  type: QType;
  instructions: string;
  criteria: string; // noul: free text; choice: "option: description" per line; score: one level per line
};

export type Answer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; probabilities: Record<string, number>; legend?: Record<string, string> };

export type WireResponse = { model: string; answers: Record<string, Answer>; usage?: { input_tokens: number; output_tokens: number }; latency_ms?: number };
export type EngineResult = { engine: string; response?: WireResponse; error?: string; latency_ms: number };

export const fmtBytes = (b: number) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : `${(b / 2 ** 20).toFixed(0)} MB`);

/** Turn the form's question drafts into wire-format questions. */
export function toWireQuestions(qs: QuestionDraft[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const q of qs) {
    const lines = q.criteria.split("\n").map((l) => l.trim()).filter(Boolean);
    if (q.type === "noul") {
      out[q.id] = q.criteria.trim() ? { type: "noul", instructions: q.instructions, criteria: q.criteria.trim() } : { type: "noul", instructions: q.instructions };
    } else if (q.type === "choice") {
      const criteria: Record<string, string | null> = {};
      for (const l of lines) {
        const i = l.indexOf(":");
        if (i > 0) criteria[l.slice(0, i).trim()] = l.slice(i + 1).trim() || null;
        else criteria[l] = null;
      }
      out[q.id] = { type: "choice", instructions: q.instructions, criteria };
    } else {
      out[q.id] = { type: "score", instructions: q.instructions, criteria: lines };
    }
  }
  return out;
}

/** Parse the state box: JSON if it parses as an object, otherwise plain text. */
export function parseState(text: string): unknown {
  const t = text.trim();
  if (t.startsWith("{")) {
    try {
      return JSON.parse(t);
    } catch {
      /* fall through: treat as text */
    }
  }
  return text;
}

export const PRESETS: { name: string; state: string; questions: QuestionDraft[] }[] = [
  {
    name: "Suspicious email",
    state: JSON.stringify(
      {
        email: {
          from: "it-support@company-helpdesk-reset.com",
          subject: "Password expires today",
          body: "Your mailbox password expires in 2 hours. Keep your current password by confirming it here: http://company-helpdesk-reset.com/keep",
        },
      },
      null,
      2,
    ),
    questions: [
      { id: "phishing", type: "noul", instructions: "Is this email a phishing attempt?", criteria: "" },
      { id: "category", type: "choice", instructions: "What kind of email is this?", criteria: "Safe: Normal personal or business email\nCredential theft: Tries to get a password or login\nPayment fraud: Tries to get money sent\nSpam: Unwanted marketing" },
      { id: "severity", type: "score", instructions: "If the recipient falls for it, how bad is it?", criteria: "No impact\nMinor nuisance\nAccount takeover\nCompany-wide breach" },
    ],
  },
  {
    name: "Support ticket",
    state: JSON.stringify({ ticket: { subject: "Charged twice", body: "Hi, I was billed twice for my plan this month. No rush, but could you refund one of the charges? Thanks!" } }, null, 2),
    questions: [
      { id: "refund", type: "noul", instructions: "Is the customer asking for a refund?", criteria: "" },
      { id: "urgent", type: "noul", instructions: "Does the customer say this is urgent?", criteria: "" },
      { id: "queue", type: "choice", instructions: "Which queue should handle it?", criteria: "billing: Charges, refunds, invoices\ntechnical: Bugs, errors, outages\naccount: Login, profile, cancellation\nother: Anything else" },
      { id: "tone", type: "score", instructions: "How heated is the message?", criteria: "Polite\nImpatient\nHostile" },
    ],
  },
  {
    name: "Comment moderation",
    state: "Honestly the referee was blind tonight, worst call of the season. Still proud of the team though.",
    questions: [
      { id: "toxic", type: "noul", instructions: "Is this comment toxic?", criteria: "" },
      { id: "threat", type: "noul", instructions: "Does this comment contain a threat?", criteria: "" },
      { id: "action", type: "choice", instructions: "What should moderation do?", criteria: "publish: Fine as is\nreview: A human should look at it\nremove: Clearly breaks the rules" },
    ],
  },
];
