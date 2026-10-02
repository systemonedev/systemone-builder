// Shared types and helpers for the Kenning pages (Overview, Models, Try it, Connect).

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
};

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
