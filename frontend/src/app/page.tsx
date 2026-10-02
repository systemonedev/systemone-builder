"use client";

import Link from "next/link";
import { Card, Empty, PageTitle, Stat, StatusPill } from "@/components/ui";
import { fmtNum, fmtPct } from "@/utils/format";
import { usePoll } from "@/utils/hooks";
import type { KenningStatus, ModelSummary } from "@/utils/kenning";

const STEPS = [
  { href: "/try", title: "Try it", text: "Ask typed questions about any text or JSON and see the probabilities, live." },
  { href: "/models", title: "Models", text: "Every trained Kenning model with its held-out results. Activate one, or export it." },
  { href: "/connect", title: "Connect", text: "Call Kenning from your own code, or run an exported model in-process." },
];

export default function Overview() {
  const status = usePoll<KenningStatus>("/kenning/status", 5000);
  const models = usePoll<ModelSummary[]>("/kenning/models", 15000);
  const s = status.data;
  const active = models.data?.find((m) => m.active) ?? null;
  const cal = active?.heldout.calibrated;
  return (
    <div>
      <PageTitle
        title="Kenning"
        sub="A local System One model: typed questions in, calibrated answers out, in one forward pass. No text generation."
      />
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Kenning server"
          value={s ? <StatusPill status={s.server.online ? "good" : "critical"} label={s.server.online ? "serving" : "offline"} /> : "–"}
          sub={s?.server.online ? `${s.server.device ?? ""}` : s?.server.error?.slice(0, 60)}
        />
        <Stat label="Active model" value={<span className="text-base">{s?.server.model ?? "–"}</span>} sub={s?.active ? "activated in Models" : "default (KENNING_MODEL)"} />
        <Stat label="Held-out accuracy" value={fmtPct(cal?.accuracy, 1)} sub={active ? `${cal?.n ?? "?"} questions, calibrated` : "no trained model active"} />
        <Stat label="Calibration error (ECE)" value={fmtNum(cal?.ece, 3)} sub="lower is better; 0.9 ≈ right 90% of the time" />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {STEPS.map((x) => (
          <Link key={x.href} href={x.href} className="rounded-lg border border-line bg-surface p-4 hover:border-ink-3">
            <div className="text-sm font-semibold text-ink">{x.title} →</div>
            <p className="mt-1 text-[13px] text-ink-3">{x.text}</p>
          </Link>
        ))}
      </div>
      <div className="mt-5">
        <Card title="How Kenning decides">
          {status.error && !s ? (
            <Empty>Cannot reach the API: {status.error}</Empty>
          ) : (
            <ul className="list-disc space-y-1.5 pl-5 text-[13px] text-ink-2">
              <li>
                You send <strong>state</strong> (text or JSON) and typed <strong>questions</strong>: <code>noul</code> (yes/no,
                answered with the probability of yes), <code>choice</code> (one of several options) and <code>score</code> (a level on
                an ordered scale).
              </li>
              <li>Every candidate answer is scored against the state in one batch; probabilities come from those scores, calibrated on held-out data.</li>
              <li>The same request always gets the same answer on the same hardware. Use the probabilities to gate: act when sure, escalate when not.</li>
              <li>
                The wire format (<code>POST /v1/systemone</code>) is compatible with TypeSafe AI&apos;s System One API. Kenning is not
                affiliated with TypeSafe AI.
              </li>
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
