"use client";

import { useEffect, useState } from "react";
import { Button, Card, PageTitle } from "@/components/ui";
import { usePoll } from "@/utils/hooks";
import type { KenningStatus } from "@/utils/kenning";

const REPO = "https://github.com/systemonedev/systemone-builder";

function Snippet({ title, lang, code, note }: { title: string; lang: string; code: string; note?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Card
      title={title}
      actions={
        <Button
          variant="ghost"
          onClick={() => {
            navigator.clipboard?.writeText(code).then(() => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            });
          }}
        >
          {copied ? "copied" : "copy"}
        </Button>
      }
    >
      {note && <p className="mb-2 text-[13px] text-ink-3">{note}</p>}
      <pre className="overflow-auto rounded-md border border-line bg-surface-2 p-3 font-mono text-[12px] leading-relaxed text-ink-2" data-lang={lang}>
        {code}
      </pre>
    </Card>
  );
}

export default function ConnectPage() {
  const status = usePoll<KenningStatus>("/kenning/status", 0);
  const [host, setHost] = useState("localhost");
  useEffect(() => setHost(window.location.hostname || "localhost"), []);
  const model = status.data?.active ?? "kenning-large-v0.1";

  const install = `pip install "systemone-client @ git+${REPO}#subdirectory=clients/python"
# to run an exported model in-process as well (adds torch + transformers):
pip install "systemone-client[local] @ git+${REPO}#subdirectory=clients/python"`;

  const direct = `from systemone import Client, Noul, Choice, Score

with Client("http://${host === "localhost" || host === "127.0.0.1" ? host : "localhost"}:8093") as client:   # the Kenning server, from this machine
    r = client.system_one(
        state={"ticket": {"subject": "Charged twice", "body": "I was billed twice this month."}},
        questions={
            "refund": Noul("Is the customer asking for a refund?"),
            "queue": Choice("Which queue should handle it?",
                            {"billing": "Charges, refunds", "technical": "Bugs, errors", "other": None}),
            "tone": Score("How heated is the message?", ["Polite", "Impatient", "Hostile"]),
        },
    )

refund = r.nouls["refund"].noul          # probability of yes (a float, not a bool)
if refund >= 0.9:
    ...                                   # act automatically
elif refund <= 0.1:
    ...                                   # safe to close
else:
    ...                                   # escalate to a person or a System 2 model
print(r.choices["queue"].choice, r.choices["queue"].probabilities)`;

  const viaApi = `import os
from systemone import Client, Noul

# Through SystemOne Builder's API (works from other machines when S1_BIND_ADDR allows it)
client = Client("http://${host}:8090/api", api_key=os.environ["S1_API_KEY"])
r = client.system_one(state="Your invoice is overdue, pay here: http://pay-now.example",
                      questions={"phishing": Noul("Is this a phishing attempt?")})`;

  const embedded = `# 1. Models page -> ${model} -> Export -> Download zip, then unzip it.
from systemone import Kenning, Noul

model = Kenning.from_pretrained("./${model}")      # runs in this process, GPU if available
r = model.system_one(state="Your invoice is overdue, pay here: http://pay-now.example",
                     questions={"phishing": Noul("Is this a phishing attempt?")})
print(r.nouls["phishing"].noul)`;

  const curl = `curl -s http://localhost:8093/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"comment": "Great match, the referee was awful though."},
  "questions": {
    "toxic":  {"type": "noul", "instructions": "Is this comment toxic?"},
    "action": {"type": "choice", "instructions": "What should moderation do?",
               "criteria": {"publish": null, "review": null, "remove": null}}
  }
}'`;

  const ts = `// Any language: POST the wire format as JSON.
const res = await fetch("http://localhost:8093/v1/systemone", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    state: { comment: "Great match, the referee was awful though." },
    questions: { toxic: { type: "noul", instructions: "Is this comment toxic?" } },
  }),
});
const { answers } = await res.json();
console.log(answers.toxic.noul);`;

  return (
    <div>
      <PageTitle title="Connect" sub="Use Kenning from your own projects: over HTTP, through this app's API, or in-process from an exported bundle." />
      <div className="grid gap-4 xl:grid-cols-2">
        <Snippet title="1 · Install the client library" lang="bash" code={install} note="Python ≥ 3.10. The client only needs httpx; the [local] extra adds torch and transformers." />
        <Snippet title="2 · Ask the Kenning server" lang="python" code={direct} note="Port 8093 is bound to 127.0.0.1: reachable from this machine only, no key." />
        <Snippet title="3 · Through the builder API (with key)" lang="python" code={viaApi} note="Uses S1_API_KEY. The API is on 127.0.0.1 unless S1_BIND_ADDR exposes it." />
        <Snippet title="4 · Run an exported model in-process" lang="python" code={embedded} note="No server needed. Same answers as the server for the same request." />
        <Snippet title="curl" lang="bash" code={curl} />
        <Snippet title="TypeScript / any language" lang="ts" code={ts} />
      </div>
      <p className="mt-4 text-xs text-ink-3">
        The wire format is compatible with TypeSafe AI&apos;s System One API, so existing clients for it can point their base URL here. Kenning and this
        project are not affiliated with TypeSafe AI.
      </p>
    </div>
  );
}
