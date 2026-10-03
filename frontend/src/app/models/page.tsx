"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, Card, Empty, ErrorNote, PageTitle, StatusPill } from "@/components/ui";
import { api } from "@/utils/api";
import { fmtNum, fmtPct } from "@/utils/format";
import { usePoll } from "@/utils/hooks";
import { fmtBytes, type Metrics, type ModelSummary } from "@/utils/kenning";
import { notify } from "@/utils/notify";

function MetricRow({ label, m }: { label: string; m: Metrics }) {
  return (
    <tr className="border-t border-line">
      <td className="py-1.5 pr-3 text-ink-2">{label}</td>
      <td className="tabular py-1.5 pr-3">{fmtPct(m?.accuracy, 1)}</td>
      <td className="tabular py-1.5 pr-3">{fmtNum(m?.brier, 3)}</td>
      <td className="tabular py-1.5">{fmtNum(m?.ece, 3)}</td>
    </tr>
  );
}

export default function ModelsPage() {
  const models = usePoll<ModelSummary[]>("/kenning/models", 10000);
  const [sel, setSel] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [exported, setExported] = useState<Record<string, { size_bytes: number; seconds: number }>>({});
  const list = models.data ?? [];
  const m = list.find((x) => x.name === sel) ?? list.find((x) => x.active) ?? list[0];

  const activate = async (name: string) => {
    setErr(null);
    try {
      await api(`/kenning/models/${encodeURIComponent(name)}/activate`, { method: "POST" });
      notify(`${name} is now serving`, "ok");
      models.reload();
    } catch (e: any) {
      setErr(e.message);
    }
  };
  const exportBundle = async (name: string) => {
    setErr(null);
    try {
      const r = await api<{ size_bytes: number; seconds: number }>(`/kenning/models/${encodeURIComponent(name)}/export`, { method: "POST" });
      setExported((x) => ({ ...x, [name]: r }));
      notify(`Export bundle ready (${fmtBytes(r.size_bytes)})`, "ok");
    } catch (e: any) {
      setErr(e.message);
    }
  };
  const remove = async (name: string) => {
    if (!window.confirm(`Delete ${name} and its export bundle? This cannot be undone.`)) return;
    setErr(null);
    try {
      await api(`/kenning/models/${encodeURIComponent(name)}`, { method: "DELETE" });
      notify(`${name} deleted`, "info");
      setSel(null);
      models.reload();
    } catch (e: any) {
      setErr(e.message);
    }
  };

  return (
    <div>
      <PageTitle title="Models" sub="Trained Kenning models in the workspace volume. Activate one to serve it; export one to use it elsewhere." />
      <ErrorNote error={err ?? models.error} />
      {!models.data ? (
        <Empty>Loading models…</Empty>
      ) : list.length === 0 ? (
        <Empty>
          No trained models yet: build one on the{" "}
          <Link className="underline" href="/train">
            Train
          </Link>{" "}
          page. Until then Kenning serves its zero-shot base model.
        </Empty>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <Card title="Registry">
            <table className="w-full text-left text-[13px]">
              <thead className="text-xs text-ink-3">
                <tr>
                  <th className="pb-2 pr-3 font-normal">model</th>
                  <th className="pb-2 pr-3 font-normal">held-out acc.</th>
                  <th className="pb-2 pr-3 font-normal">ECE</th>
                  <th className="pb-2 font-normal">size</th>
                </tr>
              </thead>
              <tbody>
                {list.map((x) => (
                  <tr
                    key={x.name}
                    onClick={() => setSel(x.name)}
                    className={`cursor-pointer border-t border-line ${m?.name === x.name ? "bg-surface-2" : "hover:bg-surface-2"}`}
                  >
                    <td className="py-2 pr-3">
                      <div className="font-medium text-ink">{x.name}</div>
                      <div className="text-xs text-ink-3">{x.created?.replace("T", " ").replace("Z", " UTC") ?? "–"}</div>
                    </td>
                    <td className="tabular py-2 pr-3">{fmtPct(x.heldout.calibrated?.accuracy, 1)}</td>
                    <td className="tabular py-2 pr-3">{fmtNum(x.heldout.calibrated?.ece, 3)}</td>
                    <td className="py-2">
                      <div className="flex items-center gap-2">
                        <span className="tabular text-ink-3">{fmtBytes(x.size_bytes)}</span>
                        {x.active && <StatusPill status="good" label="active" />}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          {m && (
            <Card
              title={m.name}
              actions={
                <>
                  <Button variant="primary" disabled={m.active} onClick={() => activate(m.name)} title="Hot-swap the Kenning server to this model">
                    {m.active ? "Active" : "Activate"}
                  </Button>
                  <Button onClick={() => exportBundle(m.name)} title="Build a zip with weights, model card, licences and checksums">
                    Export
                  </Button>
                  <Button variant="danger" disabled={m.active} onClick={() => remove(m.name)}>
                    Delete
                  </Button>
                </>
              }
            >
              {exported[m.name] && (
                <div className="mb-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-[13px] text-ink-2">
                  Bundle ready ({fmtBytes(exported[m.name].size_bytes)}).{" "}
                  <a className="font-medium text-accent underline" href={`/api/v1/kenning/models/${encodeURIComponent(m.name)}/export`} download>
                    Download zip
                  </a>{" "}
                  · weights, tokenizer, <code>kenning.json</code>, model card (README.md), NOTICE.md with licences, SHA256SUMS. Use it with{" "}
                  <code>systemone.Kenning.from_pretrained()</code> (see Connect).
                </div>
              )}
              <dl className="grid grid-cols-[9rem_1fr] gap-x-3 gap-y-1 text-[13px]">
                <dt className="text-ink-3">base model</dt>
                <dd className="text-ink-2">{m.base_model ?? "–"}</dd>
                <dt className="text-ink-3">trained on</dt>
                <dd className="break-all text-ink-2">{m.trained_on ?? "–"}</dd>
                <dt className="text-ink-3">question groups</dt>
                <dd className="tabular text-ink-2">{m.train_groups ?? "–"}</dd>
                <dt className="text-ink-3">temperatures</dt>
                <dd className="tabular text-ink-2">
                  {m.temperature ? Object.entries(m.temperature).map(([k, v]) => `${k} ${v}`).join(" · ") : "–"}
                </dd>
                <dt className="text-ink-3">max tokens / pair</dt>
                <dd className="tabular text-ink-2">{m.max_length ?? "–"}</dd>
              </dl>
              <h3 className="mb-1 mt-4 text-xs font-semibold text-ink-3">Held-out results (split of the training pool)</h3>
              <table className="w-full text-left text-[13px]">
                <thead className="text-xs text-ink-3">
                  <tr>
                    <th className="pb-1 pr-3 font-normal" />
                    <th className="pb-1 pr-3 font-normal">accuracy</th>
                    <th className="pb-1 pr-3 font-normal">Brier</th>
                    <th className="pb-1 font-normal">ECE</th>
                  </tr>
                </thead>
                <tbody>
                  <MetricRow label="zero-shot (before)" m={m.heldout.zero_shot} />
                  <MetricRow label="trained" m={m.heldout.trained} />
                  <MetricRow label="trained + calibrated" m={m.heldout.calibrated} />
                </tbody>
              </table>
              <p className="mt-2 text-xs text-ink-3">
                In-distribution numbers. Before letting a model act on its own, benchmark it on held-out data on the{" "}
                <Link className="underline" href="/verify">
                  Verify
                </Link>{" "}
                page.
              </p>
              {m.manifest?.sources && (
                <>
                  <h3 className="mb-1 mt-4 text-xs font-semibold text-ink-3">Training data</h3>
                  <table className="w-full text-left text-[13px]">
                    <tbody>
                      {Object.entries(m.manifest.sources).map(([k, src]) => (
                        <tr key={k} className="border-t border-line">
                          <td className="py-1 pr-3 text-ink-2">{src.dataset ?? k}</td>
                          <td className="tabular py-1 pr-3 text-ink-3">{src.rows}</td>
                          <td className="py-1 text-ink-3">{src.license}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
