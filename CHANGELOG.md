# Changelog

Model versions and engine changes for SystemOne Builder and Kenning. Benchmarks are macro accuracy on
the `general` suite (1,328 held-out items; `systemone bench --suite general`). Clef is `clef-flash`, Jev is
`jev-latest`, both run for comparison only — never trained on.

## Engines

SystemOne models all answer the same wire format (`POST /v1/systemone`; typed questions → calibrated
probabilities, no text). Two engine families now exist:

- **Kenning** (the reflex) — a 435M DeBERTa **cross-encoder**. Scores each option in one pass over ≤512
  tokens. ~35 ms. Strong on text, safety, routing, conversation, answer-quality; structurally cannot
  compute over the state (numbers, tables, long logs). `kenning/model.py`, served by `kenning/serve.py`.
- **Kenning-XL** (the deliberate reasoner, v0.6, experimental) — a small **decoder** (Qwen3-1.7B) with a
  **constrained answer-token readout**: renders (state, question) and reads the next-token distribution
  over the answer tokens in one pass, so the full LM reasons over the whole 4k-token state. An optional
  **deliberate mode** generates a short reasoning trace first, then reads the answer — for numeric /
  policy / table decisions the one-pass readout can't do. `kenning/xl.py`, trained by `kenning/train_xl.py`.
  Not yet wired as a served engine or published; see `docs/kenning-xl-design.md`.

## Models

### kenning-large-v0.6 (experimental; branch `kenning-xl`) — 2026-10
Decoder + readout, with gold-trace deliberate reasoning. A **two-tier cascade** routes each question:
one-pass Kenning-XL for text/agent/conversation, gold-trace **deliberate** Kenning-XL for records/tables,
the v0.5 reflex for logs/quality.

General suite, per-family mean (same basis for all columns):

| | v0.6 cascade | v0.5 | Clef | Jev |
|---|---|---|---|---|
| **macro (as served)** | **0.688** | 0.653 | 0.791 | 0.830 |
| agent | **0.867** | 0.727 | 0.793 | 0.900 |
| conversation | 1.000 | 0.979 | 1.000 | 1.000 |
| quality | **0.479** | 0.479 | 0.447 | 0.498 |
| text | 0.778 | 0.756 | 0.841 | 0.834 |
| table | 0.740 | 0.520 | 0.860 | 0.940 |
| records | 0.654 | 0.587 | 0.857 | 0.921 |
| logs | 0.520 | 0.520 | 0.740 | 0.720 |

- Served as one engine (`kenning/xl_serve.py`) with confidence- and size-gated deliberate escalation:
  **macro 0.688** (v0.5 0.653), **beats Clef on agent** (0.860), ties conversation, lifts records
  (0.587→0.663) and tables (0.520→0.660). A per-family *oracle* router reaches 0.720 — the ceiling a
  smarter policy and log/table counting-traces would approach.
- **Gold-trace training works and transfers:** trained on reasoning chains the rule generators compute for
  free (`kenning/traces.py`), the held-out arithmetic task over-daily-limit went 0.42 → **0.81**.
- Honest limits: still behind Clef on records, tables, text and logs. Deliberate mode *hurts* log-counting
  and some tasks, so it is routed selectively. Next: log/table counting traces, a 4B backbone, a served
  router engine, a full single-engine cascade benchmark.

### kenning-large-v0.5 — 2026-10 (active)
Same 435M cross-encoder, data redesigned for structured state: rule-generated records/tables/agent/logs
with exact labels, teacher-written problem cases, added public splits (HelpSteer2, jailbreak, injection,
ROPES). Trained at 512 tokens; not Clef-distilled (Clef batch endpoint hangs on long inputs).

- General macro **0.679** (v0.4 0.625; Clef 0.813; Jev 0.840 — 30-question macro basis).
- Big gains: tool-call match 0.50→0.84, jailbreak 0.64→0.96, failing-service 0.08→0.52, answer-helpfulness
  0.33→0.44 (best of the three). Numeric records/tables still weak (the cross-encoder ceiling → v0.6).

### kenning-large-v0.4 — 2026-10 (published, Apache-2.0)
Clean-licence zero-shot base (`deberta-v3-large-zeroshot-v2.0-c`) + Clef soft labels (α=0.5), multi-task +
phishing data. First published model: https://huggingface.co/systemonedev/kenning-large-v0.4
General macro 0.625; strong on text/safety, weak on structured state.

### Earlier (see docs/kenning.md)
- **v0.3** — clean-licence recipe (not served).
- **v0.2** — layout variation + teacher-written modern emails.
- **v0.1** — first multi-task model (began as a phishing classifier).
