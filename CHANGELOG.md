# Changelog

Model versions and engine changes for SystemOne Builder and Kenning. Benchmarks are macro accuracy on
the `general` suite (1,328 held-out items; `systemone bench --suite general`). Clef is `clef-flash`, Jev is
`jev-latest`, both run for comparison only — never trained on.

## Engines

SystemOne models all answer the same wire format (`POST /v1/systemone`; typed questions → calibrated
probabilities, no text). Two engine families now exist:

- **Kenning** (the reflex) — a **cross-encoder** that scores each option in one pass. Two published
  Apache-2.0 models: **v0.5** (435M DeBERTa, ≤512 tokens, ~35 ms — the fast option) and **v0.6** (≈400M
  ModernBERT, 2,048-token long context, ~291 ms, general macro **0.761**; beats Clef on
  agent/conversation/quality/logs). Neither is strictly better: v0.6 leads on our own suite, v0.5 on
  WorkflowEvals. `kenning/model.py`, served by `kenning/serve.py`.
- **Kenning-XL** (experimental decoder) — a small **decoder** (Qwen3-1.7B / 4B) with a
  **constrained answer-token readout**: renders (state, question) and reads the next-token distribution
  over the answer tokens in one pass, so the full LM reasons over the whole 4k-token state. An optional
  **deliberate mode** generates a short reasoning trace first, then reads the answer — for numeric /
  policy / table decisions the one-pass readout can't do. `kenning/xl.py`, trained by `kenning/train_xl.py`.
  Not yet wired as a served engine or published; see `docs/kenning-xl-design.md`.

## Models

### kenning-phishing-v1.1 & kenning-phishing-long-v1.1 — 2026-11 (Apache-2.0)
Domain-specialised System One phishing triage models built to sit at the front of an email/web security pipeline, trained with 50% Stage-0 signal dropout, `Clef-flash` soft-label distillation (`alpha=0.5`), and `--label-smoothing 0.05`, plus a two-stage cascade (`kenning/cascade.py`):

1. **Train/serve token-budget alignment (`kenning/train.py`, `kenning/model.py`)**: Extracted `budget_premise()` and applied `budget_groups()` in `train.py` so every hypothesis in a softmax group sees the exact same truncated state evidence.
2. **Conditional `noul_criteria` vs `noul_bare` decision bias (`kenning/model.py`, `kenning/train.py`, `kenning/calibrate.py`)**: Fits and serves separate temperatures and additive decision biases for `noul_criteria` and `noul_bare` (falling back to `noul` for older checkpoints).
3. **Stage-0 deterministic email & webpage pre-parsers + signal dropout (`kenning/phishing_data.py`, `system_one/phreshphish_suite.py`)**: Extracts structural signals (`link_vs_sender_domain_mismatch`, `reply_to_mismatch`, `any_auth_failed`, `lookalike_domain_suspected`, `sharer_same_org`, `recipient_org_count`, `attachment_risk_tier`) and drops `signals` on 50% of `structured_json` training rows so the model learns to read raw headers and links directly (`raw_email` accuracy improved `0.656 -> 0.725`).
4. **Honest benchmark split & two-stage cascade (`kenning/cascade.py`, `system_one/bench.py`)**: Keeps `phishing_counterfactual` strictly to the 48 hand-written held-out items (`p*`, `q*`), reports the 32 template-generated pairs separately as `phishing_counterfactual_synthetic`, evaluates `phreshphish` on 80 real `phreshphish/phreshphish` test-split webpages, and adds `cascade:<model>` (Pass 1 `is_phish` gate in ~33 ms; Pass 2 remaining 5 questions only when `0.05 < p < 0.92`).

| Benchmark Suite | **kenning-phishing-v1.1** (DeBERTa-v3 435M) | **kenning-phishing-long-v1.1** (ModernBERT 395M) | cascade (v1.1) | kenning-large-v0.6 (general) | Clef-flash (9B teacher) |
|---|---:|---:|---:|---:|---:|
| **`phishing_counterfactual` (48 hand-written held-out) — 6-task macro** | 0.774 | **0.795** | 0.777 | 0.413 | 0.701 |
| …`is_phish` accuracy @ 0.5 / ROC-AUC | 0.729 / 0.809 | 0.729 / 0.842 | 0.729 / 0.809 | 0.667 / 0.705 | **0.812** / **0.877** |
| …`is_phish` counterfactual pair-flip accuracy | 0.458 | 0.458 | 0.458 | 0.333 | **0.625** |
| …`is_phish` phishing auto-closed as safe (`0.9 / 0.1` gate) | **0** | **0** | **0** | 3 | **0** |
| …`recommended_action` (deliver / escalate / quarantine) | 0.625 | **0.708** | 0.622 | 0.396 | 0.479 |
| **`phishing_counterfactual_synthetic` (32 in-dist templates) — Macro** | 0.990 | **1.000** | 0.986 | 0.365 | 0.818 |
| **`phishing_layouts` (160 items across 4 layouts) — Macro / `raw_email`** | 0.756 / 0.725 | 0.719 / 0.700 | 0.756 / 0.725 | 0.556 / 0.550 | **0.769** / **0.750** |
| **`phreshphish` (80 real webpages) — `is_phish` / ROC-AUC** | **0.950** / **0.993** | 0.925 / 0.988 | **0.950** / **0.993** | 0.600 / 0.781 | 0.938 / 0.978 |
| **`PhishNChips` (400 real DI #56 emails) — `is_phishing` / ROC-AUC** | 0.662 / 0.709 | 0.713 / 0.810 | — | 0.510 / 0.633 | **0.748** / **0.852** |
| **`decision_index` (320 stratified real DI 0.3 items, 32/37 tracks)** | 22.4 | 22.1 | — | 27.4 (v0.6) | **61.9** |
| **`modern2` (20 subtle body-only lures) — Accuracy / ROC-AUC** | 0.800 / 0.900 | 0.700 / 0.920 | 0.800 / 0.900 | 0.650 / 0.890 | **0.900** / **0.960** |
| Latency p50 (`phishing_layouts` / `modern2` / `PhishNChips`) | **42 / 33 / 62 ms** | 99 / 138 / **34 ms** | **42 / 37 / — ms** | 33 / 44 / 49 ms | 915 / 367 / 238 ms |

### Measurement fixes and a third-party benchmark — 2026-10
The first run against a suite we did not write ([WorkflowEvals](https://huggingface.co/datasets/typesafe/workflowevals),
Apache-2.0, never trained on) put **v0.6 at 0.463 and v0.5 at 0.552**, against a hosted engine's 0.800 — the
reverse of our own suite. Three of the defects were ours:

- **noul criteria were folded into both hypotheses**, leaving the two strings the cross-encoder compares
  byte-identical apart from `yes.`/`no.`, so the question's most discriminating field carried no signal;
- **`truncation="only_first"` gave each option a different share of the state**, so a question's softmax
  compared scores taken over different evidence (21% of an 8,516-token invoice survived, 199-token spread
  across options);
- **`score` was graded by rounding an expected value**, which penalises only engines that answer with a
  distribution (0.267 for us by rounding vs 0.400 by argmax; a hosted engine scored 0.400 either way).

| WorkflowEvals (240 items) | original | all three fixes |
|---|---|---|
| v0.5 | 0.552 | **0.625** |
| v0.6 | 0.463 | **0.575** |
| GLiDE (hosted) | 0.800 | 0.800 |

Gap to the hosted engine **0.337 -> 0.175**, with no retraining; our own suite moved 0.753 -> 0.761. Honest
caveat: yes/no AUC went 0.598 -> 0.576, so part of that gain is base-rate alignment rather than new
discriminative power. **v0.5 is the stronger model on this suite and v0.6 on ours**; the `general` suite is
largely rule-generated by us and should be read as a development signal, not a ranking.

### kenning-large-v0.7 — 2026-10 (trained, NOT published)
Every noul question in our training data through v0.6 carried no `criteria` field, so the model had never
learned to read it. v0.7 trains on a dataset where 60% of noul questions carry criteria built from the label
descriptions the generators already hold, with everything else held constant (same 37,678 rows, question-type
mix within 0.2%). Held-out 0.9048 / ECE 0.0039, against v0.6's 0.905 / 0.006.

**The training worked and we are still not shipping it.** Discrimination improved — yes/no AUC 0.593 -> 0.622
and **best-threshold accuracy 0.702 -> 0.720, the highest we have measured** — but the probability scale became
dependent on whether criteria are present: the optimal yes/no threshold is **0.621 on WorkflowEvals, where
every question has criteria, and 0.152 on our general suite, where none do**. No single 0.5 cut serves both, so
as served it scores 0.575 on WorkflowEvals (flat) and regresses our general macro to 0.722. Not a label leak:
the yes-rate is 0.456 with criteria and 0.458 without.

This exposed a gap in our own tooling. For a two-option question `p >= 0.5` is equivalent to
`score_yes >= score_no`, which is temperature-invariant, so `systemone calibrate` can improve ECE but **cannot
move a noul decision boundary**. Correcting it needs a per-question-type bias term, which we do not have yet.

### kenning-large-v0.6 — 2026-10 (published, Apache-2.0)
The long-context reflex: a ≈400M **ModernBERT cross-encoder** (base `tasksource/ModernBERT-large-nli`) over a
2,048-token window, so it sees the whole state in one pass. General macro **0.761** (v0.5 0.653; 0.753 as
first measured, before the rendering fixes above), held-out 0.905, ECE 0.006. On WorkflowEvals it scores
0.575, behind v0.5's 0.625. Published: https://huggingface.co/systemonedev/kenning-large-v0.6

| | v0.6 | v0.5 | Clef | Jev |
|---|---|---|---|---|
| **macro** | **0.761** | 0.653 | 0.791 | 0.830 |
| agent | **0.827** | 0.727 | 0.793 | 0.900 |
| conversation | 1.000 | 0.979 | 1.000 | 1.000 |
| quality | **0.457** | 0.479 | 0.447 | 0.498 |
| text | 0.813 | 0.756 | 0.841 | 0.834 |
| table | 0.680 | 0.520 | 0.860 | 0.940 |
| records | 0.681 | 0.587 | 0.857 | 0.921 |
| logs | **0.810** | 0.520 | 0.740 | 0.720 |

- **Beats Clef on agent, conversation, answer quality and logs** (logs 0.79 beats Clef *and* Jev); closes
  most of the general-suite gap to Clef (0.791) at a twentieth of its size.
- The long context unlocks logs (0.52 → 0.79) and lifts tables/records without the decoder's deliberate
  reasoning — the reflex sees enough of the state in one pass.
- Trade-off: ~285 ms p50 vs v0.5's 35 ms (2,048-token ModernBERT-large). **v0.5 stays the fast option.**

### Kenning-XL (experimental decoder) — 2026-10
Decoder + readout, with gold-trace deliberate reasoning. A **two-tier cascade** routes each question:
one-pass Kenning-XL for text/agent/conversation, gold-trace **deliberate** Kenning-XL for records/tables,
the v0.5 reflex for logs/quality.

General suite, per-family mean (same basis for all columns):

| | XL cascade | v0.5 | Clef | Jev |
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
