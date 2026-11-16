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

### kenning-phishing-v1 & kenning-phishing-long-v1 — 2026-11 (Apache-2.0)
Domain-specialised System One phishing triage models built to sit at the front of an email security pipeline, plus three core fixes to the Kenning training and calibration stack:

1. **Train/serve token-budget alignment (`kenning/train.py`, `kenning/model.py`)**: `train.py` previously encoded `(premise, hypothesis)` with `truncation="only_first"`, giving shorter hypotheses more state tokens than longer ones within the same softmax group (whereas `model.py` pre-truncated the premise to `max_length - max(hyp_tokens) - 4`). Extracted `budget_premise()` in `model.py` and applied it in `train.py` (`budget_groups()`), eliminating the train/inference token mismatch.
2. **Conditional `noul_criteria` vs `noul_bare` decision bias (`kenning/model.py`, `kenning/train.py`, `kenning/calibrate.py`)**: Following the v0.7 finding that noul questions with a `criteria` dict operate on a shifted logit scale relative to bare yes/no questions, `calibration_key(q)` now fits and serves separate temperatures and additive decision biases for `noul_criteria` and `noul_bare` (falling back to `noul` for older checkpoints).
3. **Stage-0 deterministic email pre-parser & paired counterfactual training (`kenning/phishing_data.py`, `problems/phishing_triage.json`)**: Extracts structural signals (`link_vs_sender_domain_mismatch`, `reply_to_mismatch`, `any_auth_failed`, `lookalike_domain_suspected`, `sharer_same_org`, `recipient_org_count`, `attachment_risk_tier`) before scoring, and trains with group-aware pair splitting (`pair:<pair_id>`) so counterfactual twins never leak across train/calibration/benchmark splits.
4. **Multi-provider frontier & local benchmarking (`system_one/factory.py`, `system_one/bench.py`, `adapters/anthropic.py`)**: Added direct in-process engines (`kenning:<path>`, `xl:<model>`, `clef-direct`) and frontier LLM providers (`openai:<model>`, `anthropic:<model>`, `gemini:<model>`, `openrouter:<model>`, `ollama:<model>`), plus ROC-AUC, PR-AUC, Recall @ 1% FPR, counterfactual pair-flip accuracy, and cost per 100k items.

| Benchmark Suite | **kenning-phishing-v1** (DeBERTa-v3 435M, 512t) | **kenning-phishing-long-v1** (ModernBERT 395M, 1280t) | kenning-large-v0.6 (general) | kenning-large-phishing-v0.1 | Qwen2.5-7B-Instruct (XL readout) | Clef-flash (9B teacher) |
|---|---:|---:|---:|---:|---:|---:|
| **`phishing_counterfactual` (80 items / 40 pairs) — Macro (6 tasks)** | 0.867 | **0.888** | 0.394 | 0.342 | 0.602 | 0.748 |
| `is_phish` accuracy @ 0.5 | **0.825** | **0.825** | 0.650 | 0.512 | 0.500 | 0.775 |
| `is_phish` counterfactual pair-flip accuracy | 0.650 | **0.675** | 0.300 | 0.025 | 0.000 | 0.550 |
| `is_phish` ROC-AUC / Recall @ 1% FPR | 0.910 / **0.475** | **0.912** / 0.375 | 0.649 / 0.050 | 0.621 / 0.025 | 0.588 / 0.175 | 0.861 / 0.375 |
| `is_phish` Brier / ECE (lower better) | 0.125 / 0.093 | **0.124** / **0.060** | 0.299 / 0.270 | 0.401 / 0.414 | 0.500 / 0.500 | 0.157 / 0.122 |
| `is_phish` accuracy when automated (0.9 / 0.1) | **100.0%** (0 FN) | 96.4% (0 FN) | 66.7% (3 FN) | 53.2% (28 FN) | 50.0% (40 FN) | **100.0%** (0 FN) |
| `recommended_action` (3-way routing) accuracy | 0.812 | **0.850** | 0.325 | 0.388 | 0.575 | 0.487 |
| **`phishing_layouts` (128 items across 4 layouts) — Macro** | **0.766** | 0.750 | 0.555 | 0.508 | 0.500 | 0.750 |
| **`phreshphish` (80 webpages, Stage-2 DOM/URL triage) — `is_phish` / Macro** | **1.000** / **1.000** | **1.000** / 0.829 | 0.900 / 0.929 | — | **1.000** / 0.983 | **1.000** / **1.000** |
| **`modern2` (20 subtle body-only lures) — Accuracy / ROC-AUC** | 0.800 / **0.970** | 0.800 / 0.850 | 0.650 / 0.890 | 0.500 / 0.660 | 0.500 / 0.640 | **0.900** / 0.960 |
| **`decision_index` (100 items, 5 areas) — Index (0..100) / Macro** | 38.1 / 0.638 | **57.2** / **0.760** | 49.2 / 0.721 | — | 72.8 / 0.843 | **97.9** / **0.990** |
| Latency p50 (`phishing_layouts` / `modern2`) | 39 ms / 37 ms | **30 ms / 18 ms** | 33 ms / 19 ms | 32 ms / 21 ms | 124 ms / 46 ms | 234 ms / 130 ms |

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
