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
Domain-specialised System One phishing triage models built to sit at the front of an email/web security pipeline ([`systemonedev/kenning-phishing-v1.1`](https://huggingface.co/systemonedev/kenning-phishing-v1.1) and [`systemonedev/kenning-phishing-long-v1.1`](https://huggingface.co/systemonedev/kenning-phishing-long-v1.1)), trained with Stage-0 structural signals + 50% signal dropout, `Clef-flash` soft-label distillation (`alpha=0.30`), hard-label empirical temperature/bias calibration, and a two-stage state-aware cascade (`kenning/cascade.py`):

1. **Train/serve token-budget alignment (`kenning/train.py`, `kenning/model.py`)**: Extracted `budget_premise()` and applied `budget_groups()` in `train.py` so every hypothesis in a softmax group sees the exact same truncated state evidence.
2. **Hard-label calibration & conditional `noul_criteria` vs `noul_bare` decision bias (`kenning/model.py`, `kenning/train.py`, `kenning/calibrate.py`)**: Fits per-bucket temperatures (`T`) and balanced-accuracy decision biases against empirical hard labels (avoiding soft-target probability compression and restoring `57.5%–83.2%` gate automation on real email).
3. **Stage-0 universal email & webpage pre-parsers + signal dropout (`kenning/phishing_data.py`, `kenning/model.py`, `system_one/phreshphish_suite.py`)**: Normalizes structured, flat (`PhishNChips`), nested (`modern2`), and RFC822 states at inference time to extract structural signals (`link_vs_sender_domain_mismatch`, `ephemeral_or_shortener_link`, `trusted_cross_domain_link`, `reply_to_mismatch`, `auth_failure`, `lookalike_or_homoglyph_domain`, `role_vs_address_mismatch`, `saas_platform_claim_mismatch`, `recipient_role_vs_document_mismatch`, `sharer_same_org`, `multi_org_recipient_spread`) and drops `signals` on 50% of training rows so raw inputs remain strong (`0.725`).
4. **Honest benchmark split & two-stage state-aware cascade (`kenning/cascade.py`, `system_one/bench.py`)**: Keeps `phishing_counterfactual` strictly to the 48 hand-written held-out items (`p*`, `q*`), reports the 32 template-generated pairs separately as `phishing_counterfactual_synthetic`, evaluates `phreshphish` on 80 real test-split webpages and `PhishNChips` on 400 real DI #56 emails, and runs `cascade:<model>` (~34–49 ms Pass 1 gate + Pass 2 diagnostic routing).

| Benchmark Suite | **kenning-phishing-v1.1** (DeBERTa-v3 435M) | **kenning-phishing-long-v1.1** (ModernBERT 395M) | cascade (v1.1) | kenning-large-v0.6 (general) | Clef-flash (9B teacher) |
|---|---:|---:|---:|---:|---:|
| **`phishing_counterfactual` (48 hand-written held-out) — 6-task macro** | 0.826 | **0.868** | 0.806 | 0.399 | 0.722 |
| …`is_phish` accuracy @ 0.5 / ROC-AUC | 0.854 / 0.953 | **0.917** / **0.972** | 0.854 / 0.953 | 0.646 / 0.693 | 0.833 / 0.905 |
| …`is_phish` counterfactual pair-flip accuracy | 0.708 | **0.833** | 0.708 | 0.292 | 0.667 |
| …`is_phish` auto-triage (`0.9 / 0.1` gate) / FN auto-closed | 60.4% / **0** | **75.0%** / 1 | 60.4% / **0** | 56.2% / 3 | 25.0% / **0** |
| …`recommended_action` (deliver / escalate / quarantine) | **0.854** | 0.833 | 0.833 | 0.375 | 0.500 |
| **`phishing_counterfactual_synthetic` (32 in-dist templates) — Macro** | 0.995 | **1.000** | 0.844 | 0.464 | 0.823 |
| **`phishing_layouts` (160 items across 4 layouts) — Macro / `raw_email`** | **0.856** / 0.725 | 0.825 / 0.725 | **0.856** / 0.725 | 0.556 / 0.550 | 0.781 / **0.750** |
| **`phreshphish` (80 real webpages) — `is_phish` / ROC-AUC** | **0.963** / **0.992** | 0.925 / 0.983 | **0.963** / **0.992** | 0.600 / 0.781 | 0.938 / 0.978 |
| **`PhishNChips` (400 real DI #56 emails) — `is_phishing` / ROC-AUC** | 0.900 / 0.974 | **0.953** / **0.984** | — | 0.510 / 0.633 | 0.748 / 0.852 |
| …`PhishNChips` `verdict` (`choice`) / auto-triage (`0.9 / 0.1`) | **0.905** / 57.5% | 0.870 / **83.2%** | — | 0.527 / 45.8% | 0.743 / 41.2% |
| …`PhishNChips` Recall @ 1% FPR | 0.875 | **0.945** | — | 0.240 | 0.315 |
| **`decision_index` (320 stratified real DI 0.3 items, 32/37 tracks)** | 22.9 | 22.8 | — | 27.4 (v0.6) | **61.9** |
| …DI areas: Retrieval / Lang / Tools / Knowledge / Arts | **31.8** / 26.1 / 22.0 / 15.7 / 17.2 | **36.2** / 30.7 / 16.8 / 13.6 / 10.5 | — | 22.5 / 52.0 / 19.8 / 14.9 / 19.5 | 25.4 / **76.7 / 77.8 / 62.9 / 65.5** |
| **`modern2` (20 subtle body-only lures) — Accuracy / ROC-AUC** | **0.950** / **1.000** | 0.750 / 0.950 | **0.950** / **1.000** | 0.650 / 0.890 | 0.900 / 0.960 |
| Latency p50 (`phishing_layouts` / `modern2` / `PhishNChips` / `phreshphish`) | 41 / 32 / 75 / 120 ms | **35 / 23 / 54 / 86 ms** | 42 / 35 / — / **44 ms** | 32 / 18 / 49 / 91 ms | 210 / 113 / 238 / 321 ms |

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
