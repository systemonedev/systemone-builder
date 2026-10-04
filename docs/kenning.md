# Kenning: the local System One model

Kenning is a **System One** decision model: program **state** plus typed **questions** in,
**answers** with calibrated probabilities out, in one forward pass and without generating text.
The category and the wire format were introduced by TypeSafe AI with Jev, its hosted model;
Kenning is an open, locally run model that speaks a compatible format, and Jev is one of the
systems it is benchmarked against. Kenning is not affiliated with TypeSafe AI and is never trained
on TypeSafe outputs.

## How it works

The model is a non-generative cross-encoder (default base:
[`MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c),
435M parameters, MIT, no non-commercial data; v0.1 and v0.2 used ModernBERT-large-zeroshot-v2.0).
Each candidate answer becomes a hypothesis scored against the state:

| Question | Hypotheses | Answer |
|---|---|---|
| `noul` | `<instructions> Answer: yes.` / `... no.` | `noul` = P(yes) |
| `choice` | `<instructions> Answer: <option>. <description>` per option | softmax over options |
| `score` | `<instructions> Answer: <level>.` per level | softmax over levels; `score` = probability-weighted level |

All hypotheses of all questions in a request are scored in one batch. Distributions are
`softmax(scores / T)` with one temperature per question type, fitted on held-out data after
training, which is what calibrates the probabilities. `confidence` is `(max p − 1/n)/(1 − 1/n)`.

**Deterministic:** a request is its own batch, requests run one at a time with deterministic
kernels, and the softmax runs in float64. The same request gives bit-identical answers.

## Serving, the UI and your own code

The compose service `kenning` (GPU 0, about 2 GB of VRAM; part of the default stack) serves `POST /v1/systemone` on
`127.0.0.1:8093`, and the API exposes it as `POST /api/v1/systemone`
(`S1_SYSTEM_ONE_BACKEND=kenning`, the default; `logprob` switches to an LLM's label-token readout).

In the dashboard:

- **Models** lists every trained model in the workspace volume (`kenning/models/`) with its
  held-out results and training data. **Activate** hot-swaps the served model (no restart) and
  persists the choice in `kenning/active.json`; without it, `S1_KENNING_MODEL` (a model directory or
  a Hugging Face cross-encoder id, default the zero-shot base) is served. **Export** builds a zip
  with the weights, tokenizer, `kenning.json`, a model card, `NOTICE.md` with the licences of the
  base model and every training source, and `SHA256SUMS`.
- **Train** runs the whole recipe without a terminal:
  - **1 · Training data** builds a dataset from the public sources (`systemone data`; the defaults
    are the clean-licence recipe) and lists every dataset with its sources and licences.
  - **2 · Label with a teacher** distils Clef's probabilities into a dataset (`systemone label`)
    when the `clef` profile is running.
  - **3 · Train** fine-tunes a base model (the recommended base is MIT with no non-commercial
    data).
  - Jobs run one at a time, with live logs and progress.
  - A job that needs the GPU pauses the services on it (Kenning; the student with the pipeline;
    Clef when it isn't the teacher), waits until enough VRAM is free (`S1_KENNING_TRAIN_FREE_MB`,
    default 19500), and restarts them when it ends, whether it succeeds, fails or is cancelled.
- **Verify** benchmarks the active model on the held-out suites against Clef, Jev (opt-in, you
  confirm that each item is a paid request) or the pipeline's LLM read-out.
  - The **scoreboard** shows the latest result per suite and model: accuracy, how much the model
    would decide on its own, and how many threats it was sure were harmless.
  - **Runs** opens any past run item by item, filtered to the mistakes.
- **Try it** asks your own questions about any text or JSON, shows the probabilities, checks
  determinism (Run 10×), and can compare with TypeSafe Jev when you set `TYPESAFE_API_KEY` (each
  comparison is a paid TypeSafe request under your own TypeSafe agreement; off by default).
- **Connect** has the code: the [`systemone`](../clients/python) client library (HTTP, or an
  exported bundle in-process with `Kenning.from_pretrained`), curl and plain HTTP.

## Training

```bash
# training rows for the built-in phishing task (benchmark emails are excluded)
docker compose exec api systemone data --task phishing -n 4000
# multi-task rows: six public labelled datasets plus phishing (see "Training data" below)
docker compose exec api systemone data --task multitask --per-source 1200
# free GPU 0 (with the pipeline profile, also stop the student), train, restore
docker compose stop kenning
docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train \
  --data /workspace/kenning/datasets/multitask-train.jsonl --out /workspace/kenning/models/kenning-large-v0.1 --max-length 512
docker compose up -d --no-deps student         # pipeline profile only
# serve it: Models page -> Activate (or S1_KENNING_MODEL in .env), or start the service:
docker compose up -d --no-deps kenning
```

Training data is JSONL, one decision context per line:
`{"state": ..., "questions": {qid: {type, instructions, criteria}}, "targets": {qid: target}}`.
Targets are hard labels (noul 0/1, choice option, score level) or soft distributions from a teacher
(noul P(yes), `{option: p}`, `{"0": p, ...}`), so decisions can be distilled from a System 2 model
with their uncertainty. Training uses gradient checkpointing and a hard VRAM cap
(`--max-vram-fraction`, default 0.85): exceeding it fails with an out-of-memory error instead of
spilling into shared system memory, which on WSL2 is slow and destabilising. The held-out metrics
before training (zero-shot), after, and after calibration are written to `kenning.json` next to
the weights.

### Training data

A model trained on one task learns that task's answer habits: trained only on phishing, the first
model answered "yes" to most yes/no questions about anything else. The multi-task set
(`systemone data --task multitask`, built by `systemone_builder.kenning.multitask`) maps public labelled datasets
onto the three question types and writes a licence manifest next to the JSONL:

| Source | Licence | Teaches |
|---|---|---|
| `fancyzhx/amazon_polarity` | Apache-2.0 | sentiment as noul and choice |
| `fancyzhx/dbpedia_14` | CC-BY-SA-3.0 | topic choice over 4–10 of 14 classes; "is it about X?" |
| `clinc/clinc_oos` (plus) | CC-BY-3.0 | intent routing over 5–15 of 150 intents, with an "other" bucket |
| `google/boolq` | CC-BY-SA-3.0 | reading-comprehension yes/no |
| `nyu-mll/multi_nli` | CC-BY-3.0 (mixed per genre) | "does the text imply …?", 3-way choice |
| `google/civil_comments` | CC0-1.0 | soft yes/no targets (annotator fractions), 4-level score |
| phishing rows | see dataset card | the original task |

Class questions are asked with the true and a wrong class equally often, and genre questions across
sources ("is this a product review?" of an encyclopedia article) are mostly "no".

### Train for your own problem

A general model is a starting point. For a specific problem, say gating a computer-use agent's
actions, triaging insurance claims or moderating a chat, add cases for that problem to the training
set. You can describe the problem and have a teacher LLM write cases, bring your own labelled data, or
both. Keep some general data in the mix (`--per-source`) so the model doesn't lose its general skill.

**Problem specs** (`kenning/problems.py`). A spec names the state's fields and lists every question
with all its answers, each described:

```json
{
  "name": "computer_use",
  "title": "Computer-use agent: check the next action",
  "writes": "one step of a computer-use agent operating a web browser or desktop app for a user",
  "state": {"fields": {"goal": "the user's task", "screen": "what is visible now",
                       "history": "the previous actions", "proposed_action": "the next action"}},
  "questions": [
    {"id": "needs_confirmation", "ask": ["Should the agent ask the user before taking this action?"],
     "labels": {"yes": "spends money, sends, deletes, shares personal data", "no": "reading, navigating, drafting"}},
    {"id": "risk", "ask": ["How risky is the proposed action?"], "ordinal": true,
     "labels": {"low": "harmless, easy to undo", "medium": "inconvenient to undo", "high": "irreversible or costly"}}
  ],
  "constraints": [{"if": {"risk": "high"}, "then": {"needs_confirmation": "yes"}}]
}
```

- `state` is `{"fields": {...}}` for a record (1–12 named fields) or `{"text": "<key>"}` for one text.
- Yes/no questions become nouls, `"ordinal": true` questions become scores, and the rest become
  choices. Training rows also rephrase each question as "is the answer X?".
- `constraints` rule out combinations that can't happen, so the teacher is never asked to write one.
- Built in: `computer_use`, `insurance_claim`, `chat_moderation` (`systemone problems`). Save your own
  on the Train page, or pass a path: `--problem my_spec.json=2000`.

**How cases are written.**
1. For each case, answers are drawn at random (respecting the constraints), and the teacher writes a
   case that has them. Every label is known by construction, never guessed.
2. With the **blind check** (on by default), the teacher then answers the questions about its own case
   without seeing the intended answers. The case is kept only if every answer matches.
3. Writing continues until the requested number of cases is kept, up to 5 attempts per case.
4. The log reports how many failed the check. A teacher that keeps fewer than 20% is the wrong teacher
   for the problem.

The teacher is the pipeline's triage model, or any OpenAI-compatible server (`--teacher-url`,
`--teacher-model`, key in `S1_TEACHER_API_KEY`). In a test on `computer_use`, Qwen2.5-7B kept 23% of
its cases. A larger teacher keeps more and writes better ones. Labelling the dataset with Clef
afterwards (`systemone label`) reports how often a second model agrees, per source. Check the
teacher's licence: its outputs become your training data.

**Importing your own data** (`kenning/importer.py`). JSONL, one example per line, in one of two formats:
- training rows: `{"state", "questions", "targets"}`. Targets are hard (0/1, an option, a level index)
  or soft (P(yes), `{option: p}`).
- labelled items: `{"state", "labels"}`, with the questions in `<file>.questions.json` next to the file.

Every line is checked against the System One contract first, and the first errors are reported with
their line numbers. On the Train page, uploads ask for a licence, which goes into the dataset manifest
and the model card.

**Hold-out suites.** By default (`--holdout 0.1`), 10% of each problem and import, and at least 20
items, is held out before any training row is written. Each is saved as a benchmark suite with one
fixed question per id (`<dataset>.holdout/`). Verify lists it under *Your problems*. From the command
line:

```bash
docker compose exec api systemone bench --engines kenning,clef \
  --suite /data/workspace/kenning/datasets/agent-v1.holdout/problem-computer_use.jsonl \
  --questions /data/workspace/kenning/datasets/agent-v1.holdout/problem-computer_use.questions.json
```

Teacher-written hold-outs measure how well the student learned the teacher's labels. For a decision
that matters, also label a few hundred real cases yourself and import them; their hold-out is the
honest number.

**Calibrating on your data** (`kenning/calibrate.py`).
- Temperatures are fitted on the training distribution. On different data, the probabilities drift.
- `systemone calibrate` asks the served model the questions in your labelled file. It recovers each
  option's score from the returned probabilities and refits one temperature per question type.
- A temperature never changes which answer wins, so accuracy stays the same and only the probabilities
  move.

```bash
docker compose exec api systemone calibrate my-model --data /data/workspace/my-labels.jsonl          # report only
docker compose exec api systemone calibrate my-model --data /data/workspace/my-labels.jsonl --write  # save
docker compose exec api systemone calibrate systemonedev/kenning-large-v0.4 \
  --data /data/workspace/my-labels.jsonl --write --save-as kenning-v0.4-mine                        # a calibrated copy
```

The model must be the one currently served. The old temperatures are kept in `kenning.json` under
`calibration_history`. Activate the model again to serve the new ones.

## Benchmarks

`systemone bench --suite phishing` uses 50 unseen, balanced emails from the phishing dataset;
`--suite ood` uses tasks never trained on (`ucirvine/sms_spam`, `dair-ai/emotion`,
`fancyzhx/ag_news`, `-n` items per task); `--suite layouts` asks about the same 50 phishing emails in
four layouts (the training layout, `from`/`subject`/`body` JSON, plain text with headers, nested with
metadata) to measure layout sensitivity; `--suite modern` is 20 hand-written short, modern emails
(evaluation only, never trained on). Engines: `kenning` (this model), `jev` (opt-in TypeSafe Jev with
your own key, `TYPESAFE_API_KEY`), `local` (an LLM's label-token readout), `llm` (an LLM writing JSON).

### General suite (`--suite general`): the headline

The measure for a general-purpose decision model: 30 questions in 7 families, ~50 items each (1,328
items; 148 of them ask several questions about the same state).

| Family | What the state is | Labels from |
|---|---|---|
| text | the 14 `multi` tasks below | held-out public datasets |
| table | a table (≤ 12 rows) and a statement about it | TabFact test set |
| conversation | a multi-turn booking dialogue: which service is it about | GEM schema-guided dialogue, kept only when the conversation visibly names its service and no other |
| agent | available functions, a request and a call; a goal and the steps an agent took | Glaive function calling; generated |
| quality | a request and an assistant's answer: how helpful, is it correct | HelpSteer2 validation |
| records | JSON records with numbers, dates and policies: refunds, spending limits, access rules, ticket priority, census income | generated with exact rule-derived labels; adult census |
| logs | 60–140 log lines: is a service failing, which one | generated with labels computed from the log itself |

Generated cases come from `system_one/general_generators.py`. They are **benchmark only**: no model labels
them, so the suite has no bias toward Clef or any other teacher, and training data must never reuse
their templates or seeds. Logs are capped at ~2.5k tokens so the whole log fits Clef's 4k context.
Without the cap, the truncated part would be the last minutes, where the answer is. Sources and licences
are in `system_one/general_suite.py`.

```bash
docker compose exec api systemone bench --suite general --engines kenning,clef -n 50 --concurrency 1
```

Baseline (October 2026, one RTX 3090 each, items one at a time, 0 errors, both 5/5 identical on repeat):

| | Kenning v0.4 | Clef-flash | Gap |
|---|---|---|---|
| **Macro accuracy (30 questions)** | **0.625** | **0.813** | **−18.8** |
| text (14) | 0.754 | 0.841 | −8.7 |
| conversation (1) | 0.958 | 1.000 | −4.2 |
| table (1) | 0.480 | 0.860 | −38.0 |
| agent (3) | 0.553 | 0.793 | −24.0 |
| records (7) | 0.535 | 0.857 | −32.2 |
| logs (2) | 0.290 | 0.740 | −45.0 |
| quality (2) | 0.377 | 0.447 | −7.0 |
| Latency p50 / p95 | 34 / 61 ms | 149 / 382 ms | |

What it says:
- **On structured state, Kenning is near chance** on several tasks: tool call matches the request (0.50 vs
  1.00), over the daily limit (0.50 vs 0.92), table statement (0.48 vs 0.86). A refund (0.25 vs 0.92) is
  eligible in only a quarter of cases, so a model that always says yes scores 0.25.
- **Logs show the context limit.** Kenning reads 512 tokens of state per option, so it never sees the
  last minutes of a log (failing service 0.08 vs 0.78).
- **Answer quality is hard for both** (helpfulness 0.33 / 0.35, correctness 0.42 / 0.54).
- Kenning answers about 4× faster. Clef's lead on records, tables, agent and logs is the case for the
  redesigned data (v0.5) and a longer-context model (Kenning-XL).

### Multi-task suite (`--suite multi`)

The text family of `general` on its own: 14 tasks, ~50 class-balanced items each (~680),
from held-out splits of permissively licensed datasets that Kenning is not trained on. Items are fetched
at run time with a fixed seed and cached (`bench_cache/multi-n50-seed42.json`); nothing is redistributed.
The report adds the **macro accuracy**: the mean of per-task accuracy (exact level for the score task).
Tasks, sources and licences are in `system_one/multitask_suite.py`.

```bash
docker compose exec api systemone bench --suite multi --engines kenning,clef -n 50 --concurrency 1
```

Baseline (October 2026, one RTX 3090 each, items one at a time). Clef is the development target. Jev is
compared again once Kenning matches or beats Clef.

| Task (dataset) | Type | Kenning v0.4 | Clef-flash | Gap |
|---|---|---|---|---|
| **Macro accuracy (14 tasks)** | | **0.754** | **0.838** | **−8.4** |
| Answerable from passage (BoolQ) | noul | 0.660 | 0.880 | −22.0 |
| Claim supported by evidence (SciTail) | noul | 0.620 | 0.940 | −32.0 |
| Financial sentiment (twitter-financial-news) | choice | 0.771 | 0.812 | −4.1 |
| Toxic comment (wiki_toxic) | noul | 0.840 | 0.900 | −6.0 |
| Counterfactual statement (amazon_counterfactual) | noul | 0.760 | 0.860 | −10.0 |
| Prompt injection (deepset) | noul | **0.720** | 0.620 | +10.0 |
| Jailbreak attempt (jailbreak-classification) | noul | 0.640 | 0.860 | −22.0 |
| Fine-grained emotion (go_emotions, 6) | choice | 0.792 | 0.917 | −12.5 |
| Toxicity level (real-toxicity-prompts, 3 levels) | score | 0.646 | 0.646 | 0 |
| Encyclopedia topic (dbpedia, 6) | choice | 0.958 | 1.000 | −4.2 |
| Banking intent (banking77, 8) | choice | 0.979 | 1.000 | −2.1 |
| SMS spam | noul | 0.840 | 0.900 | −6.0 |
| Emotion (6) | choice | **0.521** | 0.500 | +2.1 |
| News topic (ag_news) | choice | 0.812 | 0.896 | −8.4 |
| Latency p50 | | 33 ms | 141 ms | |

Both engines were deterministic (5/5 repeats identical). What it says:
- **Kenning's biggest gaps are reasoning over a supplied text**: whether a passage answers a question
  (−22) or supports a claim (−32). The clean training recipe left out BoolQ (CC-BY-SA), and its NLI data
  is phrased differently from these tasks. That points at training data for evidence and passage
  questions.
- **Jailbreak detection (−22)** and **fine-grained emotion (−12.5)** are the next gaps. Prompt injection
  is already a Kenning strength (+10 over Clef).
- **Routing and topic tasks are nearly solved** by both (≥ 0.96).
- Kenning answers about 4× faster. Matching Clef's macro accuracy at Kenning's speed is the v0.5 goal.

## Results so far

| Model | Training data | Held-out accuracy (zero-shot → trained) | Held-out ECE (calibrated) |
|---|---|---|---|
| `kenning-large-phishing-v0.1` | 4,000 phishing emails, 1 epoch, 10 min | 0.67 → 0.98 | 0.005 |
| `kenning-large-v0.1` | 8,700 rows from 7 sources, 1 epoch, ~15 min | 0.71 → 0.93 | 0.008 |

**Out of domain** (180 items, tasks neither local model was trained on):

| | kenning-large-phishing-v0.1 | kenning-large-v0.1 | TypeSafe Jev |
|---|---|---|---|
| Spam: accuracy | 0.867 | 0.967 | 0.967 |
| Spam: false positives acted on (p ≥ 0.9) | 5 | 0 | 1 |
| Spam: ECE | 0.129 | 0.048 | 0.049 |
| Spam: automated / accuracy when automated | 93% / 89% | 85% / 98% | 73% / 95.5% |
| Emotion (6 options): accuracy | 0.733 | 0.750 | 0.583 |
| News topic: accuracy | 0.917 | 0.933 | 0.933 |
| Latency p50, one at a time | – | 26 ms | – |

**Phishing** (50 unseen emails):

| | kenning-large-phishing-v0.1 | kenning-large-v0.1 | TypeSafe Jev | Qwen 7B label readout |
|---|---|---|---|---|
| Accuracy (malicious?) | 1.000 | 0.960 | 0.960 | 0.800 |
| Brier | 0.001 | 0.033 | 0.044 | 0.191 |
| ECE | 0.013 | 0.039 | 0.109 | 0.199 |
| Automated at p ≥ 0.9 / ≤ 0.1 | 98% | 96% | 58% | 96% |
| Accuracy when automated | 100% | 97.9% | 100% | 81% |
| False positives acted on / phishing auto-closed | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 9 |
| Category accuracy | 1.000 | 0.940 | 0.960 | 0.880 |
| Latency p50 / p95, one at a time | 33 / 51 ms | 61 / 98 ms | 149 / 210 ms | – |
| Identical answers on repeat | 10/10 | 10/10 | 1–4/10 | 8/10 |

Latency is per request with requests sent one at a time. The server answers one request at a time to
stay deterministic, so with many concurrent requests the extra time is queueing.

**Where it stands:** `kenning-large-v0.1` matches or beats Jev on every out-of-domain task, with
calibration as good or better and answers that never change on repeat. Multi-task training cost a
little on phishing: one confident false positive (a safe email it would have quarantined) where Jev
sent its borderline cases to a human. Until that is fixed (a stricter auto-act threshold such as
0.95 for that question, more training, or a heavier phishing share), keep a human on its automated
phishing decisions. 50 emails is a small sample; a larger held-out set would make the comparison
firmer.

### kenning-large-v0.2: layout variation and teacher-written modern emails

v0.1 had learned the layout along with the task: every phishing training row looked like
`{"email": {"recipient", "body"}}`, and the public phishing data is mostly old (Enron-era spam, 419
scams). Asked about a short, modern credential lure laid out as `from`/`subject`/`body`, it said
"Safe". v0.2 adds:

- **layout variation** on 75% of all training rows (`kenning.layouts`: renamed and nested fields,
  plain text with headers, unrelated metadata; synthetic sender and subject values are drawn
  independently of the label);
- **1,492 modern emails written by the local teacher** (Qwen2.5-7B-Instruct, Apache-2.0) from 20
  phishing and 20 legitimate scenarios, including hard legitimate ones (real security alerts,
  requested password resets, verification codes, paid-invoice confirmations). Labels come from the
  requested scenario, never from the teacher's judgement (`kenning.synthetic_email`,
  `systemone data --synthetic-email N`).

10,192 rows, 1 epoch, ~22 minutes. Held-out accuracy 0.74 zero-shot → 0.94 trained, ECE 0.007
calibrated.

| | v0.1 | v0.2 | TypeSafe Jev |
|---|---|---|---|
| Modern emails (20, hand-written): accuracy | 0.70 | **0.90** | 1.00 |
| Modern: phishing auto-closed as safe (p ≤ 0.1) | 4 | **1** | 0 |
| Layouts: accuracy across the 4 layouts | 0.94–0.96 | **0.96–0.98** | 0.96 |
| Phishing (50): accuracy / ECE | 0.960 / 0.039 | 0.960 / 0.031 | 0.960 / 0.109 |
| Phishing: category accuracy | 0.940 | 0.960 | 0.960 |
| Out of domain: spam / emotion / news | 0.967 / 0.750 / 0.933 | 0.967 / 0.767 / 0.900 | 0.967 / 0.583 / 0.933 |
| Latency p50, one at a time (modern / ood) | – | 23 ms / 28 ms | ~150 ms |

**Known failure:** one subtle credential lure ("your mailbox password expires, keep it by confirming
here", from a look-alike helpdesk domain) is still answered "no" with high confidence (0.008), and a
shared-document login lure gets 0.34. The teacher's phishing tends to be loud; calm, corporate-sounding
lures look like the legitimate security notices in the data. The fix is more subtle phishing and
matched legitimate counterparts from the teacher, checked on a new held-out modern set (the 20
hand-written emails must not become the training target). Until then, keep a person in the loop
for credential-related email.

### kenning-large-v0.3: the clean-licence recipe (not served)

v0.3 tests whether Kenning can be built from permissively licensed parts only: base
`answerdotai/ModernBERT-large` (Apache-2.0, pretrained only, no classification head) and training data
under Apache-2.0, CC-BY-3.0, OANC and CC0, plus teacher-written synthetic data:

| Source | Rows | Licence |
|---|---|---|
| `nyu-mll/multi_nli` (fiction genre excluded: it contains a CC-BY-SA work) | 20,000 | OANC |
| `clinc/clinc_oos` | 3,000 | CC-BY-3.0 |
| `fancyzhx/amazon_polarity` | 2,000 | Apache-2.0 |
| `google/civil_comments` | 2,000 | CC0-1.0 |
| synthetic modern emails, 40% calm-lure / legitimate-twin pairs (`--subtle-share`) | 1,985 | generated (Qwen2.5-7B-Instruct, Apache-2.0) |
| synthetic label-conditioned tasks, 10 tasks (`kenning.synthetic_tasks`, `--synthetic-tasks`) | 2,793 | generated (Qwen2.5-7B-Instruct, Apache-2.0) |

31,778 rows, 1 epoch at lr 3e-5, ~37 minutes. Held-out accuracy 0.435 (untrained head) → 0.922, ECE 0.0035.

The held-out split is the training distribution (mostly NLI and synthetic), and it hid a large drop on
real data. A new held-out set, `--suite modern2` (20 emails written before v0.3's subtle-phishing
scenarios), was added:

| | v0.2 | v0.3 | TypeSafe Jev |
|---|---|---|---|
| Phishing (50, real dataset) | 0.96 | 0.72 | 0.96 |
| Layouts | 0.96–0.98 | 0.66–0.74 | 0.96 |
| Out of domain: spam / emotion / news | 0.967 / 0.767 / 0.900 | 0.617 / 0.550 / 0.767 | 0.967 / 0.583 / 0.933 |
| Modern (20; v0.2's and v0.3's scenarios were designed after seeing it) | 0.90 | 1.00 | 1.00 |
| **Modern 2 (20, held out): accuracy** | 0.60 | 0.65 | **0.90** |
| **Modern 2: phishing auto-closed as safe (p ≤ 0.1)** | **5** | 1 | 0 |

**Conclusions.** The clean base gives up the zero-shot training that carried v0.1/v0.2 on real data,
and ~32k clean rows do not replace it. The calm-lure / legitimate-twin pairs help (1 lure auto-closed
instead of 5), but teacher-written data alone does not reach real-world accuracy. v0.2 stayed active
until v0.4 (below).
**No Kenning version is fit yet to auto-close subtle modern phishing; keep a person in the loop.**

Next candidates: a commercially-friendly zero-shot base (MoritzLaurer's `-c` models, e.g.
`deberta-v3-large-zeroshot-v2.0-c`, trained only on permissively licensed data) with the v0.3 data,
more and better subtle-phishing pairs (a stronger teacher), and a third held-out set.

## Cloudflare Clef: benchmark engine and teacher

[Clef](https://blog.cloudflare.com/clef-decision-models/) (Cloudflare, Apache-2.0 weights, Qwen
base plus a joint schema head) speaks the same wire format. The builder runs it locally behind the
optional compose profile `clef` (`systemone_builder.kenning.clef_serve`, which loads the model
repository's own `joint_schema_model.py`):

```bash
docker compose stop student                      # Clef-flash needs most of a 24 GB GPU
docker compose --profile clef up -d clef         # first start downloads ~19 GB
docker compose exec api systemone bench --suite modern2 --engines kenning,clef
```

Clef-flash fits on one RTX 3090 (17.8 GiB loaded, ~24 GiB in use while labelling in batches of 8;
batches of 16 overflowed into shared memory and ran 4x slower).

| Suite | kenning-large-v0.2 | Clef-flash (local, RTX 3090) | TypeSafe Jev |
|---|---|---|---|
| Modern 2 (held out): accuracy / phishing auto-closed | 0.60 / 5 | 0.90 / 0 | 0.90 / 0 |
| Modern | 0.90 | 1.00 | 1.00 |
| Phishing (50): accuracy | 0.96 | 0.96 (all automated decisions correct) | 0.96 |
| Layouts | 0.96–0.98 | 0.88–0.94 | 0.96 |
| Out of domain: spam / emotion / news | 0.967 / 0.767 / 0.900 | 0.900 / 0.600 / 0.950 | 0.967 / 0.583 / 0.933 |
| Latency p50, one at a time | 23–70 ms | 220–290 ms | ~150 ms |
| GPU memory | ~1 GB | ~18 GB | – |

### Distillation (`systemone label`)

Clef's licence allows training on its outputs (TypeSafe's terms forbid it for Jev). `systemone
label <data.jsonl>` sends rows to a teacher's `POST /v1/systemone/batch` and writes soft targets
(probability of yes, or a distribution over options), blended with the existing labels by
`--alpha` (0.5 = average). It also reports how often the teacher agrees with each source's labels:
over the full v0.3 dataset (31,778 rows, Clef-flash, ~3 hours on one RTX 3090): amazon_polarity
95.8%, clinc 95.0%, synthetic emails 84.6%, MNLI 82.6%, synthetic tasks 74.3% and civil_comments
73.2%. Low agreement flags noise in the teacher-written data and the borderline toxicity labels.

### kenning-large-v0.4: clean zero-shot base + Clef soft labels (active)

v0.4 combines the two fixes v0.3 pointed to:
- base `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`: MIT, zero-shot NLI training with no
  non-commercial data, though it includes FEVER-NLI (CC-BY-SA-3.0);
- the v0.3 rows, with their labels averaged with Clef-flash's soft labels
  (`systemone label --alpha 0.5`).

```bash
docker compose exec api systemone label /data/workspace/kenning/datasets/clean-v3-train.jsonl \
  --out /workspace/kenning/datasets/clean-v3-train-clef.jsonl --alpha 0.5 --batch 8
docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train \
  --data /workspace/kenning/datasets/clean-v3-train-clef.jsonl --out /workspace/kenning/models/kenning-large-v0.4 \
  --base MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c --lr 1e-5 --epochs 1 --batch-groups 8 --max-length 512
```

1 epoch, 4,951 steps, ~1 h 55 min, peak 18.1 GiB. Held-out accuracy 0.670 (zero-shot) → 0.927;
Brier 0.038; ECE 0.025 at T=1, 0.049 after temperature fitting. The first attempt ran out of memory at
step 2,461 under the 0.85 VRAM cap: 3.2 GiB was reserved but unused, which is fragmentation. The
trainer now splits each step into micro-batches of at most `--max-pairs` (state, answer) pairs. That
gives the same gradient; the rerun matched the first run's loss at step 1,001 exactly. The trainer
also uses expandable CUDA segments and skips (and logs) a step that still runs out of memory.

| | v0.2 | v0.3 | **v0.4** | Clef-flash | TypeSafe Jev |
|---|---|---|---|---|---|
| **Modern 2 (20, held out): accuracy** | 0.60 | 0.65 | **0.75** | 0.90 | 0.90 |
| **Modern 2: phishing auto-closed as safe** | 5 | 1 | **0** | 0 | 0 |
| Modern (20) | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phishing (50, real dataset) | 0.96 | 0.72 | 0.78 (category 0.82) | 0.96 | 0.96 |
| Layouts | 0.96–0.98 | 0.66–0.74 | 0.76–0.78 | 0.88–0.94 | 0.96 |
| Out of domain: spam / emotion / news | 0.967 / 0.767 / 0.900 | 0.617 / 0.550 / 0.767 | 0.917 / 0.583 / 0.867 | 0.900 / 0.600 / 0.950 | 0.967 / 0.583 / 0.933 |
| Latency p50, one at a time | 23–70 ms | – | 33–70 ms | 220–290 ms | ~150 ms |

**Reading the table.** The phishing and layout suites come from `zefang-liu/phishing-email-dataset`.
v0.2 was trained on other emails from that dataset, so those suites are in-distribution for v0.2. The
clean recipe leaves the dataset out because it is LGPL-3.0. On data no Kenning version was trained on
(out of domain, Modern 2), v0.4 recovers most of what v0.3 lost:
- **best Kenning yet on Modern 2**, and the first with **no phishing auto-closed** there;
- across every suite, its automated decisions (p ≥ 0.9 or ≤ 0.1) were 100% correct except one spam
  message auto-closed in the out-of-domain set;
- it is more conservative: it automates 20–44% of items and sends the rest to a person or System 2.

It is still well behind Clef and Jev on real phishing and on subtle lures.

v0.4 is active: it is the first model built from the clean recipe that is safe to gate on. v0.2
stays registered for comparison. Next: more real-looking calm-lure / legitimate-twin pairs, Clef
labels at alpha 1.0 for the synthetic sources (where the teacher disagrees most with its own
labels), and a permissively licensed real phishing corpus.

## Publishing to Hugging Face (`systemone publish`)

```bash
docker compose exec api systemone publish kenning-large-v0.4             # -> huggingface.co/systemonedev/kenning-large-v0.4
docker compose exec api systemone publish my-model --org my-hf-org --private
```

Uploads the weights, tokenizer and `kenning.json`, plus:
- a model card with Hugging Face metadata, the training data and its licences, the distillation teacher,
  the held-out results, and a **Benchmarks** table. The table lists the latest `systemone bench` run per
  suite in which Kenning served that exact model, so run the suites after activating it;
- `LICENSE` (Apache-2.0) and `NOTICE.md`.

It needs `HF_TOKEN` (in `.env`) with write access to the target repo. It refuses research-only models
(a base with non-commercial data in its lineage). A published model loads anywhere with
`Kenning.from_pretrained("systemonedev/kenning-large-v0.4")` or `S1_KENNING_MODEL=systemonedev/kenning-large-v0.4`.
Both download the repo, including `kenning.json`, so the calibration comes with it.

## Licensing of the weights

No pretrained language model has a lineage free of share-alike text (ModernBERT, DeBERTa and Qwen
were pretrained on web data that includes Wikipedia). The bar Kenning follows:

- no non-commercially licensed data anywhere in the lineage;
- Kenning's own fine-tuning data permissively licensed (Apache-2.0, MIT, CC-BY, CC0, OANC) or
  generated (Qwen2.5-7B-Instruct teacher, Clef soft labels, both Apache-2.0);
- every upstream licence listed in the export bundle's `NOTICE.md`, share-alike ones included
  (e.g. FEVER-NLI, CC-BY-SA-3.0, in the fine-tuning of the `-c` zero-shot bases).

**Kenning weights are released under Apache-2.0**, the same licence as the code. The export bundle
carries the licence text (`LICENSE`), a model card whose metadata says `license: apache-2.0`, and
`NOTICE.md`; keep `NOTICE.md` with the weights. The Apache-2.0 release applies to models with a clean
lineage (v0.3 and later). A model on a base whose own fine-tuning used non-commercially licensed
data (`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`: v0.1, v0.2) is exported with `license: other`
and a research-and-evaluation note instead (`registry.NC_BASES`).

Upstream share-alike terms are not settled law for model weights: CC's guidance says they *may*
apply to a model trained on the material. Listing those sources in `NOTICE.md` and keeping the
training mix free of non-commercial data is the project's good-faith position. It is not legal
advice.
