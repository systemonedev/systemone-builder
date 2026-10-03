# Kenning: the local System One model

Kenning is a **System One** decision model: program **state** plus typed **questions** in,
**answers** with calibrated probabilities out, in one forward pass and without generating text.
The category and the wire format were introduced by TypeSafe AI with Jev, its hosted model;
Kenning is an open, locally run model that speaks a compatible format, and Jev is one of the
systems it is benchmarked against. Kenning is not affiliated with TypeSafe AI and is never trained
on TypeSafe outputs.

## How it works

The model is a non-generative cross-encoder (default base:
[`MoritzLaurer/ModernBERT-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0),
395M parameters, Apache-2.0). Each candidate answer becomes a hypothesis scored against the state:

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

The compose service `kenning` (GPU 0, about 1–2 GB of VRAM) serves `POST /v1/systemone` on
`127.0.0.1:8093`, and the API exposes it as `POST /api/v1/systemone`
(`S1_SYSTEM_ONE_BACKEND=kenning`, the default; `logprob` switches to an LLM's label-token readout).

In the dashboard:

- **Models** lists every trained model in the workspace volume (`kenning/models/`) with its
  held-out results and training data. **Activate** hot-swaps the served model (no restart) and
  persists the choice in `kenning/active.json`; without it, `S1_KENNING_MODEL` (a model directory or
  a Hugging Face cross-encoder id, default the zero-shot base) is served. **Export** builds a zip
  with the weights, tokenizer, `kenning.json`, a model card, `NOTICE.md` with the licences of the
  base model and every training source, and `SHA256SUMS`.
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
# free GPU 0, train, restore
docker compose stop student kenning
docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train \
  --data /workspace/kenning/datasets/multitask-train.jsonl --out /workspace/kenning/models/kenning-large-v0.1 --max-length 512
docker compose up -d --no-deps student
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

## Benchmarks

`systemone bench --suite phishing` uses 50 unseen, balanced emails from the phishing dataset;
`--suite ood` uses tasks never trained on (`ucirvine/sms_spam`, `dair-ai/emotion`,
`fancyzhx/ag_news`, `-n` items per task); `--suite layouts` asks about the same 50 phishing emails in
four layouts (the training layout, `from`/`subject`/`body` JSON, plain text with headers, nested with
metadata) to measure layout sensitivity; `--suite modern` is 20 hand-written short, modern emails
(evaluation only, never trained on). Engines: `kenning` (this model), `jev` (opt-in TypeSafe Jev with
your own key, `TYPESAFE_API_KEY`), `local` (an LLM's label-token readout), `llm` (an LLM writing JSON).

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
docker compose exec api systemone label /workspace/kenning/datasets/clean-v3-train.jsonl \
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

## Licensing of the weights

No pretrained language model has a lineage free of share-alike text (ModernBERT, DeBERTa and Qwen
were pretrained on web data that includes Wikipedia). The bar Kenning follows:

- no non-commercially licensed data anywhere in the lineage;
- Kenning's own fine-tuning data permissively licensed (Apache-2.0, MIT, CC-BY, CC0, OANC) or
  generated (Qwen2.5-7B-Instruct teacher, Clef soft labels, both Apache-2.0);
- every upstream licence listed in the export bundle's `NOTICE.md`, share-alike ones included
  (e.g. FEVER-NLI, CC-BY-SA-3.0, in the fine-tuning of the `-c` zero-shot bases).
