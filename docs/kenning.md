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
`fancyzhx/ag_news`, `-n` items per task). Engines: `kenning` (this model), `jev` (opt-in TypeSafe Jev with your own key,
`TYPESAFE_API_KEY`), `local` (an LLM's label-token readout), `llm` (an LLM writing JSON).

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
