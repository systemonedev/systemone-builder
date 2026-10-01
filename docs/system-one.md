# The local System One model

A System One model makes fast, typed, calibrated decisions instead of generating text: program
**state** plus typed **questions** in, **answers** with probabilities out. The idea and the
contract come from TypeSafe's [Jev](https://docs.typesafe.ai); this is the open, locally run
counterpart, and Jev is what it is benchmarked against.

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
training, which is what calibrates the probabilities. `confidence` follows Jev's formula.

**Deterministic:** a request is its own batch, requests run one at a time with deterministic
kernels, and the softmax runs in float64. The same request gives bit-identical answers.

## Serving

The compose service `s1` (GPU 0, about 1–2 GB of VRAM) serves Jev's wire format at
`POST /v1/systemone` on `127.0.0.1:8093`, and the API exposes it as `POST /api/v1/systemone`
(`S1_SYSTEM_ONE_BACKEND=model`, the default; `logprob` switches to the LLM label-token readout).
`S1_SYSTEM_ONE_MODEL` selects the weights: a trained directory in the workspace volume, or a
Hugging Face cross-encoder id for zero-shot use (the default).

## Training

```bash
# training rows for the built-in phishing task (benchmark emails are excluded)
docker compose exec api systemone s1-data -n 4000
# free GPU 0, train, restore
docker compose stop student
docker compose run --rm --no-deps s1 python -m systemone.s1model.train \
  --data /workspace/s1/datasets/phishing-train.jsonl --out /workspace/s1/models/s1-phishing-v1 --max-length 512
docker compose up -d --no-deps student
# serve it: S1_SYSTEM_ONE_MODEL=/workspace/s1/models/s1-phishing-v1 in .env, then
docker compose up -d --no-deps s1
```

Training data is JSONL, one decision context per line:
`{"state": ..., "questions": {qid: {type, instructions, criteria}}, "targets": {qid: target}}`.
Targets are hard labels (noul 0/1, choice option, score level) or soft distributions from a teacher
(noul P(yes), `{option: p}`, `{"0": p, ...}`), so decisions can be distilled from a System 2 model
with their uncertainty. Training uses gradient checkpointing and a hard VRAM cap
(`--max-vram-fraction`, default 0.85): exceeding it fails with an out-of-memory error instead of
spilling into shared system memory, which on WSL2 is slow and destabilising. The held-out metrics
before training (zero-shot), after, and after calibration are written to `s1_config.json` next to
the weights.

## Results so far (`s1-phishing-v1`, 4,000 training emails, one epoch, 10 minutes on an RTX 3090)

Held-out split of the training pool (800 questions): accuracy 0.67 zero-shot → 0.98 trained;
ECE 0.197 → 0.005 calibrated.

Benchmark (`systemone s1-bench`, 50 unseen emails, balanced):

| | s1-phishing-v1 | TypeSafe Jev | Qwen 7B label readout |
|---|---|---|---|
| Accuracy (malicious?) | 1.000 | 0.960 | 0.800 |
| Brier | 0.001 | 0.042 | 0.191 |
| ECE | 0.013 | 0.116 | 0.199 |
| Automated at p ≥ 0.9 / ≤ 0.1 | 98% | 58% | 96% |
| Phishing auto-closed as safe | 0 | 0 | 9 |
| Latency p50 / p95, one at a time | 33 / 51 ms | 164 / 215 ms | – |
| Identical answers on repeat | 10/10 | 1/10 | 8/10 |

**Known limitation:** trained on one task, it is in-distribution on that benchmark while Jev is
zero-shot, and it drifts on other tasks: on unrelated support tickets its routing (`choice`) and
severity (`score`) answers match Jev, but its yes/no answers lean towards "yes" (a polite ticket
rated urgent at 0.98, Jev 0.14). Broader, multi-task training data is the next step before it can
claim parity with Jev in general.
