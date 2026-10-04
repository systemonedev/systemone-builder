# SystemOne Builder

**Build, benchmark and serve System One decision models on your own hardware.**

A System One model answers typed questions about program state (yes/no, pick one, rate on a scale)
with calibrated probabilities, in one forward pass, in milliseconds. It doesn't write text. It is the
fast, deterministic reflex in front of slower reasoning:
- auto-act when the model is sure;
- auto-close when it is sure the answer is no;
- send everything else to a person or to an LLM.

SystemOne Builder is the open toolkit for these models. With it you can:
- **serve** one locally behind the `POST /v1/systemone` wire format;
- **train** your own: a general multi-task model, one for your own problem (agent guardrails, claims
  intake, moderation, phishing...), or both, from permissively licensed data, teacher-written cases,
  your own labelled data and distilled soft labels;
- **benchmark** it against other System One models, for calibration and safe automation as well
  as accuracy;
- **ship** it as a bundle with a model card and licence notice.

The flagship model is **Kenning**.

```text
state + questions ──▶ Kenning (cross-encoder, ~2 GB VRAM) ──▶ {"outage": {"noul": 0.76}}  ──▶ act / close / escalate
```

## Quickstart (one NVIDIA GPU)

Requirements: an NVIDIA driver, the NVIDIA Container Toolkit (or Docker Desktop with WSL2), and Docker Compose v2.20 or later.

```bash
cp .env.example .env                 # set S1_REDIS_PASSWORD (openssl rand -hex 24)
docker compose up -d                 # Redis, API, dashboard, Kenning
```

The first `up` downloads the prebuilt images from
[GitHub Container Registry](https://github.com/orgs/systemonedev/packages)
(`ghcr.io/systemonedev/systemone-{api,dashboard,kenning}`; Kenning's is the large one, it includes
PyTorch and CUDA). `S1_IMAGE_TAG` chooses the build: `main` by default, or a release version. To build
from your checkout instead: `docker compose build api dashboard kenning`.

- Dashboard: **http://localhost:3090**. *Try it* takes your own questions, *Models* trains and
  exports, *Connect* has client code.
- Kenning: `127.0.0.1:8093`. The API, with interactive docs at `/docs`, is on `127.0.0.1:8090`.

```bash
curl -s localhost:8093/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": "Our checkout page has returned 500 errors for all customers since 9:05.",
            "customer": {"plan": "enterprise"}},
  "questions": {
    "outage":   {"type": "noul",   "instructions": "Is the customer reporting an outage?"},
    "priority": {"type": "score",  "instructions": "How urgent is this ticket?",
                 "criteria": ["low", "medium", "high", "critical"]},
    "team":     {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": {"billing": null, "engineering": null, "account": null, "sales": null}}}}'
```

Every answer is one of your options with a probability: here `outage` 0.76, `priority` about 1.8
on the 0–3 scale, `team` engineering.

From Python ([`clients/python`](clients/python), `pip install ./clients/python`):

```python
from systemone import Client, Noul
answer = Client("http://localhost:8093").system_one(
    state={"ticket": "I was charged twice this month."},
    questions={"billing": Noul("Is this about billing?")})
print(answer.nouls["billing"].noul)        # 0.93
```

Out of the box, Kenning serves a zero-shot base model with no non-commercial data in its lineage,
[`deberta-v3-large-zeroshot-v2.0-c`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c)
(MIT). The trained flagship is published under Apache-2.0:
[**systemonedev/kenning-large-v0.4**](https://huggingface.co/systemonedev/kenning-large-v0.4). Serve it with
`S1_KENNING_MODEL=systemonedev/kenning-large-v0.4` in `.env`, or load it in-process with
`Kenning.from_pretrained("systemonedev/kenning-large-v0.4")`.

## Kenning results

Benchmarks run with `systemone bench`. The full method, data and caveats are in
[docs/kenning.md](docs/kenning.md).

**The headline: general decisions** (`--suite general`, 1,328 held-out items, 30 questions in 7
families, macro accuracy). Clef is the development target. Jev is compared again once Kenning matches
Clef.

| Family | Kenning v0.4 | [Clef-flash](https://blog.cloudflare.com/clef-decision-models/) |
|---|---|---|
| **All 30 questions (macro)** | **0.625** | **0.813** |
| Text: evidence, sentiment, toxicity, injection, intent, topic (14) | 0.754 | 0.841 |
| Conversations (1) | 0.958 | 1.000 |
| Answer quality (2) | 0.377 | 0.447 |
| Agent decisions: tool calls, task completion (3) | 0.553 | 0.793 |
| Records: rules over JSON with numbers and dates (7) | 0.535 | 0.857 |
| Tables (1) | 0.480 | 0.860 |
| Logs (2) | 0.290 | 0.740 |
| Latency p50 / GPU memory (one RTX 3090) | 34 ms / ~2 GB | 149 ms / ~18 GB |

Kenning is close to Clef on text, about 20 times smaller and 4 times faster, and behind on
structured state: records, tables, agent steps and long logs (it reads 512 tokens per option). That
gap is what the next version is for. Until then, train it on your own problem (below) and measure it
on your data.

<details><summary>Email suites (regression checks)</summary>

| | Kenning v0.4 | Clef-flash | TypeSafe Jev (hosted) |
|---|---|---|---|
| Held-out modern emails: accuracy | 0.75 | 0.90 | 0.90 |
| Held-out modern emails: phishing auto-closed as safe | **0** | 0 | 0 |
| Phishing dataset (50) | 0.78 | 0.96 | 0.96 |
| Out of domain: spam / emotion / news | 0.917 / 0.583 / 0.867 | 0.900 / 0.600 / 0.950 | 0.967 / 0.583 / 0.933 |

When Kenning automates a decision it was right in every suite except one spam message. It is behind on
subtle real-world phishing, so keep a person in the loop for those.
</details>

## Train your own

The **Train** page does all of this from the dashboard, and **Verify** benchmarks the result. From the
command line:

**For your own problem.** Describe it once and a teacher LLM writes labelled cases; or bring your own
labelled data; or both. Out of the box, v0.4 is unsure on agent steps (asked whether an agent should
check with the user before switching them to a $499 plan, it says 0.42), which is the kind of gap a
few thousand cases close.

```bash
# built-in specs: computer_use, insurance_claim, chat_moderation (or write your own)
docker compose exec api systemone problems
# 3,000 teacher-written cases (each checked blind), your own labelled data (validated line by
# line), and general data so the model stays good at everything else
docker compose exec api systemone data --out /data/workspace/kenning/datasets/agent-v1.jsonl \
  --problem computer_use=3000 --phishing-rows 0 --per-source 1200 \
  --import /data/workspace/my-labels.jsonl --import-licence "CC-BY-4.0"
```

- A **problem spec** is a small JSON file: the state's fields and every question with all its answers
  ([format](docs/kenning.md#train-for-your-own-problem)). Save your own on the Train page or pass
  `--problem path/to/spec.json`.
- The teacher is the pipeline's triage model, or any OpenAI-compatible server
  (`--teacher-url http://host:11434/v1 --teacher-model <name>`). A stronger teacher keeps more cases.
- 10% of each problem and import is **held out** and appears on Verify as its own suite, so you
  measure the model on the problem you trained it for.
- `systemone calibrate <model> --data <labelled.jsonl> --write` refits the temperatures on your data,
  so its probabilities are honest there (accuracy unchanged).

**Then distil, train, measure and share:**

```bash
docker compose --profile clef up -d clef                                        # optional teacher
docker compose exec api systemone label /data/workspace/kenning/datasets/<rows>.jsonl --alpha 0.5 --batch 8
docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train \
  --data /workspace/kenning/datasets/<rows>.jsonl --out /workspace/kenning/models/my-model
docker compose exec api systemone bench --suite general --engines kenning,clef
docker compose exec api systemone publish my-model --org <your-hf-org>              # share it (Apache-2.0 models)
```

Training fits in 24 GB. A run on ~30k rows takes 40 minutes to 2 hours on an RTX 3090.
[docs/kenning.md](docs/kenning.md) covers:
- data sources and their licences;
- problem specs, imports, hold-outs and calibration;
- synthetic data and distillation from Clef (Apache-2.0);
- every benchmark suite;
- how to compare with TypeSafe Jev using your own key (opt-in; Jev outputs are never used for
  training).

## Optional: the generative pipeline

The project started as a generative distillation pipeline, and it is still included, for agents
that need free-form actions. It needs 2 GPUs: a vLLM student on GPU 0 and a vLLM triage model on
GPU 1, plus an optional Ollama oracle on another machine. The pipeline:
- is a fast/slow cascade with LoRA hot-reloads;
- feeds failed actions back as DPO training data;
- includes a prompt-to-workflow generator.

```bash
# in .env: S1_PIPELINE=1 and COMPOSE_PROFILES=pipeline
docker compose up -d                 # adds the student, triage, trainer and the one-shot quickstart
```

The dashboard then shows the *Legacy pipeline* pages. See [docs/architecture.md](docs/architecture.md),
[docs/first-model.md](docs/first-model.md) and [docs/hardware.md](docs/hardware.md).

## Documentation

* **[Kenning](docs/kenning.md)**: how it works, training, distillation, benchmarks, licensing
* [API reference](docs/api.md)
* [Hardware & deployment](docs/hardware.md): GPU/RAM budget, WSL2 notes, networking
* Generative pipeline: [architecture](docs/architecture.md), [first model](docs/first-model.md),
  [BYOM](docs/byom.md), [domains](docs/domains.md), [training & evaluation](docs/training.md)

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "backend[dev]" -e "clients/python[dev]"
redis-server --daemonize yes
S1_TEST_REDIS_URL=redis://localhost:6379/15 pytest backend/tests clients/python/tests
cd frontend && npx next dev -p 3001 &    # Next.js dev server on a private port
# dashboard on :3000 (via the gateway) against `systemone serve` on :8000
S1_NEXT_URL=http://localhost:3001 S1_API_INTERNAL_URL=http://localhost:8000 PORT=3000 node gateway.mjs
```

Images: `docker compose build api dashboard kenning` builds them locally under the same names the
stack runs. CI publishes them from `main` (`.github/workflows/images.yml`) after
`.github/scan-image.sh` checks each one for credential files, key patterns and secret environment
variables.

See [CONTRIBUTING.md](CONTRIBUTING.md) (data rules, DCO sign-off), [SECURITY.md](SECURITY.md)
and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Licence and credits

Code and Kenning weights: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Each exported model
ships with its licence and a `NOTICE.md` listing the base model and every training source with
their licences. Models built on a base with non-commercial training data are marked as not
Apache-2.0. Created by [Jesus Rodriguez](https://github.com/jesusdrodriguez); maintained by
[systemonedev](https://github.com/systemonedev) ([systemone.dev](https://systemone.dev)).

The System One category and wire format were introduced by TypeSafe AI with Jev. SystemOne Builder
implements a compatible format, is not affiliated with or endorsed by TypeSafe AI, and does not
train on TypeSafe outputs. Clef is Cloudflare's (Apache-2.0); this project is not affiliated with
Cloudflare.
