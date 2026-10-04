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
- **train** your own from permissively licensed data and distilled soft labels;
- **benchmark** it against other System One models, for calibration and safe automation as well
  as accuracy;
- **ship** it as a bundle with a model card and licence notice.

The flagship model is **Kenning**.

[![Open the Kenning demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/systemonedev/systemone-builder/blob/main/notebooks/kenning_demo.ipynb)
Try Kenning in your browser on Colab's free GPU: one click, no install, any text or JSON you like.

```text
state + questions ──▶ Kenning (cross-encoder, ~2 GB VRAM) ──▶ {"phish": {"noul": 0.82}}  ──▶ act / close / escalate
```

## Quickstart (one NVIDIA GPU)

Requirements: an NVIDIA driver, the NVIDIA Container Toolkit (or Docker Desktop with WSL2), and Docker Compose v2.20 or later.

```bash
cp .env.example .env                 # set S1_REDIS_PASSWORD (openssl rand -hex 24)
docker compose up -d                 # Redis, API, dashboard, Kenning
```

The first `up` downloads, once:

| Download | Size | From |
|---|---|---|
| `systemone-kenning` image (includes PyTorch and CUDA) | 4.3 GB | [GitHub Container Registry](https://github.com/orgs/systemonedev/packages) |
| `systemone-api` and `systemone-dashboard` images | 0.1 GB each | GitHub Container Registry |
| `kenning-large-v0.4` weights | 0.9 GB | [Hugging Face](https://huggingface.co/systemonedev/kenning-large-v0.4) |
| `redis` image | ~0.05 GB | Docker Hub |

`S1_IMAGE_TAG` chooses the image build: `main` by default, or a release version. To build from your
checkout instead: `docker compose build api dashboard kenning`.

- Dashboard: **http://localhost:3090**. *Try it* takes your own questions, *Models* trains and
  exports, *Connect* has client code.
- Kenning: `127.0.0.1:8093`. The API, with interactive docs at `/docs`, is on `127.0.0.1:8090`.

```bash
curl -s localhost:8093/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"from": "it-support@examp1e-corp.com", "subject": "Password expires today",
            "body": "Keep your password: sign in at http://examp1e-corp.com.verify-login.io"},
  "questions": {
    "phish":   {"type": "noul",   "instructions": "Is this email a phishing attempt?"},
    "urgency": {"type": "score",  "instructions": "How urgent does the sender make it sound?",
                "criteria": ["none", "low", "medium", "high"]},
    "route":   {"type": "choice", "instructions": "Which team should handle it?",
                "criteria": {"security": null, "it_helpdesk": null, "finance": null, "none": null}}}}'
```

From Python ([`clients/python`](clients/python), `pip install ./clients/python`):

```python
from systemone import Client, Noul
answer = Client("http://localhost:8093").system_one(
    state={"ticket": "I was charged twice this month."},
    questions={"billing": Noul("Is this about billing?")})
print(answer.nouls["billing"].noul)        # 0.93
```

Out of the box, Kenning serves the trained flagship,
[**systemonedev/kenning-large-v0.4**](https://huggingface.co/systemonedev/kenning-large-v0.4) (Apache-2.0),
downloaded from Hugging Face on first start together with its calibration. Activate a model you
trained on the *Models* page, or set `S1_KENNING_MODEL` in `.env` to serve another one. To use it
without a server: `Kenning.from_pretrained("systemonedev/kenning-large-v0.4")` (`pip install
"systemone-client[local]"`).

## Kenning results

Benchmarks run with `systemone bench`. The full method, data and caveats are in
[docs/kenning.md](docs/kenning.md).

| | Kenning v0.4 (local) | [Clef-flash](https://blog.cloudflare.com/clef-decision-models/) (local) | TypeSafe Jev (hosted) |
|---|---|---|---|
| Held-out modern emails: accuracy | 0.75 | 0.90 | 0.90 |
| Held-out modern emails: phishing auto-closed as safe | **0** | 0 | 0 |
| Phishing dataset (50) | 0.78 | 0.96 | 0.96 |
| Out of domain: spam / emotion / news | 0.917 / 0.583 / 0.867 | 0.900 / 0.600 / 0.950 | 0.967 / 0.583 / 0.933 |
| Latency p50 (one RTX 3090) | 33–70 ms | 220–290 ms | ~150 ms (network) |
| GPU memory | ~2 GB | ~18 GB | – |

Kenning is about 20 times smaller than Clef and several times faster, and it is calibrated
conservatively: when it automates a decision it was right in every suite except one spam message.
It is still behind on subtle real-world phishing, so keep a person in the loop for those.

## Train your own

```bash
docker compose exec api systemone data --task multitask --per-source 1200      # build training rows
docker compose --profile clef up -d clef                                        # optional teacher
docker compose exec api systemone label /workspace/kenning/datasets/<rows>.jsonl --alpha 0.5 --batch 8
docker compose run --rm --no-deps kenning python -m systemone_builder.kenning.train \
  --data /workspace/kenning/datasets/<rows>.jsonl --out /workspace/kenning/models/my-model
docker compose exec api systemone bench --suite modern2 --engines kenning,clef
docker compose exec api systemone publish my-model --org <your-hf-org>              # share it (Apache-2.0 models)
```

Training fits in 24 GB. A run on ~30k rows takes 40 minutes to 2 hours on an RTX 3090.
[docs/kenning.md](docs/kenning.md) covers:
- data sources and their licences;
- synthetic data and distillation from Clef (Apache-2.0);
- calibration;
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
