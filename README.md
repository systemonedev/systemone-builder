# systemone-builder

**Open-source autonomous System-1 reflex model distillation.**

`systemone-builder` distills slow, deliberative *System-2* reasoning into sub-100ms,
deterministic *System-1* reflex models. Bring your own hardware and models, describe the
specialization you want in plain language, hit play: the framework generates the
workflow, data schemas and synthetic-data parameters, runs a local
distill → train → hot-reload loop, benchmarks the result on real held-out data and
serves it behind a confidence-routed fast/slow cascade. No closed-source APIs.

Primary example domains:

* **Real-time computer-use**: GUI/DOM and vision automation with pixel-coordinate fallbacks.
* **Cybersecurity reflex triage**: SecOps verdicts and containment actions from IDS/log events.

```text
+---------------------------------------------------------------------------------------------------+
|                                LINUX HOST (NVML Orchestrator + 120GB RAM)                         |
|  *120GB RAM Role: In-Memory Datastore (Redis) for Replay Buffering & vLLM CPU KV-Cache Offload*   |
+-----------------------------------------+---------------------------------------------------------+
|                GPU 0                    |                         GPU 1                           |
|          NVIDIA RTX 3090 (24GB)         |                 NVIDIA RTX 3090 (24GB)                  |
+-----------------------------------------+---------------------------------------------------------+
| [Role: Sub-100ms Student Engine]        | [Role: Intermediate Synchronous Triage]                 |
| *Strict Lifecycle Managed to avoid OOM* | CONTAINER C: vLLM Server (BYOM Dense / MoE)             |
| CONTAINER A: Unsloth QLoRA Trainer      | - Fast-Slow escalation layer for moderate complexity    |
| CONTAINER B: vLLM Student Runner        | - Handles live fallback generation                      |
| - Prefix Caching Enabled                |                                                         |
+-----------------------------------------+---------------------------------------------------------+
                                          |
                                     (LAN / REST)
                                          |
+---------------------------------------------------------------------------------------------------+
|                                MAC M4 MAX ENDPOINT (64GB Unified Memory)                          |
+---------------------------------------------------------------------------------------------------+
| [Role: Asynchronous Oracle & DPO Evaluator]                                                       |
| OLLAMA SERVER: Hosting `qwen3.8:27b` (or user-defined 70B+ model)                                 |
| - Offline Synthetic Data Factory (batch CoT generation to free up Linux GPUs).                    |
| - Deep Out-of-Band Threat Analysis & LLM-as-a-Judge validation.                                   |
+---------------------------------------------------------------------------------------------------+
```

## Zero-to-One Quickstart

On the Mac (oracle), once:

```bash
./mac/setup_oracle.sh            # exposes Ollama on the LAN and pulls qwen3.8:27b
```

On the Linux host (NVIDIA driver + NVIDIA Container Toolkit + Docker Compose v2):

```bash
cp .env.example .env             # set S1_ORACLE_URL to the Mac, optionally S1_API_KEY / HF_TOKEN
docker compose up
```

`docker compose up` builds and starts Redis, the orchestrator API, the vLLM student
(GPU 0, `Qwen/Qwen2.5-1.5B-Instruct`), the vLLM triage server (GPU 1) and the dashboard.
It then runs the one-shot **quickstart** service, which:

1. loads a dummy web-automation dataset (procedurally generated, or pulled from
   `S1_QUICKSTART_DATASET_URL`) as SFT data plus an unseen held-out set,
2. runs the first strict **pause vLLM → flush VRAM → Unsloth QLoRA → hot-reload** cycle on GPU 0,
3. benchmarks the fine-tuned student and prints accuracy, hallucination ratio and latency.

Open **http://&lt;linux-host&gt;:3090** for the dashboard. Interactive API docs are at `:8090/docs`.
`systemone doctor` (inside the `api` container or a local install) checks GPUs, Docker,
Redis and all three model endpoints.

## What's inside

| Module | What it does | Code |
|---|---|---|
| **A** State extraction & RAM buffering | Fuzzy scrubbing of DOM/logs to preserve vLLM prefix-cache hits; Redis replay buffer of the last 10,000 actions + state deltas | `extraction/`, `datastore/` |
| **B** Synthetic distillation & validation | Mac-side batch CoT generation from replay + seed synthesis, LLM-as-a-Judge, screenshot → element vision parsing | `factory/` |
| **C** Fine-tuning engine & VRAM orchestrator | Strict GPU 0 lifecycle (drain → stop vLLM → NVML flush check → Unsloth QLoRA on all linear layers → flush check → hot-reload), rollback | `training/`, `orchestrator/`, `docker/trainer`, `docker/student` |
| **D** Real-time playground & fast-slow routing | Calibrated confidence (self-report × decision-token probability), per-domain thresholds, Student → Triage → Oracle cascade | `routing/` |
| **E** Contextual DPO feedback loop | State-delta capture of failed actions, delta injection to the teacher, corrections studio, DPO pairs | `dpo/` |
| **F** Unified web dashboard | Prompt-to-Workflow engine, live telemetry (TTFT, prefix-cache hit rate, loss), routing map, replay scrubber, DPO studio | `workflow/`, `telemetry/`, `frontend/` |
| **G** Evaluation & accuracy suite | Held-out benchmarking: accuracy, success rate, hallucination ratio, latency/TTFT distributions, calibration, readiness gates | `evaluation/` |
| **H** Open-source DX | One-command quickstart, BYOM adapters/plugins, starter templates, CLI | `quickstart/`, `adapters/`, `templates/`, `cli.py` |

Backend code lives under `backend/systemone/`.

## Using it

```bash
# route an observation (the reflex path)
curl -s localhost:8090/api/v1/act/secops -H 'content-type: application/json' -d '{
  "observation": {"kind": "json", "data": {"source": "suricata_eve", "src_ip": "192.168.1.150",
                  "payload_snippet": "GET /../../../../etc/passwd HTTP/1.1\r\n\r\n"}}}'

# report what happened (feeds the state-delta DPO loop)
curl -s localhost:8090/api/v1/feedback -H 'content-type: application/json' \
  -d '{"seq": 42, "outcome": "failure", "post_observation": {...}}'

# create a new specialization from a prompt
curl -s localhost:8090/api/v1/workflows/generate -H 'content-type: application/json' \
  -d '{"prompt": "Block SSH brute-forcers on our bastions, never block 10.20.0.0/16"}'
```

Complete client loops are in [`examples/`](examples): a Playwright computer-use agent and a
Suricata EVE reflex tailer that drives nftables.

## Documentation

* **[Build your first System-1 model](docs/first-model.md)**: step-by-step walkthrough
* [Architecture](docs/architecture.md): data flow, routing, lifecycle state machine
* [Hardware & deployment](docs/hardware.md): GPU/RAM budget, Mac oracle, networking
* [BYOM](docs/byom.md): plugging in your own student/triage/oracle models and adapters
* [Domains & templates](docs/domains.md): data contracts, starter templates, Prompt-to-Workflow
* [Training & evaluation](docs/training.md): factory, lifecycle, DPO, benchmarking
* [API reference](docs/api.md)

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "backend[dev]"
redis-server --daemonize yes
S1_TEST_REDIS_URL=redis://localhost:6379/15 pytest backend/tests
cd frontend && NEXT_PUBLIC_S1_API_PORT=8000 npm run dev   # dashboard on :3000 against `systemone serve` on :8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under Apache-2.0.
