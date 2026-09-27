# Hardware & deployment

## Reference topology

| Machine | Role | Services |
|---|---|---|
| Linux, GPU 0 (RTX 3090 24GB) | Sub-100ms student | Container A `systemone-trainer` (one-shot), Container B `systemone-student` (vLLM) |
| Linux, GPU 1 (RTX 3090 24GB) | Synchronous triage | Container C `systemone-triage` (vLLM) |
| Linux, 120GB RAM | Datastore + KV offload | Redis (`maxmemory 64gb`), vLLM CPU KV cache |
| Mac M4 Max (64GB) | Asynchronous oracle | Ollama `qwen3.8:27b` (factory, judge, vision, escalations) |

## RAM budget (Linux host)

| Consumer | Budget | Setting |
|---|---|---|
| Redis (replay, queues, datasets index) | 64 GB | `docker/redis/redis.conf` `maxmemory` |
| Student vLLM CPU KV offload | 32 GB | `S1_KV_OFFLOAD_GB` |
| Triage vLLM CPU KV offload | 16 GB | `S1_TRIAGE_KV_OFFLOAD_GB` |
| OS, Docker, API, trainer dataloaders | ~8 GB | |

Both vLLM containers start through `docker/student/serve_student.py`. It checks every
optional tuning flag against the installed `vllm serve --help` output and skips the ones
that version doesn't support, logging a warning. Upgrading or downgrading the
`vllm/vllm-openai` image therefore never breaks startup because of a renamed or removed flag.

CPU KV offload uses vLLM's native offloading backend (`--kv-offloading-backend native
--kv-offloading-size`). If your build lacks it, the flags are skipped; you can pass another
connector through `S1_VLLM_EXTRA_ARGS` (triage: `S1_TRIAGE_VLLM_EXTRA_ARGS`), e.g. LMCache:
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.
Older vLLM releases that still have `--swap-space` can use it via `S1_SWAP_SPACE_GB`.

## WSL2 / Docker Desktop hosts

vLLM disables pinned host memory under WSL2 by default, and its V2 model runner then fails
with `RuntimeError: UVA is not available`. The launcher detects a WSL2 kernel and sets
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`. This needs WSL2 kernel ≥ 4.19.121; run `wsl --update` if
yours is older, or set the variable to `0` in `.env` to opt out.

**Storage.** Keep the workspace (model cache, runs, datasets) in the default Docker named
volume `systemone-workspace`. A bind mount of a Windows folder (`/mnt/c/...`) goes through
the 9P bridge at a few MB/s; loading a 3 GB model then takes more than 15 minutes. The
launcher logs a warning when it detects this.

**CPU KV offload.** vLLM pre-faults its offload region in `/dev/shm`. On WSL2 the kernel
can refuse to back it (`OSError: [Errno 14] Bad address`). Before starting vLLM, the launcher
performs the same `madvise(MADV_POPULATE_WRITE)` call vLLM uses, halving the size until it
succeeds, or disabling offload if nothing fits. You can set the sizes explicitly with
`S1_KV_OFFLOAD_GB` / `S1_TRIAGE_KV_OFFLOAD_GB`; `S1_KV_OFFLOAD_PROBE=0` skips the probe. Also raise the VM's memory in `%UserProfile%\.wslconfig` (`memory=`) so that Redis's
64 GB and the offload buffers fit.

## Linux prerequisites

* NVIDIA driver ≥ 550, NVIDIA Container Toolkit, Docker Engine + Compose v2
* About 100 GB free disk for Hugging Face caches and training runs (`./workspace`)
* The API container mounts `/var/run/docker.sock` (to stop/start containers A/B) and runs with
  `pid: host` (so NVML can attribute VRAM to processes). Anyone with access to the API can
  therefore control Docker on the host. Set `S1_API_KEY`, and don't expose port 8000 beyond
  your LAN.

## Mac oracle

Run `mac/setup_oracle.sh`. It sets `OLLAMA_HOST=0.0.0.0:11434`, keep-alive and parallelism,
then pulls the model. Point `S1_ORACLE_URL` at the Mac. For screenshot-only GUIs, also pull a
vision model and set `S1_ORACLE_VISION_MODEL`.

To verify connectivity from the Linux host: `curl http://<mac-ip>:11434/api/tags`, then
`docker compose exec api systemone doctor`.

## Ports

| Port | Service |
|---|---|
| 3000 | Dashboard |
| 8000 | LAN REST + WebSocket API |
| 8001 | Student vLLM (debug) |
| 8002 | Triage vLLM (debug) |
| 6379 | Redis |
| 11434 | Ollama on the Mac |

## Single-GPU or different hardware

Everything is configurable through `S1_*` variables. On one GPU, point `S1_TRIAGE_URL` at
any OpenAI-compatible endpoint (or at the student itself) and remove the `triage` service.
With more VRAM, raise `S1_GPU_MEMORY_UTILIZATION`, `S1_MAX_MODEL_LEN` and the triage model size.
