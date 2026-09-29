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
| Redis (replay, queues, datasets index) | 32 GB | `docker/redis/redis.conf` `maxmemory` (raise on a dedicated Linux host) |
| Student vLLM CPU KV offload | 32 GB (0 on WSL2) | `S1_KV_OFFLOAD_GB` |
| Triage vLLM CPU KV offload | 16 GB (0 on WSL2) | `S1_TRIAGE_KV_OFFLOAD_GB` |
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

## Docker Desktop / WSL2 hosts: bring-up checklist

Reference: Windows 11, 192 GB RAM, `.wslconfig` `memory=160GB`, 2x RTX 3090.

1. **Secrets in `.env`.** `S1_REDIS_PASSWORD` is required, and compose refuses to start
   without it (`openssl rand -hex 24`). Set `S1_API_KEY` (`openssl rand -hex 32`) before
   setting `S1_BIND_ADDR=0.0.0.0`; the API refuses a LAN bind without a real key.
2. **Ports.** Everything is on 127.0.0.1 by default. API 8090 and dashboard 3090 follow
   `S1_BIND_ADDR`. Redis 6379 and the raw vLLM servers (8091/8092, no auth) are always
   host-local. The dashboard is built to call `S1_API_PORT`, so rebuild it after changing that.
3. **GPU isolation.** Docker Desktop ignores `device_ids` / `NVIDIA_VISIBLE_DEVICES`, so every
   container sees both GPUs. The vLLM launcher and the trainer detect this and set
   `CUDA_VISIBLE_DEVICES` to `S1_GPU_INDEX` (student/trainer 0, triage 1; PCI bus order). On
   native Linux the same compose file keeps working, because only one GPU is visible there. A
   static `CUDA_VISIBLE_DEVICES=1` would hide triage's only GPU on native Linux.
4. **Staged start.** Triage waits for a healthy student, and the API waits for both, so two
   engines never load at once. Restart policies are `"no"`, so a VM crash can't turn into a
   restart loop.
5. **Memory.** `mem_limit` applies to redis (40g), student (48g), triage (32g), api (4g),
   dashboard (1g) and the trainer (48g, private 8g `/dev/shm`, no `ipc: host`), each
   overridable with `S1_*_MEM_LIMIT`. Redis `maxmemory` is 32gb. CPU KV offload is **0 by
   default on WSL2**. Raise `S1_KV_OFFLOAD_GB` / `S1_TRIAGE_KV_OFFLOAD_GB` gradually while
   watching `docker stats` and `vmmemWSL` in Task Manager. The offload regions live in the
   shared `/dev/shm` (`ipc: host`) and may not count against a container's `mem_limit`.
6. **Bring-up order:**
   `docker compose up -d redis student` → wait for healthy → `docker compose up -d triage`
   → healthy → `docker compose up -d api dashboard`.

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
`S1_KV_OFFLOAD_GB` / `S1_TRIAGE_KV_OFFLOAD_GB`; `S1_KV_OFFLOAD_PROBE=0` skips the probe.

Both vLLM containers use `ipc: host`, so offload regions live in the host's `/dev/shm`,
which WSL2 sizes at half the VM's memory by default. The launcher removes offload files left
by crashed engines (`S1_CLEAN_SHM=0` disables this). It also caps each instance at 45% of the
free `/dev/shm` (`S1_KV_OFFLOAD_MAX_SHM_FRACTION`), so student and triage both fit. Also raise the VM's memory in `%UserProfile%\.wslconfig` (`memory=`) so that Redis's
64 GB and the offload buffers fit.

## Linux prerequisites

* NVIDIA driver ≥ 550, NVIDIA Container Toolkit, Docker Engine + Compose v2
* About 100 GB free disk for Hugging Face caches and training runs (`./workspace`)
* The API container mounts `/var/run/docker.sock` to stop/start containers A/B, and runs as
  root so it can use the socket. Anyone with access to the API can therefore control Docker on
  the host. That's why it binds to 127.0.0.1 by default and refuses a LAN bind without a real
  `S1_API_KEY`.

## Pinning versions

* **vLLM:** set `S1_VLLM_VERSION` in `.env` to the `vllm/vllm-openai` tag you validated;
  the default is `latest`. The launcher tolerates CLI drift, but a pin keeps rebuilds
  reproducible.
* **Trainer:** `docker/trainer/requirements.txt` is not pinned yet. Each training run
  records the exact package versions it used in `workspace/runs/<run>/result.json` under
  `environment`. After a successful run, copy them into the file as `name==version`.

## Mac oracle

Run `mac/setup_oracle.sh`. It sets `OLLAMA_HOST=0.0.0.0:11434`, keep-alive and parallelism,
then pulls the model. Point `S1_ORACLE_URL` at the Mac. For screenshot-only GUIs, also pull a
vision model and set `S1_ORACLE_VISION_MODEL`.

To verify connectivity from the Linux host: `curl http://<mac-ip>:11434/api/tags`, then
`docker compose exec api systemone doctor`.

## Ports

| Port | Service |
|---|---|
| 3090 | Dashboard (`S1_DASHBOARD_PORT`, bound to `S1_BIND_ADDR`) |
| 8090 | REST + WebSocket API (`S1_API_PORT`, bound to `S1_BIND_ADDR`) |
| 8091 | Student vLLM (debug, 127.0.0.1 only, no auth) |
| 8092 | Triage vLLM (debug, 127.0.0.1 only, no auth) |
| 6379 | Redis (127.0.0.1 only, password required) |
| 11434 | Ollama on the Mac |

## Single-GPU or different hardware

Everything is configurable through `S1_*` variables. On one GPU, point `S1_TRIAGE_URL` at
any OpenAI-compatible endpoint (or at the student itself) and remove the `triage` service.
With more VRAM, raise `S1_GPU_MEMORY_UTILIZATION`, `S1_MAX_MODEL_LEN` and the triage model size.
