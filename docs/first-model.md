# Build your first System-1 model

This walkthrough takes you from a fresh checkout to a fine-tuned reflex model that is serving
traffic, benchmarked, and improving itself. Part A uses the built-in quickstart (web
automation on a 1.5B model). Part B does the same for a specialization of your own.

Commands run on the Linux host (or WSL2 shell) from the repository root unless noted.

---

## Part 0: One-time setup

### 0.1 Mac oracle

On the Mac:

```bash
./mac/setup_oracle.sh      # OLLAMA_HOST=0.0.0.0, pulls qwen3.8:27b
```

Restart Ollama, then check from the Linux host:

```bash
curl http://<mac-ip>:11434/api/tags
```

### 0.2 Configure `.env`

```bash
cp .env.example .env
```

| Variable | Set it to |
|---|---|
| `S1_ORACLE_URL` | `http://<mac-ip>:11434` (a Tailscale IP works) |
| `S1_API_KEY` | any secret; enter the same value in the dashboard sidebar |
| `HF_TOKEN` | only needed for gated models (Llama, Gemma) |

### 0.3 WSL2 / Docker Desktop

Models, datasets and training runs live in the Docker named volume `systemone-workspace`
(the default). That volume sits on Docker's native disk, so weights load at full speed.
Don't set `S1_WORKSPACE_VOLUME` to a Windows path (`/mnt/c/...`): reading safetensors
through the Windows bridge runs at a few MB/s, and a 1.5B model then takes more than 15
minutes to load.

If you previously ran with `./workspace`, the old folder isn't used anymore. Models download
again into the volume once, which is usually faster than copying them over the bridge.

### 0.4 Start the stack

```bash
git pull
docker compose build            # api, student/triage launcher, trainer, dashboard
docker compose up -d
```

Open **http://&lt;host&gt;:3000**. The sidebar lists every service with its state:

| State | Meaning |
|---|---|
| ● online | operational |
| ▲ starting | the detail column shows progress (downloading, loading weights 2/3 shards, compiling, allocating KV cache) |
| ◆ training | GPU 0 is in a training cycle; the student is intentionally down and triage serves traffic |
| ✕ crashed / offline | the detail column shows the last error line |

On first start, expect several minutes of downloads (1.5B ≈ 3 GB, 14B-AWQ ≈ 10 GB). Later
starts load from the cached volume in well under a minute.

When everything shows online, run:

```bash
docker compose exec api systemone doctor
```

Every line should read `[ok]`.

---

## Part A: Your first model with the quickstart (≈ 10 min after models are cached)

The one-shot `quickstart` service does all of this automatically on `docker compose up`.
Follow it with:

```bash
docker compose logs -f quickstart
```

If it gave up earlier (for example because the student was crashing), run it again:

```bash
docker compose run --rm quickstart
```

What it does, and where to see each step in the UI:

1. **Loads data.** It generates about 600 dummy web-automation training samples (logins,
   search, cookie banners, validation errors, navigation, modals, a canvas target that needs
   `CLICK_XY`, ambiguous pages that must `ESCALATE`) plus 50 unseen held-out samples.
   *Factory & Training → Synthetic factory* shows the SFT and held-out counts for
   `computer_use`.
2. **Trains on GPU 0 under the strict lifecycle.** *Factory & Training → GPU 0 lifecycle*
   steps through serving → draining → paused → flushing → training → reloading → serving.
   While it trains, the sidebar shows Student as ◆ training and routed traffic goes to
   triage. The loss curve is on the *Telemetry* page.
3. **Hot-reloads the fine-tuned weights.** *served model* switches to
   `/workspace/runs/computer_use-sft-…/merged`.
4. **Benchmarks the model.** *Evaluation → Reports* shows accuracy, success rate,
   hallucination ratio, the latency histogram, calibration, and a **ready / not ready**
   verdict.

Try the result:

- *Playground*: domain `computer_use`, the preset login page, press **Act**. You get the
  action, which tier answered (student / triage / oracle), the confidence breakdown and the
  latency.
- *Fast-Slow Routing*: watch decisions animate across the network map. Move the
  `computer_use` threshold slider and see more or fewer requests escalate.

That's a working System-1 model. Part B builds one for your own specialization.

---

## Part B: Your own specialization

### B1. Define the domain

Pick one of:

- **From a prompt.** Open *Prompt-to-Workflow* and describe the job, including what must
  never happen. For example: *"Watch sshd auth logs on our bastion hosts. Block IPs that
  brute-force root or service accounts, never block 10.20.0.0/16, escalate anything touching
  domain admins."* Press **Generate workspace**, review the draft (actions, threshold,
  extractor, scenarios), edit the JSON if needed, then **Activate**. Keep *bootstrap* checked
  so the oracle starts generating training trajectories immediately.
- **From a starter template.** `computer_use`, `secops`, `desktop_vision` or
  `auth_log_bruteforce`:
  ```bash
  curl -X POST localhost:8000/api/v1/templates/auth_log_bruteforce/install \
       -H 'content-type: application/json' -H "X-API-Key: $S1_API_KEY" -d '{"bootstrap": true}'
  ```

Pick the threshold by how costly a wrong autonomous action is: about 0.70 for GUI
convenience tasks, 0.93–0.97 for containment actions.

### B2. Build the training set

Aim for at least 300 accepted SFT samples before the first training cycle. There are three
sources, and you can combine them:

1. **Seed synthesis** (no real data needed). *Factory & Training → Seed synthesis*: choose
   your domain and 20–50 states per scenario. The Mac generates states, reasons about each
   one, and the LLM judge filters the results. Watch *accepted / rejected* grow.
2. **Your real labelled data.** Import it in bulk:
   ```bash
   curl -X POST localhost:8000/api/v1/datasets/<domain>/sft/bulk -H 'content-type: application/json' \
        -H "X-API-Key: $S1_API_KEY" -d @samples.json   # [{"state": {...}, "action": {...}}, ...]
   ```
3. **Live traffic** (the self-improving path). Point your client at `POST /api/v1/act/<domain>`
   (see `examples/`). Anything the student is unsure about is answered by triage or the
   oracle, and the factory turns those answers into training samples automatically.

### B3. Build a held-out set of real samples

This is what tells you whether the model is ready. Use real observations the model will
never train on, ideally 50 or more, covering the hard cases.

- *Evaluation → Held-out data*: paste a JSON array of
  `{"observation" | "state", "expected", "acceptable"?, "tags"?}`, then **Import**.
- Or run live traffic with outcome feedback, then **Promote verified replay traffic**.
  Records that were already used for training are skipped automatically.

### B4. Train

*Factory & Training*: pick the domain, mode **SFT**, then **Run training cycle**. You don't
need to do anything else; the lifecycle handles the GPU. A 1.5B model on a few hundred
samples trains in minutes.

Automatic training also starts once `S1_AUTO_TRAIN_MIN_SAMPLES` (default 256) new samples
have accumulated.

### B5. Evaluate and decide

*Evaluation → Run benchmark*: target **student**. Read the readiness checks:

| Check | What to do if it fails |
|---|---|
| accuracy / success rate | more or better data, especially for the tags that score worst (*by_tag* in the report) |
| hallucination ratio | the model invents element ids or IOCs; add examples where the right answer is ESCALATE |
| p95 latency | keep the student small, shorten states (`max_nodes`, `payload_max`), and check the prefix-cache hit rate on *Telemetry* |
| ECE (calibration) | tick *push the fitted confidence calibration to the live router* and re-run |

Then set the domain's threshold on *Fast-Slow Routing* so that the success rate on
autonomous actions meets your bar.

To compare against the teachers, run the same benchmark with target **triage** and
**oracle**.

### B6. Deploy and keep improving

1. Run your client against `/act/<domain>` (`examples/playwright_agent.py`,
   `examples/suricata_reflex.py`).
2. After each executed action, `POST /api/v1/feedback` with the post-action observation.
   Failures (error banners, no-op clicks, attacks that succeed after an ALLOW) become
   correction candidates.
3. *DPO Corrections*: approve, edit or reject the teacher's corrections. High-confidence ones
   are auto-approved when the judge score is at least `S1_DPO_AUTO_APPROVE_MIN_JUDGE`.
4. Retrain with mode **DPO** once you have about 64 or more pairs (automatic by default),
   then re-run the benchmark.
5. If a new run regresses, *Factory & Training → Training runs → serve* switches back to any
   earlier run, or to the base model.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Weights load at a few MB/s, shards stuck at 0% | The workspace is on a host-shared filesystem (the launcher logs a WARNING). Unset `S1_WORKSPACE_VOLUME` to use the named volume. |
| `OSError: [Errno 14] Bad address` in `shared_offload_region` | The kernel couldn't back the CPU KV offload region. The launcher now probes and shrinks it automatically (look for `CPU KV offload probe` lines). To pin a size yourself, set `S1_KV_OFFLOAD_GB` / `S1_TRIAGE_KV_OFFLOAD_GB` (`0` disables offload). |
| `UVA is not available` | Pinned memory is disabled (WSL2). The launcher enables it; run `wsl --update` if your kernel is older than 4.19.121. |
| `unrecognized arguments` from vLLM | Rebuild the launcher image (`docker compose build student`) and recreate the containers (`docker compose up -d --force-recreate student triage`). |
| Training cycle rolled back | See the run's error on *Factory & Training*. `docker compose logs api` has the full trace. If a reload times out, increase `S1_STUDENT_START_TIMEOUT_S`. |
| Oracle shows `degraded` | Run `ollama pull <model>` on the Mac. |
