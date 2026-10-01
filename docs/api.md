# API reference

Base URL `http://<linux-host>:8000/api/v1`. When `S1_API_KEY` is set, send `X-API-Key` on every
request (the WebSocket accepts `?api_key=`). `GET /health` is always open. Interactive docs:
`/docs` (Swagger) and `/redoc`.

## WebSocket

`GET /api/v1/ws?channels=routing,telemetry` streams bus events
`{channel, type, ts, data}`. The first message is `{type: "hello", telemetry, events}`.
Channels: `routing`, `telemetry`, `lifecycle`, `training`, `factory`, `dpo`, `eval`, `workflow`, `replay`, `system`.

## system

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health |
| GET | `/health/deep` | Deep Health |
| GET | `/hardware` | Hardware |
| GET | `/hardware/gpus` | Gpus |
| GET | `/hardware/audit` | Audit |
| GET | `/hardware/containers/{name}/logs` | Container Logs |

## replay

| Method | Path | Description |
|---|---|---|
| GET | `/replay/stats` | Stats |
| POST | `/replay` | Append |
| GET | `/replay` | List Records |
| GET | `/replay/scrub` | Temporal scrubber: jump to an offset from the head of the buffer. |
| GET | `/replay/{seq}` | Get Record |

## extraction

| Method | Path | Description |
|---|---|---|
| GET | `/domains` | List Domains |
| GET | `/domains/{domain_id}` | Get Domain |
| POST | `/extract/{domain_id}` | Extract |
| GET | `/extract/collector.js` | Browser-side element collector for Playwright (``page.evaluate``). |

## routing

| Method | Path | Description |
|---|---|---|
| POST | `/act/{domain_id}` | Act |
| DELETE | `/sessions/{domain_id}/{session_id}` | Reset Session |
| GET | `/routing/thresholds` | Thresholds |
| PUT | `/routing/thresholds/{domain_id}` | Set Threshold |
| GET | `/routing/stats` | Routing Stats |
| GET | `/routing/escalations` | Escalations |
| GET | `/routing/escalations/{job_id}` | Escalation |
| GET | `/models` | Models |

## factory

| Method | Path | Description |
|---|---|---|
| GET | `/factory/status` | Factory Status |
| POST | `/factory/start` | Factory Start |
| POST | `/factory/stop` | Factory Stop |
| POST | `/factory/synthesize` | Synthesize |
| GET | `/datasets/{domain_id}` | Dataset Stats |
| GET | `/datasets/{domain_id}/{split}` | Dataset Tail |
| POST | `/datasets/{domain_id}/sft` | Add Sample |
| POST | `/datasets/{domain_id}/sft/bulk` | Add Samples Bulk |
| GET | `/training/status` | Training Status |
| POST | `/training/run` | Training Run |
| GET | `/training/runs` | Training Runs |
| GET | `/training/runs/{run_id}/metrics` | Training Metrics |
| POST | `/training/rollback` | Training Rollback |
| POST | `/training/recover` | Training Recover |
| POST | `/vision/parse` | Vision Parse |

## dpo

| Method | Path | Description |
|---|---|---|
| POST | `/feedback` | Feedback |
| GET | `/dpo/stats` | Dpo Stats |
| GET | `/dpo/candidates` | Dpo Candidates |
| GET | `/dpo/candidates/{cid}` | Dpo Candidate |
| POST | `/dpo/candidates/{cid}/review` | Dpo Review |
| POST | `/dpo/candidates/{cid}/retry` | Dpo Retry |
| POST | `/eval/{domain_id}/heldout` | Import Heldout |
| POST | `/eval/{domain_id}/heldout/from-replay` | Promote verified real traffic (outcome=success) into the held-out set. |
| POST | `/eval/run` | Eval Run |
| GET | `/eval/reports` | Eval Reports |
| GET | `/eval/reports/{report_id}` | Eval Report |

## dashboard

| Method | Path | Description |
|---|---|---|
| PUT | `/domains/{domain_id}` | Put Domain |
| DELETE | `/domains/{domain_id}` | Delete Domain |
| GET | `/telemetry` | Telemetry |
| GET | `/events` | Recent Events |
| POST | `/workflows/generate` | Generate |
| GET | `/workflows/drafts` | Drafts |
| POST | `/workflows/activate` | Activate |

## byom

| Method | Path | Description |
|---|---|---|
| GET | `/byom` | Byom |
| PUT | `/byom/{role}` | Swap |
| POST | `/byom/validate` | Validate |
| GET | `/templates` | Templates |
| POST | `/templates/{template_id}/install` | Install |

## system-one

The [Jev](https://docs.typesafe.ai) System One contract, served by the local engine: a client can
switch between TypeSafe Jev and this server by changing only the base URL and key.

| Method | Path | Description |
|---|---|---|
| POST | `/systemone` | `{state, questions}` → `{model, answers, usage}`, Jev's wire format |

```bash
curl -s http://localhost:8090/api/v1/systemone -H "X-API-Key: $S1_API_KEY" -H 'Content-Type: application/json' -d '{
  "state": {"email": {"subject": "Are you at your desk?", "body": "I need a wire transfer sent out immediately."}},
  "questions": {
    "is_malicious": {"type": "noul", "instructions": "Is this email a phishing attempt or threat?"},
    "category": {"type": "choice", "instructions": "What type of email is this?",
                 "criteria": {"Safe": null, "BEC": null, "Credential_Harvesting": null, "Spam": null}},
    "severity": {"type": "score", "instructions": "How severe would it be?",
                 "criteria": ["None", "Low", "Medium", "High", "Critical"]}}}'
```

Question types: `noul` (`criteria` optional), `choice` (`criteria` maps 2–255 option names to a
description or `null`; the local engine reads up to 20), `score` (`criteria` is an ordered list of
2–10 levels). Each answer is read from one forward pass of the local model (`S1_SYSTEM_ONE_LOCAL_URL` /
`S1_SYSTEM_ONE_LOCAL_MODEL`, default: the triage server); `confidence` is `(max p − 1/n)/(1 − 1/n)`, the
formula Jev's published examples follow. The response adds `latency_ms`, which Jev does not send.

### Benchmark: `systemone s1-bench`

Runs the same questions over the same labelled items on each engine and reports accuracy,
calibration (Brier, ECE), gating (automation rate, false positives acted on, false negatives
auto-closed at `--hi`/`--lo`), latency, errors, and determinism (items asked twice).

```bash
docker compose exec api systemone s1-bench --engines local,jev,llm -n 50
docker compose exec api systemone s1-bench --suite /data/my.jsonl --questions /data/questions.json
```

Engines: `local` (this server's engine), `jev` (TypeSafe Jev, authenticated with
`TYPESAFE_API_KEY`), `llm` (the oracle writing JSON answers, the generative baseline). The built-in `phishing` suite
samples a balanced set from the public `zefang-liu/phishing-email-dataset` and caches it, so every
run and engine sees the same emails. Full per-item results are saved to `eval_results/`.
