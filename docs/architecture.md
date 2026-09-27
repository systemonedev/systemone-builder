# Architecture

## Request path (System-1)

```
client observation ─► StateExtractor ─► PromptBuilder ─► Student (GPU 0 vLLM)
 (html / ax_tree /      fuzzy scrub       static prefix        │ guided JSON + logprobs
  elements / eve /      entity vault      + canonical state    ▼
  syslog / json /                                         ConfidenceScorer
  screenshot*)                                                 │
                         conf ≥ τ ◄────────────────────────────┤
                            │                                  │ conf < τ, invalid, ungrounded, ESCALATE
                            ▼                                  ▼
                        execute                         Triage (GPU 1 vLLM, synchronous)
                                                               │ conf < τ_triage
                                                               ▼
                                                  Oracle queue (Mac Ollama, asynchronous)
                                                  caller gets ESCALATE, or waits
```

\* Screenshots are converted to bbox elements by the Mac vision parser first.

Every decision is appended to the **replay buffer**: a Redis hash plus a sorted-set index
capped at 10,000 records by an atomic Lua append/trim. Each record carries the scrubbed state,
the executed action, the full route, latencies, the student's proposal, and any oracle label.
Decisions are also published on the event bus, which feeds the WebSocket gateway, the
telemetry collector and the routing visualizer.

## Prefix-cache protection

vLLM's automatic prefix caching reuses KV blocks only for byte-identical prefixes, so the
pipeline makes consecutive prompts share as much prefix as possible:

* **Static prefix**: system prompt, action contract, JSON schema and few-shots are identical
  for every request in a domain.
* **Canonical state**: fixed key order from most stable to most volatile (`prompt_key_order`),
  compact separators and sorted nested keys.
* **Fuzzy scrubbing**:
  * timestamps, UUIDs, JWTs, hex digests, opaque tokens and framework-generated ids become
    stable placeholders;
  * cache-busting query parameters become `<VAR>`;
  * counters become `<N>`;
  * volatile log fields (flow ids, ephemeral ports, packet counters) are dropped.
* **Entity vault**: optional reversible tokenization (`203.0.113.9` → `<IP_1>`); actions are
  rehydrated before they are returned.
* **DOM ids**: assigned by document order after filtering, so the same layout yields the same
  ids. Selectors and bounding boxes stay in an out-of-prompt index.

The dashboard shows per-session prefix overlap, vLLM's own prefix-cache hit rate (scraped
from `/metrics`), and cached-token counts per request.

## Confidence

The score combines two signals:

* `self` is the model's own `confidence_score`.
* `tokens` is the joint probability of the tokens spelling each decision field (action,
  target id, verdict, IOCs), taken as the minimum across fields. It is computed from the
  streamed logprobs.

`confidence = self^(1-w) · tokens^w`, followed by an optional per-domain Platt calibration
fitted by the evaluation sandbox. The score is forced to 0 by any of these gates: unparseable
or schema-invalid output, references that don't exist in the state (hallucination), an
explicit `ESCALATE`, or the student being offline for training.

## GPU 0 lifecycle

```
SERVING → DRAINING → PAUSED → FLUSHING → TRAINING → FLUSHING → RELOADING → SERVING
                                                    (any failure → rollback → SERVING)
```

| Phase | What happens |
|---|---|
| DRAINING | The router stops sending traffic to the student; GPU 1 triage serves it, and in-flight requests finish. |
| PAUSED | Container B is stopped. |
| FLUSHING | NVML must report GPU 0 below `S1_VRAM_FLUSH_THRESHOLD_MB`, otherwise the cycle aborts. |
| TRAINING | Container A (Unsloth) runs one-shot. Loss streams from its stdout (`S1_METRIC` lines) into Redis and the dashboard. |
| RELOADING | `workspace/state/student.json` is repointed to the merged weights (or LoRA adapter). Container B restarts, and traffic returns only after `/health` succeeds and `/v1/models` lists the new model. |

## Learning loops

| Loop | Source | Output |
|---|---|---|
| Replay distillation | triage/oracle answers, audited student actions | SFT samples (`state → CoT → action`) |
| Seed synthesis | teacher-generated states per scenario | SFT samples |
| State-delta DPO | reported / inferred failures + delta | DPO pairs (chosen vs. failed) and SFT samples |
| Calibration | evaluation sandbox | Platt parameters pushed to the router |

A training cycle starts automatically once `S1_AUTO_TRAIN_MIN_SAMPLES` new SFT samples or
`S1_DPO_AUTO_TRAIN_MIN_PAIRS` new DPO pairs have accumulated.
