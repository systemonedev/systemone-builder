# Training, DPO & evaluation

## Synthetic factory (Mac)

* **Replay distillation** runs continuously and is on by default (`S1_AUTO_FACTORY`). It
  walks the replay buffer from a Redis cursor:
  * oracle-answered escalations are judged and kept;
  * triage answers are re-derived by the oracle with full CoT;
  * a `replay_sample_rate` share of confident student actions is audited.
* **Seed synthesis** (`POST /api/v1/factory/synthesize`): the oracle invents diverse states
  per scenario (easy, ambiguous, adversarial and escalation cases), scrubs them like live
  traffic, then labels them.
* **LLM-as-a-Judge** gates every sample (`judge_min_score`). The judge score also caps the
  confidence the student is trained to emit, which keeps the self-report calibrated.

Datasets live in `workspace/datasets/<domain>/{sft,dpo,heldout}.jsonl`.

## Training cycle

Trigger a cycle with `POST /api/v1/training/run {"domain": "...", "mode": "sft"|"dpo"}`, or let
the automatic thresholds start one.

* **SFT** retrains a QLoRA adapter on the base model over the cumulative dataset. Loss is
  applied to the completion (the action JSON) only, and all linear projections are targeted.
* **DPO** refines the currently served weights on (chosen, rejected) pairs.

With `S1_RELOAD_MODE=merged` (the default), weights are exported as merged 16-bit and vLLM
restarts on them. With `adapter`, vLLM serves the base model plus the LoRA adapter.

To serve a previous run's weights, or the base model, use `POST /api/v1/training/rollback`.

## State-delta DPO loop

After executing an action, clients `POST /api/v1/feedback` with the post-action observation.
The delta is computed from it: added/removed/changed nodes, error banners, no-op clicks, and
follow-up attacks after an ALLOW. A failure is inferred when the client doesn't state the
outcome. For each failure:

1. The initial state, the failed action and the delta go to the teacher.
2. The teacher proposes a correction.
3. The judge reviews it. Above `S1_DPO_AUTO_APPROVE_MIN_JUDGE` it is auto-approved; otherwise
   it waits in the **DPO Corrections Studio**, where a human can approve, edit or reject it.

Approved corrections become DPO pairs, and also SFT samples.

## Evaluation sandbox

Import real, unseen samples with `POST /api/v1/eval/{domain}/heldout` (raw observations or
states, expected action, acceptable alternatives, tags), or promote verified replay traffic.
Then run `POST /api/v1/eval/run`.

| Metric | Definition |
|---|---|
| accuracy | full action match (label + target/coordinates/text/IOCs), allowing acceptable alternatives |
| success rate | accuracy on the actions the model would take autonomously |
| coverage / escalation rate | share handled without escalating |
| hallucination ratio | ungrounded references or schema-invalid output |
| latency & TTFT | p50/p90/p95/p99, histogram, share under 100 ms |
| calibration | ECE, Brier, reliability bins; the fitted Platt calibration can be pushed to the router |
| deployment readiness | pass/fail against configurable gates |
