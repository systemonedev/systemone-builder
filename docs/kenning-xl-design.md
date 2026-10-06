# Kenning-XL (v0.6): the architecture change

## Where v0.5 is stuck, and why it's architectural

v0.5 (435M cross-encoder) on the general suite: macro 0.679 vs Clef 0.813, Jev 0.840. It now **beats**
both big engines on safety and answer-quality, but it's stuck on exactly the families that need
**computation over the state**:

| family | v0.5 | Clef | the failure |
|---|---|---|---|
| records | 0.587 | 0.857 | can't sum amounts, compare to a limit, count rows |
| table | 0.520 | 0.860 | can't read a cell by row/col; 512-token truncation |
| logs | 0.520 | 0.740 | can't count errors over a window; truncation |

This is not a data gap — v0.5 trained on thousands of these with exact labels and some tasks *regressed*
(foreign-transaction 0.88→0.64, over-limit 0.50→0.42). It's the **architecture**: a cross-encoder scores
each option `(state, "answer: X")` **independently** with a single entailment pass over ≤512 tokens.
There is no place in that computation to add two numbers, scan a column, or hold 2,000 log lines. More
parameters wouldn't fix the *shape* of the computation.

## The idea: a two-tier System One — fast reflex + deliberate reasoner

The project's whole thesis is System 1 vs System 2. v0.6 makes that literal instead of metaphorical:

- **Tier 1 — Kenning (v0.5 cross-encoder, 435M, ~35 ms).** The reflex. It already *beats Clef* on text,
  safety, routing, conversation, answer-quality. Keep it as the default; it answers most decisions.
- **Tier 2 — Kenning-XL (a small decoder + a decision head, ~50–100 ms).** The deliberate reasoner, used
  for the families Tier 1 structurally can't do (records, tables, logs) **or** whenever Tier 1 is
  unsure (low calibrated confidence). An LM backbone can actually compute over the state, and it reads
  4k+ tokens so nothing is truncated.

A one-line **gate** routes each question: send it to XL if its family is a known hard one, or if Tier 1's
confidence is below a threshold. The aggregate latency stays near Tier 1's because the easy 70–80% never
pay the XL cost — this is the "local GPU is all you need" advantage used deliberately: we *have* latency
headroom vs Clef (35 ms vs 125 ms), so spend it only where it buys accuracy.

This is novel for the decision-model space: Clef/Jev are single monolithic models. A calibrated
reflex+deliberation cascade on one box, each tier open and swappable, is a genuinely different design —
and it's the honest expression of what "System One" was supposed to mean.

## Kenning-XL itself: one-pass constrained readout, not generation

The decision-model contract is "typed questions in, calibrated probabilities out, no text." A decoder LM
can honour that **without generating** by reading the next-token distribution at a decision position,
restricted to the answer tokens:

- **noul** → render `...<state><question> Answer (yes/no):` and read the logits for the ` yes` / ` no`
  tokens; `P(yes) = softmax([z_yes, z_no])[0]`.
- **choice** → map each option to a distinct sentinel token (`A`, `B`, … or the option's first token
  made unique); read logits over just those; softmax.
- **score** → levels map to tokens `0..k`; read logits over them.

One forward pass, no sampling, deterministic, temperature-calibrated per type exactly like the
cross-encoder. The win over the cross-encoder is that this single pass runs the **full LM** over the
**whole state** — so latent arithmetic, column scans and counting are now *possible*, and the 512 cap is
gone (4k+ context).

**Where one pass isn't enough (hard arithmetic), distill the reasoning in:**
- A reasoning teacher (local **DeepSeek-R1-Distill-Qwen-32B-abliterated**, which emits chain-of-thought,
  and the **Qwen3-30B-A3B-abliterated** writer — both won't refuse security-adjacent cases) produces a
  short reasoning trace **and** the answer for the hard record/table/log cases.
- The student is trained two ways on the same items:
  1. **answer-head distillation**: match the teacher's (and Clef's calibrated) answer distribution at
     the readout position — the fast default path.
  2. **optional trace supervision**: also train it to generate the short reasoning before the answer, so
     a **"deliberate mode"** exists (generate ≤64 reasoning tokens, then read the answer) for the hardest
     questions — a latency/accuracy dial, off by default.
- Calibrated soft labels still come from **Clef** (the batch-hang is an ops bug to fix, not a design
  blocker; single-request labelling works).

## Backbone and training (fits our 2×3090)

- **Student:** `Qwen/Qwen3-1.7B` first (Apache-2.0, ~50–70 ms one-pass on a 3090), then try
  `Qwen3-4B-Instruct-2507` if 1.7B underfits records. Both local-cacheable.
- **Fine-tune:** QLoRA (4-bit base + LoRA adapters) on one 3090; the decision-head readout is just the
  LM head restricted to answer tokens, so the only new params are the LoRA adapters (+ optional per-type
  temperature). Loss = cross-entropy / KL between the student's answer-token distribution and the target
  (hard label ⊕ Clef soft label), plus optional LM loss on the reasoning trace.
- **Data:** the v0.6 corpus is the v0.5 generators **plus** reasoning traces on the hard families from
  the abliterated reasoner — heavy on records/tables/logs where the lift is. Long states kept whole
  (that's the point).
- **Teachers, this round:** abliterated Qwen3-30B (writer) + abliterated DeepSeek-R1-32B (reasoning
  traces) + Clef (calibrated labels). The abliteration matters: these cases include security-adjacent
  content a guarded model refuses to write.

## How it plugs into the builder

- New engine `kenning-xl` (a `serve` variant that loads the decoder + does the constrained readout at
  `/v1/systemone`), so it benchmarks and serves through the exact same wire format and bench harness.
- New `train_xl.py` (QLoRA + readout distillation), exposed as a train kind so the Train page and the
  recipe can build it.
- The gate (`kenning` → escalate to `kenning-xl`) is a thin router config, measurable on Verify.

## Ship gate (honesty rules unchanged)
Match or beat Clef's macro on `general` **with the cascade** (Tier 1 + XL on hard/unsure), no regression
on the families Tier 1 already wins, latency p50 still well under Clef's. Only then compare to Jev and
publish. Never train on Jev output.

## Build order (incremental, validated — no blind overnight runs)
1. **Prove the readout** (no training): load base Qwen3-1.7B, implement the constrained noul/choice/score
   readout, run it zero-shot on a slice of the general records/tables/logs items. If a *4k-context,
   untrained* 1.7B already clears the 435M cross-encoder on records, the architecture thesis holds.
2. Build `train_xl.py` (QLoRA + distillation), prove on a tiny slice attended (loss drops, held-out up).
3. Generate the reasoning-trace data (abliterated teachers) for the hard families.
4. Full QLoRA train; benchmark the **cascade** vs Clef/Jev; write it up.
