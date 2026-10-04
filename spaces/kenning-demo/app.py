"""Kenning demo: ask typed questions about any text or JSON and see calibrated answers.

Runs systemonedev/kenning-large-v0.4 in-process through the `systemone-client` package, on a
GPU when there is one (Colab) and otherwise on CPU (KENNING_DEVICE overrides). Nothing you type
is stored; there are no API keys.

    python app.py                 # local
    # Colab: notebooks/kenning_demo.ipynb in SystemOne Builder
"""

from __future__ import annotations

import html
import json
import os
import time

import gradio as gr
import torch
from systemone import Choice, Kenning, Noul, Score

MODEL_ID = "systemonedev/kenning-large-v0.4"
MAX_STATE_CHARS = 4000
DEVICE = os.environ.get("KENNING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
model = Kenning.from_pretrained(MODEL_ID, device=DEVICE)
WHERE = "on a GPU" if DEVICE.startswith("cuda") else "on CPU (about 40 ms on a GPU)"

TYPES = ["noul (yes/no)", "choice (pick one)", "score (ordered scale)"]

# (state, [(type, instructions, options)] x 3)
PRESETS = {
    "Phishing triage": (
        json.dumps({"email": {
            "from": "security@paypa1-support.com",
            "subject": "Your account will be closed",
            "body": "URGENT: we detected unusual activity. Verify your identity within 24 hours at "
                    "http://paypa1-support.com/verify or your account will be closed."}}, indent=2),
        [(TYPES[0], "Is this email a phishing attempt?", ""),
         (TYPES[1], "What kind of email is this?",
          "phishing: Tries to steal credentials, money or access\nspam: Unwanted marketing\n"
          "legitimate: A genuine message from a real sender"),
         (TYPES[2], "How much pressure does the sender put on the reader?", "none\nsome\na lot")],
    ),
    "Support ticket routing": (
        json.dumps({"ticket": {"subject": "Refund?", "body": "I was charged twice this month, please fix it."},
                    "customer": {"plan": "pro", "tenure_months": 14}}, indent=2),
        [(TYPES[0], "Is this ticket about billing?", ""),
         (TYPES[1], "Which team should handle it?",
          "billing: Charges, refunds, invoices\ntechnical: Bugs, errors, outages\nother: Anything else"),
         (TYPES[2], "How urgent is it?", "can wait\nthis week\ntoday")],
    ),
    "Content moderation": (
        "Honestly the new update is great, but whoever designed the settings page should be fired "
        "into the sun. Took me 20 minutes to find dark mode.",
        [(TYPES[0], "Does this post harass or threaten a person?", ""),
         (TYPES[1], "What is the overall sentiment?", "positive\nnegative\nmixed"),
         (TYPES[2], "How toxic is the language?", "none\nmild\nsevere")],
    ),
}


def _state(text: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Add some state: text or a JSON object.")
    if len(text) > MAX_STATE_CHARS:
        raise gr.Error(f"Keep the state under {MAX_STATE_CHARS} characters for this demo.")
    if text.startswith("{"):
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return text


def _question(kind: str, instructions: str, options: str):
    instructions = (instructions or "").strip()
    if not instructions:
        return None
    lines = [ln.strip() for ln in (options or "").splitlines() if ln.strip()]
    if kind.startswith("noul"):
        return Noul(instructions)
    if kind.startswith("choice"):
        opts = {}
        for ln in lines:
            name, _, desc = ln.partition(":")
            opts[name.strip()] = desc.strip() or None
        if not 2 <= len(opts) <= 20:
            raise gr.Error(f"'{instructions}': a choice needs 2-20 options, one per line ('name: description').")
        return Choice(instructions, opts)
    if not 2 <= len(lines) <= 10:
        raise gr.Error(f"'{instructions}': a score needs 2-10 levels, lowest first, one per line.")
    return Score(instructions, lines)


def _bar(label: str, p: float, winner: bool) -> str:
    color = "#2b6cb0" if winner else "#8996a8"
    return (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span style="width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(label)}</span>'
            f'<span style="flex:1;background:#e2e8f0;border-radius:4px;height:14px">'
            f'<span style="display:block;width:{p * 100:.1f}%;background:{color};height:14px;border-radius:4px"></span></span>'
            f'<span style="width:56px;text-align:right;font-family:monospace">{p:.3f}</span></div>')


def ask(state_text, t1, i1, o1, t2, i2, o2, t3, i3, o3):
    state = _state(state_text)
    questions = {}
    for n, (t, i, o) in enumerate(((t1, i1, o1), (t2, i2, o2), (t3, i3, o3)), 1):
        q = _question(t, i, o)
        if q is not None:
            questions[f"q{n}"] = q
    if not questions:
        raise gr.Error("Ask at least one question.")
    t0 = time.perf_counter()
    r = model.system_one(state=state, questions=questions)
    ms = (time.perf_counter() - t0) * 1000

    parts = []
    for qid, q in questions.items():
        parts.append(f'<div style="margin:12px 0 4px"><b>{html.escape(q.instructions)}</b></div>')
        if qid in r.nouls:
            p = r.nouls[qid].noul
            parts += [_bar("yes", p, p >= 0.5), _bar("no", 1 - p, p < 0.5)]
            band = "sure: act on yes" if p >= 0.9 else "sure: act on no" if p <= 0.1 else "unsure: send to a person"
            parts.append(f'<div style="color:#55616f;font-size:90%">noul = {p:.3f} → {band} (thresholds 0.9 / 0.1)</div>')
        elif qid in r.choices:
            a = r.choices[qid]
            parts += [_bar(k, v, k == a.choice) for k, v in a.probabilities.items()]
        else:
            a = r.scores[qid]
            parts += [_bar(a.legend.get(k, k), v, False) for k, v in a.probabilities.items()]
            parts.append(f'<div style="color:#55616f;font-size:90%">score = {a.score:.2f} (probability-weighted level index)</div>')
    parts.append(f'<div style="margin-top:12px;color:#55616f;font-size:90%">{r.model} · {ms:.0f} ms {WHERE} · '
                 f'same input, same answer</div>')
    return "".join(parts), r.raw["answers"]


def load_preset(name):
    state, qs = PRESETS[name]
    out = [state]
    for t, i, o in qs:
        out += [t, i, o]
    return out


with gr.Blocks(title="Kenning: System One decision model") as demo:
    gr.Markdown(
        "# Kenning: ask typed questions, get calibrated answers\n"
        "A **System One** model answers *yes/no*, *pick one* and *rate on a scale* questions about your "
        "data with a probability for every option, in one pass, without generating text. This is "
        f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (Apache-2.0), running {'on a GPU' if DEVICE.startswith('cuda') else 'on CPU'}. "
        "Learn more at [systemone.dev](https://systemone.dev) · run it yourself with "
        "[SystemOne Builder](https://github.com/systemonedev/systemone-builder)."
    )
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS), value="Phishing triage", label="Example", scale=1)
    with gr.Row():
        with gr.Column(scale=5):
            state = gr.Code(label="State: text or a JSON object", language="json", lines=12)
            rows = []
            for n in (1, 2, 3):
                with gr.Group():
                    with gr.Row():
                        t = gr.Dropdown(TYPES, label=f"Question {n} type", scale=1)
                        i = gr.Textbox(label="Question", scale=3)
                    o = gr.Textbox(label="Options (choice: 'name: description' per line; score: levels, lowest first)",
                                   lines=3)
                rows += [t, i, o]
            go = gr.Button("Ask Kenning", variant="primary")
        with gr.Column(scale=4):
            answer = gr.HTML(label="Answers")
            raw = gr.JSON(label="Raw answer (System One wire format)")
    gr.Markdown(
        "Probabilities are calibrated on Kenning's training data: measure them on your own data before "
        "letting the model act alone. Nothing you enter is stored. "
        "Not affiliated with TypeSafe AI or Cloudflare."
    )
    preset.change(load_preset, preset, [state] + rows)
    go.click(ask, [state] + rows, [answer, raw])
    demo.load(load_preset, preset, [state] + rows)

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
