---
title: Kenning Demo
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Typed questions in, calibrated answers out. No generation.
models:
  - systemonedev/kenning-large-v0.4
---

# Kenning demo

Ask **System One** questions (yes/no, pick one, rate on a scale) about any text or JSON, and get a
calibrated probability for every option from
[systemonedev/kenning-large-v0.4](https://huggingface.co/systemonedev/kenning-large-v0.4).

- Runs on a free CPU through [`systemone-client`](https://pypi.org/project/systemone-client/)
  (`Kenning.from_pretrained`): about 1-3 seconds per request here, about 40 ms on a GPU.
- Nothing you enter is stored, and the Space uses no API keys.
- Source: [`spaces/kenning-demo`](https://github.com/systemonedev/systemone-builder/tree/main/spaces/kenning-demo)
  in SystemOne Builder. Docs: [systemone.dev](https://systemone.dev).

## Running it

- **Colab (free):** [notebooks/kenning_demo.ipynb](https://colab.research.google.com/github/systemonedev/systemone-builder/blob/main/notebooks/kenning_demo.ipynb) downloads this `app.py` and runs it on
  Colab's GPU with a temporary public link.
- **Locally:** `pip install "systemone-client[local]" gradio && python app.py`.
- **As a Hugging Face Space:** this folder is a ready Gradio Space (the header above is its config).
  Hugging Face requires a PRO (personal) or Team (organisation) plan to host Gradio Spaces on free
  CPUs, so it isn't published yet.

Not affiliated with TypeSafe AI or Cloudflare.
