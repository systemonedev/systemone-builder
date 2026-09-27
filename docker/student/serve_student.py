"""Container B entrypoint: launch the vLLM student runner on GPU 0.

The lifecycle orchestrator hot-reloads new weights by rewriting
``/workspace/state/student.json`` and restarting this container:

    {"model": "/workspace/runs/<run>/merged",      # or a Hugging Face id
     "served_name": "student",
     "lora": {"name": "student-<run>", "path": "/workspace/runs/<run>/adapter"} | null}

Tuning for the sub-100ms reflex path:

* ``--enable-prefix-caching`` - automatic KV-cache sharing across requests
  that share the (scrubbed, canonical) prompt prefix.
* CPU KV-cache offload into the 120GB system RAM (``S1_KV_OFFLOAD_GB``) so
  evicted prefix blocks are swapped out instead of recomputed.
* ``--enable-prompt-tokens-details`` - report cached tokens per request.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

STATE = Path(os.environ.get("S1_STUDENT_STATE", "/workspace/state/student.json"))


def main() -> None:
    state = {}
    if STATE.exists():
        state = json.loads(STATE.read_text())
    model = state.get("model") or os.environ.get("S1_STUDENT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    served = state.get("served_name") or os.environ.get("S1_STUDENT_SERVED_NAME", "student")
    args = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", os.environ.get("S1_VLLM_PORT", "8000"),
        "--served-model-name", served,
        "--gpu-memory-utilization", os.environ.get("S1_GPU_MEMORY_UTILIZATION", "0.85"),
        "--max-model-len", os.environ.get("S1_MAX_MODEL_LEN", "8192"),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--swap-space", os.environ.get("S1_SWAP_SPACE_GB", "16"),
        "--max-num-seqs", os.environ.get("S1_MAX_NUM_SEQS", "64"),
        "--generation-config", "vllm",
    ]
    offload = os.environ.get("S1_KV_OFFLOAD_GB", "32")
    if offload and offload != "0":
        args += ["--kv-offloading-backend", os.environ.get("S1_KV_OFFLOAD_BACKEND", "native"),
                 "--kv-offloading-size", offload]
    lora = state.get("lora")
    if lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        args += ["--enable-lora", "--max-lora-rank", os.environ.get("S1_MAX_LORA_RANK", "64"),
                 "--lora-modules", f"{lora['name']}={lora['path']}"]
    args += shlex.split(os.environ.get("S1_VLLM_EXTRA_ARGS", ""))
    print("[serve_student] exec:", " ".join(shlex.quote(a) for a in args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
