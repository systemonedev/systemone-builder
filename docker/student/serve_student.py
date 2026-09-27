"""vLLM launcher for Container B (student, GPU 0) and Container C (triage, GPU 1).

Student: the lifecycle orchestrator hot-reloads new weights by rewriting
``/workspace/state/student.json`` and restarting the container:

    {"model": "/workspace/runs/<run>/merged",      # or a Hugging Face id
     "served_name": "student",
     "lora": {"name": "student-<run>", "path": "/workspace/runs/<run>/adapter"} | null}

Triage (``S1_VLLM_ROLE=triage``): serves ``S1_TRIAGE_MODEL`` under its own id.

vLLM's CLI changes between releases (e.g. ``--swap-space`` was removed), so
optional tuning flags are checked against the installed version's
``vllm serve --help`` output and unsupported ones are dropped with a warning
instead of crashing the container:

* ``--enable-prefix-caching`` - automatic KV-cache sharing across requests
* CPU KV-cache offload into system RAM (``S1_KV_OFFLOAD_GB``)
* ``--enable-prompt-tokens-details`` - report cached tokens per request
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path

STATE = Path(os.environ.get("S1_STUDENT_STATE", "/workspace/state/student.json"))
ROLE = os.environ.get("S1_VLLM_ROLE", "student")


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def supported_flags() -> set[str] | None:
    """Flags accepted by the installed ``vllm serve`` (None if undetectable)."""
    for args in (["vllm", "serve", "--help=all"], ["vllm", "serve", "--help"]):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = out.stdout + out.stderr
        flags = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", text))
        if "--max-model-len" in flags:
            return flags
    return None


def in_wsl() -> bool:
    # Same test vLLM uses (vllm/platforms/interface.py); inside a container the
    # kernel string is the host's, so this also detects Docker Desktop / WSL2.
    return "microsoft" in " ".join(platform.uname()).lower()


def main() -> None:
    if in_wsl() and "VLLM_WSL2_ENABLE_PIN_MEMORY" not in os.environ:
        # vLLM disables pinned memory under WSL2 by default, and its V2 model
        # runner cannot start without it ("RuntimeError: UVA is not available").
        # WSL2 kernels >= 4.19.121 support pinned memory; set
        # VLLM_WSL2_ENABLE_PIN_MEMORY=0 explicitly to opt out.
        os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
        print(f"[serve_{ROLE}] WSL2 kernel detected ({platform.release()}): enabling VLLM_WSL2_ENABLE_PIN_MEMORY=1", flush=True)
    if ROLE == "triage":
        model = env("S1_TRIAGE_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        served, lora = model, None
        defaults = {"util": "0.90", "len": "16384", "offload": "16"}
    else:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        model = state.get("model") or env("S1_STUDENT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        served = state.get("served_name") or env("S1_STUDENT_SERVED_NAME", "student")
        lora = state.get("lora")
        defaults = {"util": "0.85", "len": "8192", "offload": "32"}

    required = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", env("S1_VLLM_PORT", "8000"),
        "--served-model-name", served,
        "--gpu-memory-utilization", env("S1_GPU_MEMORY_UTILIZATION", defaults["util"]),
        "--max-model-len", env("S1_MAX_MODEL_LEN", defaults["len"]),
    ]
    # (flag, value or None) - each dropped if this vLLM does not know it
    optional: list[tuple[str, str | None]] = [
        ("--enable-prefix-caching", None),
        ("--enable-prompt-tokens-details", None),
        ("--max-num-seqs", env("S1_MAX_NUM_SEQS", "64")),
        ("--generation-config", "vllm"),
    ]
    offload = env("S1_KV_OFFLOAD_GB", defaults["offload"])
    if offload != "0":
        optional += [("--kv-offloading-backend", env("S1_KV_OFFLOAD_BACKEND", "native")),
                     ("--kv-offloading-size", offload)]
    swap = os.environ.get("S1_SWAP_SPACE_GB")  # only for older vLLM releases
    if swap:
        optional.append(("--swap-space", swap))
    if lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        optional += [("--enable-lora", None), ("--max-lora-rank", env("S1_MAX_LORA_RANK", "64")),
                     ("--lora-modules", f"{lora['name']}={lora['path']}")]

    flags = supported_flags()
    args = list(required)
    for flag, value in optional:
        if flags is not None and flag not in flags:
            if flag in ("--enable-lora", "--lora-modules"):
                raise SystemExit(f"[serve_{ROLE}] this vLLM does not support {flag}; set S1_RELOAD_MODE=merged")
            print(f"[serve_{ROLE}] warning: installed vLLM does not support {flag}; skipping", flush=True)
            continue
        args += [flag] + ([value] if value is not None else [])
    args += shlex.split(os.environ.get("S1_VLLM_EXTRA_ARGS", ""))
    print(f"[serve_{ROLE}] exec:", " ".join(shlex.quote(a) for a in args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
