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

import errno
import json
import mmap
import os
import platform
import re
import shlex
import subprocess
import time
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


def pin_gpu(tag: str) -> None:
    """Pin this process to GPU ``S1_GPU_INDEX`` when isolation is not enforced.

    Compose (``device_ids``) and the orchestrator (``DeviceRequest``) expose a
    single GPU per container on native Linux, where index 0 inside the
    container is the right one. Docker Desktop / WSL2 exposes every GPU to every
    container regardless, so CUDA would default to GPU 0 for all of them and
    the triage server would land on the student's GPU. In that case pin
    explicitly. PCI bus order makes CUDA's numbering match NVML's.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    want = os.environ.get("S1_GPU_INDEX")
    if want is None or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        visible = [ln for ln in out.splitlines() if ln.startswith("GPU ")]
    except (OSError, subprocess.TimeoutExpired):
        visible = []
    if not visible:  # no nvidia-smi in the image: ask NVML directly
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = [str(i) for i in range(pynvml.nvmlDeviceGetCount())]
            pynvml.nvmlShutdown()
        except Exception:
            visible = []
    if len(visible) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = want
        print(f"[{tag}] {len(visible)} GPUs visible (container GPU isolation not enforced, e.g. WSL2): "
              f"pinning to GPU {want} via CUDA_VISIBLE_DEVICES", flush=True)


def in_wsl() -> bool:
    # Same test vLLM uses (vllm/platforms/interface.py); inside a container the
    # kernel string is the host's, so this also detects Docker Desktop / WSL2.
    return "microsoft" in " ".join(platform.uname()).lower()


SLOW_FS = {"9p", "drvfs", "fuse", "fuse.grpcfuse", "virtiofs", "fakeowner", "cifs", "smb3", "nfs", "nfs4"}


def mount_fstype(path: str) -> str | None:
    """Filesystem type of the mount holding ``path`` (longest prefix wins)."""
    best, fstype = "", None
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and path.startswith(parts[1]) and len(parts[1]) > len(best):
                    best, fstype = parts[1], parts[2]
    except OSError:
        return None
    return fstype


def warn_slow_storage() -> None:
    hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    fstype = mount_fstype(hf_home)
    if fstype in SLOW_FS:
        print(
            f"[serve_{ROLE}] WARNING: model cache {hf_home} is on a '{fstype}' mount. Weight loading from a "
            "host-shared filesystem (e.g. a Windows drive under WSL2) can run at a few MB/s. Use the default "
            "named volume (S1_WORKSPACE_VOLUME unset) or keep the repo inside the Linux filesystem.",
            flush=True,
        )


SHM = "/dev/shm"


def clean_stale_offload_files() -> None:
    """Remove vLLM CPU-offload regions left in /dev/shm by crashed engines.

    With ``ipc: host`` every vLLM container shares the host's /dev/shm (RAM-backed
    tmpfs). An engine that dies after creating its region can leave a file of
    tens of GB behind; it survives container restarts and the next start fails
    with ``OSError: [Errno 28] No space left on device``. Unlinking is safe even
    for a region a running engine has mapped: its mapping stays valid and the
    memory is released when that process exits.
    """
    freed = 0
    try:
        names = os.listdir(SHM)
    except OSError:
        return
    for name in names:
        if name.startswith(("vllm_offload_", "s1_offload_probe_")):
            path = os.path.join(SHM, name)
            try:
                freed += os.stat(path).st_size
                os.unlink(path)
            except OSError:
                pass
    if freed:
        print(f"[serve_{ROLE}] removed stale vLLM offload files from {SHM} ({freed / 1024**3:.1f} GB)", flush=True)


def shm_free_gb() -> float:
    try:
        st = os.statvfs(SHM)
        return st.f_bavail * st.f_frsize / 1024**3
    except OSError:
        return 0.0


def probe_offload_gb(requested: int) -> int:
    """Largest CPU KV offload region (GB) the kernel can actually back.

    vLLM pre-faults its offload region in /dev/shm with
    madvise(MADV_POPULATE_WRITE) and only tolerates EINVAL; kernels that cannot
    back the pages (WSL2, tight shm or memory limits) return EFAULT/ENOMEM and
    the engine dies after the model has loaded. Probe with the same call first
    and halve the size until it succeeds.
    """
    free = shm_free_gb()
    # leave headroom in /dev/shm for NCCL/IPC and a co-located vLLM instance
    cap = int(free * float(os.environ.get("S1_KV_OFFLOAD_MAX_SHM_FRACTION", "0.45")))
    size = min(requested, cap)
    print(f"[serve_{ROLE}] {SHM}: {free:.1f} GB free; offload capped at {size} GB (requested {requested} GB)", flush=True)
    populate = getattr(mmap, "MADV_POPULATE_WRITE", 23)
    while size >= 1:
        path = f"/dev/shm/s1_offload_probe_{os.getpid()}"
        nbytes = size * 1024**3
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        try:
            os.unlink(path)
            os.ftruncate(fd, nbytes)
            with mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE) as m:
                t0 = time.monotonic()
                m.madvise(populate, 0, nbytes)
                print(f"[serve_{ROLE}] CPU KV offload probe: {size} GB OK ({time.monotonic() - t0:.1f}s)", flush=True)
                return size
        except OSError as exc:
            if exc.errno == errno.EINVAL:  # no MADV_POPULATE_WRITE: vLLM falls back itself
                return size
            print(f"[serve_{ROLE}] CPU KV offload probe: {size} GB failed ({errno.errorcode.get(exc.errno, exc.errno)})", flush=True)
            size //= 2
        finally:
            os.close(fd)
    return 0


def wsl_safe_mode() -> bool:
    """S1_VLLM_SAFE_MODE: 1 = --enforce-eager, anything else = off (the default)."""
    return os.environ.get("S1_VLLM_SAFE_MODE", "").strip().lower() in ("1", "true", "on", "yes")


def log_memory_budget() -> None:
    """Print the memory this container may use, so logs show each engine's budget."""
    limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw != "max" and int(raw) < 1 << 60:
                limit = int(raw)
            break
        except (OSError, ValueError):
            continue
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = None
    gb = lambda b: f"{b / 1024**3:.0f} GB" if b else "unknown"  # noqa: E731
    print(f"[serve_{ROLE}] memory: container limit {gb(limit) if limit else 'none'}, host/VM total {gb(total)}", flush=True)


def main() -> None:
    pin_gpu(f"serve_{ROLE}")
    log_memory_budget()
    warn_slow_storage()
    if os.environ.get("S1_CLEAN_SHM", "1") != "0":
        clean_stale_offload_files()
    # vLLM parses these with int(): an empty value (e.g. "VAR=" in .env or an
    # empty compose default) crashes it, so treat empty as unset.
    for var in ("VLLM_WSL2_ENABLE_PIN_MEMORY", "VLLM_USE_V2_MODEL_RUNNER"):
        if os.environ.get(var) == "":
            del os.environ[var]
    if in_wsl():
        # Never pin host memory on WSL2. vLLM turns it off there on purpose
        # (NVIDIA's CUDA-on-WSL known limitations); forcing it on routes large
        # page-locked buffers through the dxg GPU-paravirtualization channel,
        # and that is what killed the whole WSL2 VM (and once Windows) here.
        # The V1 model runner needs no pinned/UVA buffers, so use it instead
        # of the V2 runner ("RuntimeError: UVA is not available").
        # Override either variable in .env only to experiment.
        os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        print(f"[serve_{ROLE}] WSL2 kernel detected ({platform.release()}): pinned memory "
              f"{'ON (overridden in env)' if os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] == '1' else 'off'}, "
              f"V2 model runner {'on' if os.environ['VLLM_USE_V2_MODEL_RUNNER'] == '1' else 'off'}", flush=True)
    safe = wsl_safe_mode()
    if safe:
        # No torch.compile / CUDA-graph capture at startup: slower per token,
        # but the fewest moving parts while a model loads under WSL2.
        print(f"[serve_{ROLE}] safe mode: --enforce-eager (unset S1_VLLM_SAFE_MODE to disable)", flush=True)
    if ROLE == "triage":
        model = env("S1_TRIAGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
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
    if ROLE == "triage":
        # Same launch as the reference vLLM servers, which run AWQ models on this
        # machine's GPU 1 without trouble: few sequences, chunked prefill, fp16.
        optional: list[tuple[str, str | None]] = [
            ("--dtype", env("S1_DTYPE", "half")),
            ("--max-num-seqs", env("S1_MAX_NUM_SEQS", "3")),
            ("--enable-chunked-prefill", None),
        ]
    else:
        optional = [
            ("--enable-prefix-caching", None),
            ("--enable-prompt-tokens-details", None),
            ("--max-num-seqs", env("S1_MAX_NUM_SEQS", "64")),
            ("--generation-config", "vllm"),
        ]
    if safe:
        optional.append(("--enforce-eager", None))
    # Unset => role default on Linux, but 0 on WSL2: pre-faulting tens of GB of
    # shared, GPU-registered host memory has taken the whole WSL2 VM down.
    # Set S1_KV_OFFLOAD_GB explicitly to opt in (raise it gradually).
    if not os.environ.get("S1_KV_OFFLOAD_GB") and in_wsl():
        print(f"[serve_{ROLE}] WSL2: CPU KV offload disabled by default (set S1_KV_OFFLOAD_GB to enable)", flush=True)
    offload = env("S1_KV_OFFLOAD_GB", "0" if in_wsl() else defaults["offload"])
    if offload != "0" and os.environ.get("S1_KV_OFFLOAD_PROBE", "1") != "0":
        fitted = probe_offload_gb(int(float(offload)))
        if str(fitted) != offload:
            print(f"[serve_{ROLE}] reducing CPU KV offload from {offload} GB to {fitted} GB "
                  "(the kernel could not back more shared memory)", flush=True)
        offload = str(fitted)
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
