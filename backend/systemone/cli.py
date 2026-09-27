"""``systemone`` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys


def cmd_serve(_: argparse.Namespace) -> int:
    from systemone.api.app import main

    main()
    return 0


def cmd_quickstart(a: argparse.Namespace) -> int:
    from systemone.quickstart.runner import run

    return run(a.api, a.api_key or os.environ.get("S1_API_KEY"), a.train_samples, a.heldout_samples, not a.no_train)


def cmd_gen_dataset(a: argparse.Namespace) -> int:
    from systemone.quickstart.webgen import generate

    with open(a.out, "w") as f:
        for row in generate(a.n, a.seed):
            f.write(json.dumps(row) + "\n")
    print(f"wrote {a.n} samples to {a.out}")
    return 0


def cmd_validate_model(a: argparse.Namespace) -> int:
    from systemone.adapters.byom import validate_hf_model

    res = asyncio.run(validate_hf_model(a.model, os.environ.get("HF_TOKEN")))
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_templates(_: argparse.Namespace) -> int:
    from systemone.templates import list_templates

    for t in list_templates():
        print(f"{t['id']:<22} {t['kind']:<13} tau={t['threshold']:.2f}  {t['name']}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check the local host against the reference topology."""
    from systemone.adapters.byom import resolve
    from systemone.adapters.factory import build_adapter
    from systemone.config import get_settings

    s = get_settings()
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}")

    print("hardware")
    try:
        from systemone.orchestrator.gpu import NvmlGpuMonitor

        gpus = NvmlGpuMonitor().snapshot()
        check("NVML", True, f"{len(gpus)} GPU(s): " + ", ".join(f"{g.name} {g.memory_total_mb // 1024}GB" for g in gpus))
        check("student GPU present", any(g.index == s.student_gpu for g in gpus), f"GPU {s.student_gpu}")
        check("triage GPU present", any(g.index == s.triage_gpu for g in gpus), f"GPU {s.triage_gpu}")
    except Exception as exc:
        check("NVML", False, repr(exc))
    try:
        import docker

        docker.from_env().ping()
        check("Docker Engine", True)
    except Exception as exc:
        check("Docker Engine", False, repr(exc))
    try:
        with open("/proc/meminfo") as f:
            total_kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
        check("system RAM", total_kb / 1024**2 >= 100, f"{total_kb / 1024**2:.0f} GB (reference: 120 GB)")
    except Exception:
        pass
    check("disk free (workspace)", shutil.disk_usage(".").free / 1024**3 > 100, f"{shutil.disk_usage('.').free / 1024**3:.0f} GB free")

    print("services")

    async def svc() -> None:
        from systemone.datastore.store import create_redis

        r = create_redis(s.redis_url)
        try:
            await r.ping()
            info = await r.info("memory")
            check("Redis", True, f"maxmemory {info.get('maxmemory_human')}")
        except Exception as exc:
            check("Redis", False, repr(exc))
        finally:
            await r.aclose()
        for role, cfg in resolve(s).items():
            a = build_adapter(cfg.adapter, cfg.url, cfg.model, 10, cfg.api_key())
            h = await a.health()
            detail = f"{cfg.adapter} {cfg.url} ({cfg.model})"
            if h.get("ok"):
                try:
                    models = await a.list_models()
                    if role == "oracle" and cfg.model not in models:
                        h["ok"] = False
                        detail += f" - model not pulled; run: ollama pull {cfg.model}"
                except Exception:
                    pass
            check(f"{role} endpoint", bool(h.get("ok")), detail)
            await a.aclose()

    asyncio.run(svc())
    print("\nall checks passed" if ok else "\nsome checks failed")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="systemone", description="systemone-builder: System-1 reflex model distillation")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the orchestrator + LAN REST API").set_defaults(fn=cmd_serve)
    q = sub.add_parser("quickstart", help="load the dummy dataset, train and benchmark the tiny student")
    q.add_argument("--api", default=os.environ.get("S1_API", "http://localhost:8000"))
    q.add_argument("--api-key", default=None)
    q.add_argument("--train-samples", type=int, default=600)
    q.add_argument("--heldout-samples", type=int, default=50)
    q.add_argument("--no-train", action="store_true")
    q.set_defaults(fn=cmd_quickstart)
    g = sub.add_parser("gen-dataset", help="write the dummy web-automation dataset as JSONL")
    g.add_argument("out")
    g.add_argument("-n", type=int, default=600)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(fn=cmd_gen_dataset)
    v = sub.add_parser("validate-model", help="check a Hugging Face model can be fine-tuned with Unsloth")
    v.add_argument("model")
    v.set_defaults(fn=cmd_validate_model)
    sub.add_parser("templates", help="list starter templates").set_defaults(fn=cmd_templates)
    sub.add_parser("doctor", help="check hardware, Docker, Redis and model endpoints").set_defaults(fn=cmd_doctor)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
