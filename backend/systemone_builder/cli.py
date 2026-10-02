"""``systemone`` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys


def cmd_serve(_: argparse.Namespace) -> int:
    from systemone_builder.api.app import main

    main()
    return 0


def cmd_quickstart(a: argparse.Namespace) -> int:
    from systemone_builder.quickstart.runner import run

    return run(a.api, a.api_key or os.environ.get("S1_API_KEY"), a.train_samples, a.heldout_samples, not a.no_train)


def cmd_gen_dataset(a: argparse.Namespace) -> int:
    from systemone_builder.quickstart.webgen import generate

    with open(a.out, "w") as f:
        for row in generate(a.n, a.seed):
            f.write(json.dumps(row) + "\n")
    print(f"wrote {a.n} samples to {a.out}")
    return 0


def cmd_validate_model(a: argparse.Namespace) -> int:
    from systemone_builder.adapters.byom import validate_hf_model

    res = asyncio.run(validate_hf_model(a.model, os.environ.get("HF_TOKEN")))
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


def cmd_templates(_: argparse.Namespace) -> int:
    from systemone_builder.templates import list_templates

    for t in list_templates():
        print(f"{t['id']:<22} {t['kind']:<13} tau={t['threshold']:.2f}  {t['name']}")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    """Benchmark System One engines (local, Jev, LLM) on one labelled suite."""
    import time
    from pathlib import Path

    from systemone_builder.config import get_settings
    from systemone_builder.system_one.bench import format_reports, jsonl_suite, layout_suite, ood_suite, phishing_suite, run_benchmark
    from systemone_builder.system_one.factory import build_engine

    s = get_settings()

    async def go() -> int:
        if a.suite == "phishing":
            suite = await phishing_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "ood":
            suite = await ood_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "layouts":
            suite = await layout_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "modern":
            # 20 hand-written short, modern emails (evaluation only, never trained on)
            here = Path(__file__).parent / "system_one" / "suites"
            suite = jsonl_suite(here / "modern_email.jsonl", here / "modern_email.questions.json", "malicious")
        else:
            if not a.questions:
                print("a JSONL suite needs --questions questions.json", file=sys.stderr)
                return 2
            suite = jsonl_suite(Path(a.suite), Path(a.questions), a.gate)
        engines = [build_engine(k.strip(), s) for k in a.engines.split(",") if k.strip()]
        try:
            res = await run_benchmark(suite, engines, concurrency=a.concurrency, hi=a.hi, lo=a.lo,
                                      repeat_check=a.repeat_check)
        finally:
            for e in engines:
                await e.aclose()
        print(format_reports(suite, res.reports))
        out = Path(a.out or s.data_dir / "eval_results") / f"bench-{suite.name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res.to_json(), indent=1, default=str))
        print(f"\nfull results: {out}")
        return 0

    return asyncio.run(go())


async def synthetic_email_rows(s, datasets, n_per_class: int, seed: int):  # noqa: ANN001, ANN201
    """Teacher-written modern emails as training rows (cached in datasets/synthetic-email.jsonl)."""
    import random

    from systemone_builder.adapters.byom import resolve
    from systemone_builder.kenning.data import questions_for
    from systemone_builder.kenning.synthetic_email import generate

    teacher = resolve(s)["triage"]
    cache = datasets / f"synthetic-email-n{n_per_class}-seed{seed}.jsonl"
    if cache.exists():
        emails = [json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
        print(f"[kenning-data] reusing {len(emails)} synthetic emails from {cache}")
    else:
        print(f"[kenning-data] asking {teacher.model} at {teacher.url} for {2 * n_per_class} modern emails ...", flush=True)
        emails = await generate(teacher.url, teacher.model, n_per_class, seed, api_key=teacher.api_key())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(e) + "\n" for e in emails))
    rng = random.Random(seed + 7)
    rows = []
    for e in emails:
        qs, targets = questions_for(rng, e["malicious"])
        rows.append({"state": {"email": e["email"]}, "questions": qs, "targets": targets})
    meta = {"dataset": f"synthetic: written by {teacher.model} from labelled scenarios", "file": str(cache),
            "license": "generated (teacher: Qwen2.5-7B-Instruct, Apache-2.0)"}
    return rows, meta


def cmd_data(a: argparse.Namespace) -> int:
    """Write System One training rows (JSONL): the phishing task or the multi-task mix."""
    from pathlib import Path

    from systemone_builder.config import get_settings
    from systemone_builder.system_one.bench import phishing_suite
    from systemone_builder.kenning.data import phishing_training_rows
    from systemone_builder.kenning.multitask import multitask_rows

    s = get_settings()
    cache = s.kenning_dir() / "bench_cache"
    datasets = s.kenning_dir() / "datasets"
    out = Path(a.out or datasets / f"{a.task}-train.jsonl")

    async def go() -> int:
        await phishing_suite(a.bench_n, a.bench_seed, cache)  # make sure the benchmark set exists to exclude it
        if a.task == "phishing":
            rows = await phishing_training_rows(a.n, a.seed, cache)
            manifest = None
        else:
            phishing_file = out.parent / "phishing-train.jsonl"
            if not phishing_file.exists():
                phishing_file.parent.mkdir(parents=True, exist_ok=True)
                prow = await phishing_training_rows(4000, 7, cache)
                phishing_file.write_text("".join(json.dumps(r) + "\n" for r in prow))
            extra = {}
            if a.synthetic_email:
                extra["synth_email"] = await synthetic_email_rows(s, out.parent, a.synthetic_email, a.seed)
            rows, manifest = await multitask_rows(a.per_source, a.seed, phishing_file, a.phishing_rows,
                                                  a.sources.split(",") if a.sources else None, a.layout_variation,
                                                  extra)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if manifest:
            out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
            print(f"mean noul target (share of 'yes'): {manifest['noul_mean_target']}")
        print(f"wrote {len(rows)} rows to {out}")
        return 0

    return asyncio.run(go())


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check the local host against the reference topology."""
    from systemone_builder.adapters.byom import resolve
    from systemone_builder.adapters.factory import build_adapter
    from systemone_builder.config import get_settings

    s = get_settings()
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}")

    print("hardware")
    try:
        from systemone_builder.orchestrator.gpu import NvmlGpuMonitor

        gpus = NvmlGpuMonitor().snapshot()
        check("NVML", True, f"{len(gpus)} GPU(s): " + ", ".join(f"{g.name} {g.memory_total_mb // 1024}GB" for g in gpus))
        for g in gpus:
            print(f"         GPU {g.index}: {g.memory_used_mb}/{g.memory_total_mb} MiB used, {g.utilization_pct}% util")
        try:
            import docker as _docker

            from systemone_builder.orchestrator.gpu import placement_warnings

            dc = _docker.from_env()
            up = set()
            for role, name in (("student", s.student_container), ("triage", s.triage_container)):
                try:
                    if dc.containers.get(name).status == "running":
                        up.add(role)
                except Exception:
                    pass
            for w in placement_warnings(gpus, {"student": s.student_gpu, "triage": s.triage_gpu}, up):
                check("GPU placement", False, w)
        except Exception:
            pass
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
        from systemone_builder.datastore.store import create_redis

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
    b = sub.add_parser("bench", aliases=["s1-bench"], help="benchmark System One engines (Kenning, local LLM, opt-in Jev) on a labelled suite")
    b.add_argument("--suite", default="phishing",
                   help="'phishing', 'ood' (tasks never trained on; -n per task), 'layouts' (the phishing emails in 4 layouts), 'modern' (20 hand-written modern emails) or a JSONL file of {id, state, labels}")
    b.add_argument("--questions", help="questions JSON for a JSONL suite ({qid: {type, instructions, criteria}})")
    b.add_argument("--gate", help="noul question used for automation metrics (default: first noul)")
    b.add_argument("--engines", default="kenning,local",
                   help="comma-separated: kenning, local, llm (slow), jev (opt-in: needs TYPESAFE_API_KEY; your TypeSafe agreement applies)")
    b.add_argument("-n", type=int, default=50, help="items to sample for the built-in suite")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--concurrency", type=int, default=8)
    b.add_argument("--hi", type=float, default=0.9, help="act automatically at p >= hi")
    b.add_argument("--lo", type=float, default=0.1, help="close automatically at p <= lo")
    b.add_argument("--repeat-check", type=int, default=10, help="items asked twice to check determinism (0 = off)")
    b.add_argument("--out", help="directory for the JSON results (default: <data_dir>/eval_results)")
    b.set_defaults(fn=cmd_bench)
    d = sub.add_parser("data", aliases=["s1-data"], help="write System One training rows (benchmark items are always excluded)")
    d.add_argument("--task", choices=["phishing", "multitask"], default="multitask")
    d.add_argument("--out", help="JSONL path (default: <kenning_dir>/datasets/<task>-train.jsonl)")
    d.add_argument("-n", type=int, default=4000, help="phishing rows (--task phishing)")
    d.add_argument("--per-source", type=int, default=1200, help="rows per public dataset (--task multitask)")
    d.add_argument("--phishing-rows", type=int, default=1500, help="phishing rows mixed in (--task multitask)")
    d.add_argument("--sources", help="comma-separated subset of: amazon, dbpedia, clinc, boolq, nli, civil")
    d.add_argument("--synthetic-email", type=int, default=0, metavar="N",
                   help="add N phishing + N legitimate modern emails written by the triage model from labelled "
                        "scenarios (--task multitask; cached in datasets/)")
    d.add_argument("--layout-variation", type=float, default=0.75,
                   help="share of rows re-laid out (renamed/nested fields, plain text, metadata) so the model "
                        "does not learn one state layout (--task multitask; 0 = off)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--bench-n", type=int, default=50, help="benchmark suite size to exclude (as used by bench)")
    d.add_argument("--bench-seed", type=int, default=42)
    d.set_defaults(fn=cmd_data)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
