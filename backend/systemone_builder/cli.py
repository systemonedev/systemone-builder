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
        elif a.suite == "general":
            from systemone_builder.system_one.general_suite import general_suite
            suite = await general_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "multi":
            from systemone_builder.system_one.multitask_suite import multitask_suite
            suite = await multitask_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "layouts":
            suite = await layout_suite(a.n, a.seed, s.kenning_dir() / "bench_cache")
        elif a.suite == "modern":
            # 20 hand-written short, modern emails (evaluation only, never trained on)
            here = Path(__file__).parent / "system_one" / "suites"
            suite = jsonl_suite(here / "modern_email.jsonl", here / "modern_email.questions.json", "malicious")
        elif a.suite == "modern2":
            # 20 more, written before v0.3's subtle-phishing scenarios: a held-out check for them
            here = Path(__file__).parent / "system_one" / "suites"
            suite = jsonl_suite(here / "modern_email_2.jsonl", here / "modern_email_2.questions.json", "malicious")
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


async def synthetic_email_rows(s, datasets, n_per_class: int, seed: int, subtle_share: float = 0.0):  # noqa: ANN001, ANN201
    """Teacher-written modern emails as training rows (cached in datasets/synthetic-email.jsonl)."""
    import random

    from systemone_builder.adapters.byom import resolve
    from systemone_builder.kenning.data import questions_for
    from systemone_builder.kenning.synthetic_email import generate

    teacher = resolve(s)["triage"]
    tag = f"-subtle{int(subtle_share * 100)}" if subtle_share else ""
    cache = datasets / f"synthetic-email-n{n_per_class}-seed{seed}{tag}.jsonl"
    if cache.exists():
        emails = [json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
        print(f"[kenning-data] reusing {len(emails)} synthetic emails from {cache}")
    else:
        print(f"[kenning-data] asking {teacher.model} at {teacher.url} for {2 * n_per_class} modern emails ...", flush=True)
        emails = await generate(teacher.url, teacher.model, n_per_class, seed, concurrency=32,
                                api_key=teacher.api_key(), subtle_share=subtle_share)
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


async def synthetic_task_rows(s, datasets, per_task: int, seed: int):  # noqa: ANN001, ANN201
    """Label-conditioned synthetic decision tasks (cached in datasets/)."""
    from systemone_builder.adapters.byom import resolve
    from systemone_builder.kenning.synthetic_tasks import TASKS, generate, to_training_rows

    teacher = resolve(s)["triage"]
    cache = datasets / f"synthetic-tasks-n{per_task}-seed{seed}.jsonl"
    if cache.exists():
        texts = [json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
        print(f"[kenning-data] reusing {len(texts)} synthetic task texts from {cache}")
    else:
        print(f"[kenning-data] asking {teacher.model} for {per_task * len(TASKS)} label-conditioned texts "
              f"({len(TASKS)} tasks) ...", flush=True)
        texts = await generate(teacher.url, teacher.model, per_task, seed, concurrency=32, api_key=teacher.api_key())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(t) + "\n" for t in texts))
    meta = {"dataset": f"synthetic: {len(TASKS)} label-conditioned tasks written by {teacher.model}", "file": str(cache),
            "license": "generated (teacher: Qwen2.5-7B-Instruct, Apache-2.0)"}
    return to_training_rows(texts, seed + 13), meta


async def problem_rows(s, datasets, ref: str, n: int, seed: int, holdout: float, teacher_url: str | None,  # noqa: ANN001
                       teacher_model: str | None, check: bool, no_think: bool = False, concurrency: int = 16,
                       balance: bool = False):  # noqa: ANN201
    """Teacher-written rows for a problem spec, plus its held-out benchmark items (cases cached in datasets/)."""
    import re

    from systemone_builder.adapters.byom import resolve
    from systemone_builder.kenning import problems

    prob = problems.load_problem(ref, datasets.parent / "problems")
    if teacher_url:
        url, model, key = teacher_url, teacher_model, os.environ.get("S1_TEACHER_API_KEY")
        if not model:
            raise SystemExit("--teacher-model is required with --teacher-url")
    else:
        t = resolve(s)["triage"]
        url, model, key = t.url, teacher_model or t.model, t.api_key()
    tag = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")[:40]
    cache = datasets / f"problem-{prob.name}-n{n}-seed{seed}-{tag}{'' if check else '-unchecked'}{'-bal' if balance else ''}.jsonl"
    if cache.exists():
        cases = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"[kenning-data] reusing {len(cases)} {prob.name} cases from {cache}")
    else:
        stats: dict[str, int] = {}
        print(f"[kenning-data] asking {model} for {n} {prob.name} cases"
              f"{' (each checked blind by the teacher)' if check else ''} ...", flush=True)
        cases = await problems.write_cases(prob, n, seed, url, model, key, concurrency=concurrency, check=check,
                                           stats=stats, extra={"reasoning_effort": "none"} if no_think else None,
                                           balance=balance)
        print(f"[kenning-data] {prob.name}: {stats.get('asked', 0)} asked, {stats.get('written', 0)} written, "
              f"{stats.get('disagreed', 0)} failed the blind check, {len(cases)} kept", flush=True)
        if len(cases) < n:
            print(f"[kenning-data] warning: only {len(cases)} of {n} {prob.name} cases; a stronger teacher "
                  "(--teacher-url/--teacher-model) or clearer label descriptions keep more", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    train, held = problems.split(cases, holdout, seed, prob.name)
    meta = {"dataset": f"problem {prob.name}: {prob.title}, cases written by {model}", "file": str(cache),
            "license": f"generated (teacher: {model}; check its licence)", "blind_check": check}
    hold = (problems.holdout_items(prob, held), problems.canonical_questions(prob)) if held else None
    return problems.training_rows(prob, train, seed), meta, hold


def import_rows(path: str, licence: str, holdout: float, seed: int):  # noqa: ANN201
    """Your own labelled rows (validated), plus held-out benchmark items."""
    from pathlib import Path

    from systemone_builder.kenning import importer, problems

    src = Path(path)
    qpath = src.with_name(src.stem + ".questions.json")
    rows, questions = importer.load_rows(src, qpath if qpath.exists() else None)
    train, held = problems.split(rows, holdout, seed, src.stem)
    meta = {"dataset": f"imported from {src.name}", "file": str(src), "license": licence}
    items = importer.holdout_items(held, questions, src.stem) if held else []
    print(f"[kenning-data] import {src.name}: {len(rows)} valid rows, {len(items)} held out", flush=True)
    return train, meta, ((items, questions) if items else None)


def cmd_data(a: argparse.Namespace) -> int:
    """Write System One training rows (JSONL): the phishing task, or a mix of general and problem data."""
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
        if a.task == "phishing" or a.phishing_rows:
            await phishing_suite(a.bench_n, a.bench_seed, cache)  # make sure the benchmark set exists to exclude it
        holdouts: dict[str, tuple[list[dict], dict]] = {}
        if a.task == "phishing":
            rows = await phishing_training_rows(a.n, a.seed, cache)
            manifest = None
        else:
            phishing_file = out.parent / "phishing-train.jsonl"
            if a.phishing_rows and not phishing_file.exists():
                phishing_file.parent.mkdir(parents=True, exist_ok=True)
                prow = await phishing_training_rows(4000, 7, cache)
                phishing_file.write_text("".join(json.dumps(r) + "\n" for r in prow))
            extra = {}
            if a.structured:
                from systemone_builder.kenning import structured

                for kind in structured.GENERATORS:
                    extra[f"structured:{kind}"] = (structured.rows(kind, a.structured, a.seed), {
                        "dataset": f"structured {kind} ({structured.FAMILY[kind]}): rule-generated, exact labels",
                        "license": "generated (kenning/structured.py)"})
            for spec in a.problem or []:
                ref, _, n = spec.partition("=")
                prow, meta, hold = await problem_rows(s, datasets, ref, int(n or 500), a.seed, a.holdout,
                                                      a.teacher_url, a.teacher_model, not a.no_check,
                                                      a.teacher_no_think, a.teacher_concurrency, a.balance_problems)
                key = f"problem:{Path(ref).stem}"
                extra[key] = (prow, meta)
                if hold:
                    holdouts[key] = hold
            for i, path in enumerate(a.import_ or []):
                licence = (a.import_licence or [])[i] if i < len(a.import_licence or []) else "provided by the user"
                irow, meta, hold = import_rows(path, licence, a.holdout, a.seed)
                key = f"import:{Path(path).stem}"
                extra[key] = (irow, meta)
                if hold:
                    holdouts[key] = hold
            if a.synthetic_email:
                extra["synth_email"] = await synthetic_email_rows(s, out.parent, a.synthetic_email, a.seed, a.subtle_share)
            if a.synthetic_tasks:
                extra["synth_tasks"] = await synthetic_task_rows(s, out.parent, a.synthetic_tasks, a.seed)
            source_rows = {k: int(v) for k, v in (p.split("=") for p in a.source_rows.split(","))} if a.source_rows else None
            sources = None if not a.sources else [] if a.sources == "none" else a.sources.split(",")
            rows, manifest = await multitask_rows(a.per_source, a.seed, phishing_file if a.phishing_rows else None,
                                                  a.phishing_rows, sources, a.layout_variation, extra, source_rows,
                                                  a.genre_share)
            if holdouts:
                hdir = out.with_suffix(".holdout")
                hdir.mkdir(parents=True, exist_ok=True)
                manifest["holdouts"] = {}
                for key, (items, questions) in holdouts.items():
                    stem = key.replace(":", "-")
                    (hdir / f"{stem}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items),
                                                        encoding="utf-8")
                    (hdir / f"{stem}.questions.json").write_text(json.dumps(questions, indent=1), encoding="utf-8")
                    manifest["holdouts"][key] = {"items": len(items), "file": f"{hdir.name}/{stem}.jsonl"}
                    print(f"[kenning-data] held out {len(items)} {key} items for benchmarking: {hdir / (stem + '.jsonl')}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if manifest:
            out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
            print(f"mean noul target (share of 'yes'): {manifest['noul_mean_target']}")
        print(f"wrote {len(rows)} rows to {out}")
        return 0

    return asyncio.run(go())


def cmd_problems(a: argparse.Namespace) -> int:  # noqa: ARG001
    """List built-in problem specs."""
    from systemone_builder.kenning.problems import builtin_problems

    for p in builtin_problems().values():
        sm = p.summary()
        print(f"{p.name:<18} {p.title}")
        print(f"{'':<18} state: {', '.join(sm['state'])}")
        for qid, labels in sm["questions"].items():
            print(f"{'':<18} - {qid}: {' / '.join(labels)}")
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    """Refit a Kenning model's temperatures on your labelled data (accuracy unchanged)."""
    from pathlib import Path

    from systemone_builder.config import get_settings
    from systemone_builder.kenning import registry
    from systemone_builder.kenning.calibrate import calibrate, save
    from systemone_builder.kenning.model import local_dir

    s = get_settings()
    if "/" in a.model:  # a Hugging Face repo: calibrate a local copy
        if not a.save_as:
            print("a Hugging Face model can't be changed in place: add --save-as <name> for a calibrated local copy")
            return 2
        src = Path(local_dir(a.model))
    else:
        src = registry.model_dir(s.kenning_dir(), a.model)
    report = calibrate(src, Path(a.data), a.url or s.kenning_url)
    print(f"{report['model']} on {report['data']}: {report['questions']} answers")
    for kind in sorted(report["after"]):
        b, f = report["before"][kind], report["after"][kind]
        t0, t1 = report["temperature_before"].get(kind), report["temperature_after"].get(kind)
        temps = f"T {t0} -> {t1}  " if kind != "all" else " " * (len(f"T {t0} -> {t1}  ") if t0 else 0)
        print(f"  {kind:<7} n={b['n']:<5} {temps}accuracy {b['accuracy']:.3f}  ECE {b['ece']:.3f} -> {f['ece']:.3f}  "
              f"NLL {b['nll']:.3f} -> {f['nll']:.3f}")
    if not a.write:
        print("dry run: add --write to save the new temperatures")
        return 0
    dst = src
    if a.save_as:
        dst = s.kenning_dir() / "models" / registry._valid(a.save_as)
        if dst.exists():
            print(f"a model named {a.save_as!r} already exists")
            return 2
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache", "*.lock"))
    save(dst, report, a.save_as)
    print(f"saved to {dst}: activate it again (Models page) to serve the new temperatures")
    return 0


def cmd_label(a: argparse.Namespace) -> int:
    """Distil: add a teacher's soft targets to training rows (Clef by default)."""
    from pathlib import Path

    from systemone_builder.config import get_settings
    from systemone_builder.kenning.distill import label_file

    s = get_settings()
    url = a.teacher_url or s.clef_url
    src = Path(a.data)
    dst = Path(a.out or src.with_name(src.stem + f"-{a.teacher_name}.jsonl"))
    report = label_file(src, dst, url, a.teacher_name, a.alpha, a.batch, a.limit, skip_exact=not a.label_exact)
    print(f"wrote {dst}")
    for source, r in sorted(report.items()):
        print(f"  {source:<12} {r['questions']:>6} questions  teacher agrees with labels {r['agreement']:.1%}  "
              f"failed {r['failed']}")
    return 0


def cmd_publish(a: argparse.Namespace) -> int:
    """Upload a trained Kenning model to the Hugging Face Hub (needs HF_TOKEN with write access)."""
    import os

    from systemone_builder.config import get_settings
    from systemone_builder.kenning import registry

    s = get_settings()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set (a Hugging Face token with write access to the target repo)")
        return 2
    repo = a.repo or f"{a.org}/{a.model}"
    results = s.data_dir / "eval_results"
    bench = registry.recorded_benchmarks(results, a.model)
    print(f"publishing {a.model} to {repo} ({'private' if a.private else 'public'}); "
          f"recorded benchmarks: {', '.join(sorted(bench)) or 'none'}")
    try:
        url = registry.publish(s.kenning_dir(), a.model, repo, token, results, a.private)
    except registry.RegistryError as exc:
        print(f"refused: {exc}")
        return 1
    print(f"published: {url}")
    return 0


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
    sub.add_parser("problems", help="list the built-in problem specs for `systemone data --problem`").set_defaults(
        fn=cmd_problems)
    sub.add_parser("doctor", help="check hardware, Docker, Redis and model endpoints").set_defaults(fn=cmd_doctor)
    b = sub.add_parser("bench", aliases=["s1-bench"], help="benchmark System One engines (Kenning, local LLM, opt-in Jev) on a labelled suite")
    b.add_argument("--suite", default="phishing",
                   help="'general' (the headline: text, tables, conversations, agents, answer quality, records, logs; -n per task), 'phishing', 'multi' (14 decision tasks never trained on; -n per task), 'ood' (tasks never trained on; -n per task), 'layouts' (the phishing emails in 4 layouts), 'modern' (20 hand-written modern emails) or a JSONL file of {id, state, labels}")
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
    d.add_argument("--sources", help="comma-separated subset of: amazon, dbpedia, clinc, boolq, nli, civil, helpsteer, "
                                     "jailbreak, injection, ropes; "
                                     "'none' for problem or imported data only")
    d.add_argument("--structured", type=int, default=0, metavar="N",
                   help="add N rule-generated rows per structured generator (records, tables, agent steps, logs; "
                        "exact labels, see kenning/structured.py)")
    d.add_argument("--genre-share", type=float, default=0.3,
                   help="share of public-source rows that also ask 'what kind of text is this?' (0 = none)")
    d.add_argument("--problem", action="append", metavar="NAME[=ROWS]",
                   help="add teacher-written cases for a problem spec: a built-in name (see `systemone problems`), a "
                        "spec saved on the Train page, or a path to a spec JSON; ROWS defaults to 500 (repeatable)")
    d.add_argument("--import", dest="import_", action="append", metavar="FILE.jsonl",
                   help="add your own labelled rows: training rows, or {state, labels} lines with FILE.questions.json "
                        "next to them (repeatable)")
    d.add_argument("--import-licence", action="append", metavar="TEXT",
                   help="licence of each --import, in the same order (recorded in the manifest)")
    d.add_argument("--holdout", type=float, default=0.1,
                   help="share of each problem and import held out as a benchmark suite (0 = none)")
    d.add_argument("--teacher-url", help="OpenAI-compatible base URL of the teacher that writes problem cases "
                                         "(default: the triage model); key in S1_TEACHER_API_KEY if it needs one")
    d.add_argument("--teacher-model", help="teacher model name (required with --teacher-url)")
    d.add_argument("--balance-problems", action="store_true",
                   help="draw each problem's first question uniformly (balance e.g. keep/drop)")
    d.add_argument("--teacher-no-think", action="store_true",
                   help="send reasoning_effort=none, turning off a reasoning teacher's thinking (Ollama, recent vLLM)")
    d.add_argument("--teacher-concurrency", type=int, default=16, help="parallel teacher requests")
    d.add_argument("--no-check", action="store_true",
                   help="keep problem cases without the teacher's blind check (faster, noisier labels)")
    d.add_argument("--synthetic-email", type=int, default=0, metavar="N",
                   help="add N phishing + N legitimate modern emails written by the triage model from labelled "
                        "scenarios (--task multitask; cached in datasets/)")
    lb = sub.add_parser("label", help="distil: add a teacher's soft targets to training rows (Clef by default)")
    lb.add_argument("data", help="training JSONL (from `systemone data`)")
    lb.add_argument("--out", help="output JSONL (default: <data>-<teacher-name>.jsonl)")
    lb.add_argument("--teacher-url", help="teacher server root speaking /v1/systemone/batch (default: S1_CLEF_URL)")
    lb.add_argument("--teacher-name", default="clef-flash")
    lb.add_argument("--alpha", type=float, default=0.5, help="1 = teacher only, 0.5 = average with existing labels")
    lb.add_argument("--batch", type=int, default=16)
    lb.add_argument("--limit", type=int, help="only the first N rows (quick checks)")
    lb.add_argument("--label-exact", action="store_true",
                    help="also send rule-generated (exact-label) rows to the teacher (default: pass them "
                         "through untouched; their labels are exact and they are the slowest rows)")
    lb.set_defaults(fn=cmd_label)
    cb = sub.add_parser("calibrate", help="refit a Kenning model's temperatures on your own labelled data "
                                          "(the model must be the one being served)")
    cb.add_argument("model", help="registered model name, or a Hugging Face id with --save-as")
    cb.add_argument("--data", required=True, help="labelled JSONL: training rows, or {state, labels} lines with "
                                                 "<file>.questions.json next to it")
    cb.add_argument("--url", help="Kenning server (default: S1_KENNING_URL)")
    cb.add_argument("--write", action="store_true", help="save the new temperatures (default: report only)")
    cb.add_argument("--save-as", metavar="NAME", help="write a calibrated copy under this name instead of in place")
    cb.set_defaults(fn=cmd_calibrate)
    pb = sub.add_parser("publish", help="upload a trained Kenning model to the Hugging Face Hub (Apache-2.0 models only)")
    pb.add_argument("model", help="registered model name, e.g. kenning-large-v0.4")
    pb.add_argument("--org", default="systemonedev", help="Hugging Face user or org (repo: <org>/<model>)")
    pb.add_argument("--repo", help="full repo id, overriding --org")
    pb.add_argument("--private", action="store_true")
    pb.set_defaults(fn=cmd_publish)
    d.add_argument("--subtle-share", type=float, default=0.0,
                   help="share of synthetic emails drawn from calm-lure / legitimate-twin pairs")
    d.add_argument("--synthetic-tasks", type=int, default=0, metavar="N",
                   help="add N label-conditioned texts per synthetic task (10 tasks) written by the triage model")
    d.add_argument("--source-rows", help="per-source row counts overriding --per-source, e.g. nli=20000,clinc=3000")
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
