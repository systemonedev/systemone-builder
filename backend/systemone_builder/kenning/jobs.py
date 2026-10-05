"""Kenning jobs: build data, label with a teacher, train and benchmark from the API.

The Train and Verify pages drive these. One job runs at a time, because they
share a GPU with the serving containers. A job that needs the GPU pauses the
services on it (Kenning, the student, Clef - except the one it uses), waits for
the memory to be released, and restarts what it paused when it ends, whether
it succeeded, failed or was cancelled.

    data   ``systemone data``  (subprocess in the API container)
    label  ``systemone label`` (subprocess; Clef must be running)
    train  ``kenning.train``   (one-shot container from the Kenning image)
    bench  ``systemone bench`` per suite (subprocess)

Every parameter is validated against an allow-list and turned into an argument
list (never a shell string). Jobs are files under ``<kenning_dir>/jobs``
(``<id>.json`` + ``<id>.log``), so they survive API restarts; a job that was
running when the API stopped is marked interrupted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx

from systemone_builder.kenning import registry
from systemone_builder.orchestrator.docker_ctl import RunSpec

if TYPE_CHECKING:
    from systemone_builder.runtime import Runtime

log = logging.getLogger(__name__)

KINDS = ("data", "label", "train", "bench", "recipe")
SUITES = ("general", "multi", "modern2", "modern", "phishing", "layouts", "ood")
ENGINES = ("kenning", "clef", "jev", "local")
DATA_SOURCES = ("amazon", "dbpedia", "clinc", "boolq", "nli", "civil", "helpsteer", "jailbreak", "injection", "ropes")
SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}$")
URL = re.compile(r"^https?://[\w.-]+(:\d+)?(/[\w./-]*)?$")
HOLDOUT = re.compile(r"^holdout:([a-z0-9][a-z0-9._-]{0,63})/([a-z0-9][\w.:-]{0,140})$")
STEP = re.compile(r"step (\d+)/(\d+)")
ROWS = re.compile(r"(\d+)/(\d+) rows")


class JobError(ValueError):
    pass


# ------------------------------------------------------------- validation
def _int(p: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    v = p.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not lo <= v <= hi:
        raise JobError(f"{key} must be an integer from {lo} to {hi}")
    return int(v)


def _float(p: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = p.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise JobError(f"{key} must be a number from {lo} to {hi}")
    return float(v)


def _slug(p: dict[str, Any], key: str, default: str | None = None) -> str:
    v = p.get(key, default)
    if not isinstance(v, str) or not SLUG.match(v):
        raise JobError(f"{key} must be lower-case letters, digits, '.', '_' or '-' (max 64)")
    return v


def dataset_path(home: Path, name: str) -> Path:
    if not SLUG.match(name or ""):
        raise JobError(f"invalid dataset name {name!r}")
    return home / "datasets" / f"{name}.jsonl"


def problems_dir(home: Path) -> Path:
    return home / "problems"


def imports_dir(home: Path) -> Path:
    return home / "imports"


def list_problems(home: Path) -> list[dict[str, Any]]:
    """Built-in problem specs and the ones saved from the Train page."""
    from systemone_builder.kenning.problems import builtin_problems, parse_problem

    out = [{**p.summary(), "builtin": True} for p in builtin_problems().values()]
    for f in sorted(problems_dir(home).glob("*.json")) if problems_dir(home).is_dir() else []:
        try:
            out.append({**parse_problem(json.loads(f.read_text(encoding="utf-8"))).summary(), "builtin": False})
        except (ValueError, OSError):
            continue
    return out


def list_imports(home: Path) -> list[dict[str, Any]]:
    root = imports_dir(home)
    out = []
    for f in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
        if not SLUG.match(f.stem):  # uploads in progress
            continue
        meta = f.with_suffix(".meta.json")
        m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        out.append({"name": f.stem, "rows": m.get("rows"), "questions": m.get("questions", []),
                    "license": m.get("license"), "size_bytes": f.stat().st_size})
    return out


def holdout_paths(home: Path, suite: str) -> tuple[Path, Path]:
    """(items, questions) for a 'holdout:<dataset>/<key>' suite, checked against the dataset manifest."""
    m = HOLDOUT.match(suite)
    if not m:
        raise JobError(f"unknown suite {suite!r} (suites: {', '.join(SUITES)}, or holdout:<dataset>/<key>)")
    ds, key = m.groups()
    manifest = dataset_path(home, ds).with_suffix(".manifest.json")
    info = (json.loads(manifest.read_text()) if manifest.exists() else {}).get("holdouts", {}).get(key)
    if not info:
        raise JobError(f"dataset {ds!r} has no hold-out {key!r}")
    items = (home / "datasets" / info["file"]).resolve()
    if (home / "datasets").resolve() not in items.parents or not items.exists():
        raise JobError(f"hold-out file for {suite!r} is missing")
    return items, items.with_name(items.stem + ".questions.json")


def list_datasets(home: Path) -> list[dict[str, Any]]:
    root = home / "datasets"
    out = []
    for f in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
        if not SLUG.match(f.stem):
            continue
        m = f.with_suffix(".manifest.json")
        manifest = json.loads(m.read_text()) if m.exists() else None
        if manifest is None:  # caches (synthetic emails, tasks) have no manifest: not training sets
            continue
        with f.open("rb") as fh:
            rows = sum(1 for _ in fh)
        out.append({"name": f.stem, "rows": rows, "size_bytes": f.stat().st_size, "modified": f.stat().st_mtime,
                    "sources": {k: {"rows": v.get("rows"), "license": v.get("license"), "dataset": v.get("dataset")}
                                for k, v in (manifest.get("sources") or {}).items()},
                    "teacher": manifest.get("teacher"),
                    "holdouts": {k: v.get("items") for k, v in (manifest.get("holdouts") or {}).items()}})
    return sorted(out, key=lambda d: d["modified"], reverse=True)


def validate(kind: str, p: dict[str, Any], home: Path, *, pipeline: bool, has_jev_key: bool) -> dict[str, Any]:
    """Normalised, validated parameters for a job (raises JobError)."""
    if kind not in KINDS:
        raise JobError(f"kind must be one of {', '.join(KINDS)}")
    p = dict(p or {})
    if kind == "data":
        name = _slug(p, "name")
        if dataset_path(home, name).exists():
            raise JobError(f"dataset {name!r} already exists; pick another name")
        sources = p.get("sources", list(DATA_SOURCES))
        if not isinstance(sources, list) or any(s not in DATA_SOURCES for s in sources):
            raise JobError(f"sources must be a subset of {', '.join(DATA_SOURCES)}")
        known = {x["name"] for x in list_problems(home)}
        probs = p.get("problems") or {}
        if not isinstance(probs, dict) or any(k not in known for k in probs):
            raise JobError(f"problems must map problem names ({', '.join(sorted(known))}) to row counts")
        imports = p.get("imports") or []
        have = {x["name"] for x in list_imports(home)}
        if not isinstance(imports, list) or any(not isinstance(x, str) or x not in have for x in imports):
            raise JobError("imports must be names of uploaded files")
        if not sources and not probs and not imports and not _int(p, "phishing_rows", 0, 0, 50_000)                 and not _int(p, "structured", 0, 0, 50_000):
            raise JobError("pick at least one source, problem, import or phishing rows")
        teacher_url = p.get("teacher_url") or None
        if teacher_url is not None and (not isinstance(teacher_url, str) or not URL.match(teacher_url)):
            raise JobError("teacher_url must be an http(s) URL, e.g. http://host:11434/v1")
        teacher_model = p.get("teacher_model") or None
        if teacher_model is not None and (not isinstance(teacher_model, str) or not re.match(r"^[\w./:-]{1,128}$", teacher_model)):
            raise JobError("teacher_model must be a model name")
        if teacher_url and not teacher_model:
            raise JobError("teacher_model is required with teacher_url")
        if probs and not teacher_url and not pipeline:
            raise JobError("problem cases are written by a teacher LLM: start the pipeline profile (the triage "
                           "model), or give teacher_url and teacher_model (any OpenAI-compatible server)")
        rows = p.get("source_rows") or {}
        if not isinstance(rows, dict) or any(k not in sources for k in rows):
            raise JobError("source_rows keys must be among the chosen sources")
        out = {"name": name, "sources": sources,
               "per_source": _int(p, "per_source", 1200, 10, 100_000),
               "source_rows": {k: _int(rows, k, 0, 10, 100_000) for k in rows},
               "phishing_rows": _int(p, "phishing_rows", 0, 0, 50_000),
               "synthetic_email": _int(p, "synthetic_email", 0, 0, 10_000),
               "subtle_share": _float(p, "subtle_share", 0.4, 0.0, 1.0),
               "synthetic_tasks": _int(p, "synthetic_tasks", 0, 0, 10_000),
               "layout_variation": _float(p, "layout_variation", 0.75, 0.0, 1.0),
               "seed": _int(p, "seed", 7, 0, 2**31 - 1),
               "structured": _int(p, "structured", 0, 0, 50_000),
               "genre_share": _float(p, "genre_share", 0.3, 0.0, 1.0),
               "problems": {k: _int(probs, k, 500, 20, 20_000) for k in probs},
               "imports": list(dict.fromkeys(imports)),
               "holdout": _float(p, "holdout", 0.1, 0.0, 0.5),
               "check": p.get("check", True) is not False,
               "teacher_url": teacher_url, "teacher_model": teacher_model}
        if (out["synthetic_email"] or out["synthetic_tasks"]) and not pipeline:
            raise JobError("synthetic rows are written by the triage LLM: start the pipeline profile "
                           "(S1_PIPELINE=1, docker compose --profile pipeline up -d) or set them to 0")
        return out
    if kind == "label":
        src = _slug(p, "dataset")
        if not dataset_path(home, src).exists():
            raise JobError(f"no dataset named {src!r}")
        name = _slug(p, "name", f"{src}-clef"[:64])
        if dataset_path(home, name).exists():
            raise JobError(f"dataset {name!r} already exists; pick another name")
        return {"dataset": src, "name": name, "alpha": _float(p, "alpha", 0.5, 0.0, 1.0),
                "batch": _int(p, "batch", 8, 1, 64), "limit": _int(p, "limit", 0, 0, 10_000_000)}
    if kind == "train":
        ds = _slug(p, "dataset")
        if not dataset_path(home, ds).exists():
            raise JobError(f"no dataset named {ds!r}")
        name = _slug(p, "name")
        if (home / "models" / name).exists():
            raise JobError(f"a model named {name!r} already exists")
        base = p.get("base", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c")
        if not isinstance(base, str) or not HF_ID.match(base):
            raise JobError("base must be a Hugging Face model id (org/name)")
        return {"dataset": ds, "name": name, "base": base,
                "lr": _float(p, "lr", 1e-5, 1e-7, 1e-3), "epochs": _float(p, "epochs", 1.0, 0.05, 20.0),
                "batch_groups": _int(p, "batch_groups", 8, 1, 128), "max_length": _int(p, "max_length", 512, 64, 4096),
                "max_pairs": _int(p, "max_pairs", 48, 4, 512), "seed": _int(p, "seed", 13, 0, 2**31 - 1)}
    if kind == "recipe":
        return _validate_recipe(p, home, pipeline=pipeline, has_jev_key=has_jev_key)
    # bench
    suites = p.get("suites") or ["general"]
    engines = p.get("engines") or ["kenning"]
    if not isinstance(suites, list) or any(not isinstance(s, str) for s in suites):
        raise JobError(f"suites must be a list of {', '.join(SUITES)} or holdout:<dataset>/<key>")
    for s in suites:
        if s not in SUITES:
            holdout_paths(home, s)  # raises unless it's a hold-out of an existing dataset
    if not isinstance(engines, list) or not engines or any(e not in ENGINES for e in engines):
        raise JobError(f"engines must be a subset of {', '.join(ENGINES)}")
    if "jev" in engines:
        if not has_jev_key:
            raise JobError("Jev needs TYPESAFE_API_KEY in .env")
        if p.get("confirm_paid") is not True:
            raise JobError("Jev requests are billed under your TypeSafe agreement: confirm_paid must be true")
    if "local" in engines and not pipeline:
        raise JobError("the 'local' engine reads the triage LLM: it needs the pipeline profile")
    return {"suites": list(dict.fromkeys(suites)), "engines": list(dict.fromkeys(engines)),
            "n": _int(p, "n", 50, 5, 1000), "repeat_check": _int(p, "repeat_check", 5, 0, 100),
            "confirm_paid": p.get("confirm_paid") is True}


def _validate_recipe(p: dict[str, Any], home: Path, *, pipeline: bool, has_jev_key: bool) -> dict[str, Any]:
    """One chained run: generate data -> (distil) -> train -> benchmark. Validates every stage up front.

    The later stages' inputs don't exist yet, so their *parameters* are checked here (ranges, names free)
    while existence checks that only make sense after the data stage are deferred to run time.
    """
    name = _slug(p, "name")
    if (home / "models" / name).exists():
        raise JobError(f"a model named {name!r} already exists")
    dataset, labelled = f"{name}-data", f"{name}-data-clef"
    # data stage: reuse the data validator with the derived dataset name (all its checks apply)
    data = validate("data", {**(p.get("data") or {}), "name": dataset}, home, pipeline=pipeline, has_jev_key=has_jev_key)
    # train stage: validate parameters only (the dataset is produced by stage 1)
    tr = p.get("train") or {}
    base = tr.get("base", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c")
    if not isinstance(base, str) or not HF_ID.match(base):
        raise JobError("train.base must be a Hugging Face model id (org/name)")
    train = {"base": base, "lr": _float(tr, "lr", 1e-5, 1e-7, 1e-3), "epochs": _float(tr, "epochs", 2.0, 0.05, 20.0),
             "batch_groups": _int(tr, "batch_groups", 8, 1, 128), "max_length": _int(tr, "max_length", 1024, 64, 4096),
             "max_pairs": _int(tr, "max_pairs", 48, 4, 512), "seed": _int(tr, "seed", 13, 0, 2**31 - 1)}
    # bench stage: built-in suites only (the dataset's own hold-outs don't exist until stage 1 runs)
    bn = p.get("bench") or {}
    suites = bn.get("suites") or ["general"]
    if not isinstance(suites, list) or any(x not in SUITES for x in suites):
        raise JobError(f"recipe bench suites must be a subset of {', '.join(SUITES)}")
    engines = bn.get("engines") or ["kenning", "clef"]
    if not isinstance(engines, list) or not engines or any(e not in ENGINES for e in engines):
        raise JobError(f"engines must be a subset of {', '.join(ENGINES)}")
    if "jev" in engines:
        if not has_jev_key:
            raise JobError("Jev needs TYPESAFE_API_KEY in .env")
        if bn.get("confirm_paid") is not True:
            raise JobError("Jev requests are billed under your TypeSafe agreement: confirm_paid must be true")
    if "local" in engines and not pipeline:
        raise JobError("the 'local' engine reads the triage LLM: it needs the pipeline profile")
    bench = {"suites": list(dict.fromkeys(suites)), "engines": list(dict.fromkeys(engines)),
             "n": _int(bn, "n", 50, 5, 1000), "repeat_check": _int(bn, "repeat_check", 5, 0, 100),
             "confirm_paid": bn.get("confirm_paid") is True}
    return {"name": name, "model": name, "dataset": dataset, "labelled": labelled,
            "distill": p.get("distill", True) is not False, "alpha": _float(p, "alpha", 0.5, 0.0, 1.0),
            "data": data, "train": train, "bench": bench}


def commands(kind: str, p: dict[str, Any], home: Path, clef_url: str) -> list[list[str]]:
    """Argument lists for the subprocess jobs (data, label, bench)."""
    cli = [sys.executable, "-m", "systemone_builder.cli"]
    if kind == "data":
        argv = cli + ["data", "--task", "multitask", "--out", str(dataset_path(home, p["name"])),
                      "--sources", ",".join(p["sources"]) or "none", "--per-source", str(p["per_source"]),
                      "--phishing-rows", str(p["phishing_rows"]), "--layout-variation", str(p["layout_variation"]),
                      "--seed", str(p["seed"])]
        if p["source_rows"]:
            argv += ["--source-rows", ",".join(f"{k}={v}" for k, v in p["source_rows"].items())]
        if p["synthetic_email"]:
            argv += ["--synthetic-email", str(p["synthetic_email"]), "--subtle-share", str(p["subtle_share"])]
        if p["synthetic_tasks"]:
            argv += ["--synthetic-tasks", str(p["synthetic_tasks"])]
        if p.get("structured"):
            argv += ["--structured", str(p["structured"])]
        if "genre_share" in p:
            argv += ["--genre-share", str(p["genre_share"])]
        for name, rows in p.get("problems", {}).items():
            argv += ["--problem", f"{name}={rows}"]
        for name in p.get("imports", []):
            meta = imports_dir(home) / f"{name}.meta.json"
            licence = json.loads(meta.read_text(encoding="utf-8")).get("license") if meta.exists() else None
            argv += ["--import", str(imports_dir(home) / f"{name}.jsonl"),
                     "--import-licence", licence or "provided by the user"]
        if p.get("problems") or p.get("imports"):
            argv += ["--holdout", str(p["holdout"])]
        if p.get("teacher_url"):
            argv += ["--teacher-url", p["teacher_url"], "--teacher-model", p["teacher_model"]]
        if p.get("problems") and not p.get("check", True):
            argv.append("--no-check")
        return [argv]
    if kind == "label":
        argv = cli + ["label", str(dataset_path(home, p["dataset"])), "--out", str(dataset_path(home, p["name"])),
                      "--teacher-url", clef_url, "--teacher-name", "clef-flash",
                      "--alpha", str(p["alpha"]), "--batch", str(p["batch"])]
        if p["limit"]:
            argv += ["--limit", str(p["limit"])]
        return [argv]
    if kind == "bench":
        out = []
        for suite in p["suites"]:
            common = ["--engines", ",".join(p["engines"]), "--concurrency", "1", "--repeat-check", str(p["repeat_check"])]
            if suite in SUITES:
                n = 60 if suite == "ood" else 50 if suite in ("multi", "general") else p["n"]
                out.append(cli + ["bench", "--suite", suite, "-n", str(n)] + common)
            else:
                items, questions = holdout_paths(home, suite)
                out.append(cli + ["bench", "--suite", str(items), "--questions", str(questions)] + common)
        return out
    raise JobError(f"{kind} does not run as a subprocess")


# ------------------------------------------------------------------ runner
class KenningJobs:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        self.s = rt.settings
        self._task: asyncio.Task[None] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._current: dict[str, Any] | None = None
        self._cancel = False

    # ---- storage
    @property
    def home(self) -> Path:
        return self.s.kenning_dir()

    @property
    def dir(self) -> Path:
        return self.home / "jobs"

    def _save(self, job: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{job['id']}.json.part"
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(self.dir / f"{job['id']}.json")

    def _log(self, job: dict[str, Any], line: str) -> None:
        with (self.dir / f"{job['id']}.log").open("a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")
        m = STEP.search(line) or ROWS.search(line)
        if m:
            job["progress"] = {"done": int(m.group(1)), "total": int(m.group(2))}
        if "full results:" in line:
            job.setdefault("result", {}).setdefault("files", []).append(line.split("full results:", 1)[1].strip())

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        jobs = []
        for f in self.dir.glob("*.json"):
            try:
                jobs.append(json.loads(f.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda j: j.get("created", 0), reverse=True)[:limit]

    def get(self, job_id: str, tail: int = 200) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{12}", job_id or ""):
            raise JobError("invalid job id")
        f = self.dir / f"{job_id}.json"
        if not f.exists():
            raise KeyError(job_id)
        job = self._current if self._current and self._current["id"] == job_id else json.loads(f.read_text())
        logf = self.dir / f"{job_id}.log"
        lines = logf.read_text(encoding="utf-8", errors="replace").splitlines() if logf.exists() else []
        return {**job, "log": lines[-tail:]}

    def reconcile(self) -> None:
        """At API start: jobs that were running when it stopped are interrupted."""
        for job in self.list(1000):
            if job.get("status") in ("queued", "running"):
                job.update(status="interrupted", finished=time.time(),
                           error="the API restarted while this job was running")
                self._save(job)

    # ---- control
    def start(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.busy:
            raise JobError(f"another job is running ({self._current['kind']} {self._current['id']}); wait or cancel it")
        if getattr(self.rt, "lifecycle", None) is not None and self.rt.lifecycle.busy:
            raise JobError("a pipeline training cycle holds the GPU; wait for it to finish")
        p = validate(kind, params, self.home, pipeline=self.s.pipeline, has_jev_key=bool(self.s.typesafe_api_key))
        job = {"id": uuid.uuid4().hex[:12], "kind": kind, "params": p, "status": "queued", "created": time.time(),
               "started": None, "finished": None, "progress": None, "result": {}, "error": None, "paused": []}
        self._save(job)
        self._current, self._cancel = job, False
        self._task = asyncio.create_task(self._run(job))
        return job

    async def cancel(self, job_id: str) -> dict[str, Any]:
        if not self._current or self._current["id"] != job_id or not self.busy:
            raise JobError("that job is not running")
        self._cancel = True
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
        if self._current["kind"] == "train":
            await self.rt.docker.stop(self.s.kenning_train_container, timeout_s=10)
        return self._current

    async def _run(self, job: dict[str, Any]) -> None:
        job.update(status="running", started=time.time())
        self._save(job)
        try:
            if job["kind"] == "recipe":
                await self._recipe(job)
            elif job["kind"] == "train":
                await self._train(job)
            elif job["kind"] == "label":
                await self._label(job)
            else:
                for argv in commands(job["kind"], job["params"], self.home, self.s.clef_url):
                    await self._subprocess(job, argv)
            job["status"] = "cancelled" if self._cancel else "succeeded"
        except asyncio.CancelledError:
            job.update(status="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - every failure ends up on the job
            job.update(status="cancelled" if self._cancel else "failed", error=str(exc))
            self._log(job, f"[job] {'cancelled' if self._cancel else 'failed'}: {exc}")
            if not self._cancel:
                log.exception("kenning job %s failed", job["id"])
        finally:
            job["finished"] = time.time()
            self._save(job)
            self._proc = None
            self._current = None if self._current is job else self._current

    async def _subprocess(self, job: dict[str, Any], argv: list[str]) -> None:
        if self._cancel:
            raise JobError("cancelled")
        self._log(job, "[job] $ " + " ".join(argv[1:]))
        self._proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        assert self._proc.stdout is not None
        last_save = 0.0
        async for raw in self._proc.stdout:
            self._log(job, raw.decode(errors="replace"))
            if time.monotonic() - last_save > 2:
                self._save(job)
                last_save = time.monotonic()
        code = await self._proc.wait()
        if code != 0:
            raise JobError("cancelled" if self._cancel else f"{argv[3]} exited with code {code}")

    # ---- GPU
    @asynccontextmanager
    async def _gpu(self, job: dict[str, Any], gpu: int, keep: str | None = None,
                   free_mb: int | None = None) -> AsyncIterator[None]:
        """Pause our services on ``gpu`` (except ``keep``), restart them afterwards."""
        on_gpu = {self.s.kenning_container: self.s.kenning_gpu, self.s.clef_container: self.s.clef_gpu}
        if self.s.pipeline:
            on_gpu[self.s.student_container] = self.s.student_gpu
        paused: list[str] = []
        try:
            for name, g in on_gpu.items():
                if g != gpu or name == keep:
                    continue
                info = await self.rt.docker.status(name)
                if info.status in ("running", "restarting"):
                    self._log(job, f"[job] pausing {name} to free GPU {gpu}")
                    await self.rt.docker.stop(name, timeout_s=60)
                    paused.append(name)
            job["paused"] = paused
            self._save(job)
            if free_mb:
                status = await self.rt.gpu.wait_for_flush(gpu, self.s.vram_flush_threshold_mb, 180.0,
                                                          required_free_mb=free_mb)
                self._log(job, f"[job] GPU {gpu}: {status.memory_free_mb} MiB free")
            yield
        finally:
            for name in paused:
                try:
                    self._log(job, f"[job] restarting {name}")
                    await self.rt.docker.start(name)
                except Exception as exc:  # noqa: BLE001
                    self._log(job, f"[job] could not restart {name}: {exc} (start it with docker compose up -d)")

    async def _clef_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                (await c.get(f"{self.s.clef_url.rstrip('/')}/health")).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def _label(self, job: dict[str, Any], params: dict[str, Any] | None = None) -> None:
        params = params or job["params"]
        if not await self._clef_healthy():
            raise JobError("Clef is not running: docker compose --profile clef up -d clef "
                           "(it needs most of a 24 GB GPU)")
        # Clef labelling in batches fills the GPU: pause the other services on it.
        async with self._gpu(job, self.s.clef_gpu, keep=self.s.clef_container):
            for argv in commands("label", params, self.home, self.s.clef_url):
                await self._subprocess(job, argv)

    async def _train(self, job: dict[str, Any], params: dict[str, Any] | None = None) -> None:
        p, s = params or job["params"], self.s
        ws = s.workspace_container_path
        out = f"{ws}/kenning/models/{p['name']}"
        command = ["python", "-m", "systemone_builder.kenning.train",
                   "--data", f"{ws}/kenning/datasets/{p['dataset']}.jsonl", "--out", out, "--base", p["base"],
                   "--lr", str(p["lr"]), "--epochs", str(p["epochs"]), "--batch-groups", str(p["batch_groups"]),
                   "--max-length", str(p["max_length"]), "--max-pairs", str(p["max_pairs"]), "--seed", str(p["seed"])]
        env = {"PYTHONUNBUFFERED": "1", "HF_HOME": f"{ws}/hf_cache", "KENNING_HOME": f"{ws}/kenning",
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "NVIDIA_VISIBLE_DEVICES": str(s.kenning_gpu),
               "CUDA_VISIBLE_DEVICES": s.kenning_cuda_device or str(s.kenning_gpu)}
        token = _env("HF_TOKEN")
        if token:
            env["HF_TOKEN"] = token
        spec = RunSpec(name=s.kenning_train_container, image=s.kenning_image, command=command, gpu=s.kenning_gpu,
                       environment=env, volumes={s.workspace_host_path: {"bind": ws, "mode": "rw"}},
                       shm_size="8g", mem_limit=s.kenning_train_mem_limit)
        loop = asyncio.get_running_loop()

        def on_log(line: str) -> None:
            loop.call_soon_threadsafe(self._log, job, line)

        async with self._gpu(job, s.kenning_gpu, free_mb=s.kenning_train_free_mb):
            if self._cancel:
                raise JobError("cancelled")
            self._log(job, "[job] $ " + " ".join(command[2:]))
            saver = asyncio.create_task(self._autosave(job))
            try:
                code = await self.rt.docker.run_to_completion(spec, on_log=on_log, timeout_s=12 * 3600)
            finally:
                saver.cancel()
            await asyncio.sleep(0.1)
        if code != 0:
            raise JobError("cancelled" if self._cancel else f"training exited with code {code}")
        summary = registry.summary(self.home, p["name"])
        job["result"] = {"model": p["name"], "heldout": summary["heldout"]}

    async def _recipe(self, job: dict[str, Any]) -> None:
        """Generate -> distil (optional) -> train -> activate + benchmark, as one job."""
        rp = job["params"]
        self._log(job, f"[recipe] stage 1/4: generate dataset {rp['dataset']}")
        for argv in commands("data", rp["data"], self.home, self.s.clef_url):
            await self._subprocess(job, argv)
        train_dataset = rp["dataset"]
        if rp["distill"] and await self._clef_healthy():
            self._log(job, f"[recipe] stage 2/4: distil labels with Clef -> {rp['labelled']}")
            await self._label(job, {"dataset": rp["dataset"], "name": rp["labelled"], "alpha": rp["alpha"],
                                    "batch": 16, "limit": 0})
            train_dataset = rp["labelled"]
        else:
            why = "distillation off" if not rp["distill"] else "Clef not running"
            self._log(job, f"[recipe] stage 2/4: {why}; training on base labels")
        self._log(job, f"[recipe] stage 3/4: train {rp['model']} on {train_dataset}")
        await self._train(job, {**rp["train"], "dataset": train_dataset, "name": rp["model"]})
        self._log(job, f"[recipe] stage 4/4: activate {rp['model']} and benchmark")
        await self._activate_and_bench(job, rp)

    async def _activate_and_bench(self, job: dict[str, Any], rp: dict[str, Any]) -> None:
        url = self.s.kenning_url.rstrip("/")
        async with httpx.AsyncClient(timeout=300) as c:
            for _ in range(60):  # kenning was restarted after training freed its GPU
                try:
                    (await c.get(f"{url}/health")).raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(5)
            r = await c.post(f"{url}/admin/load", json={"model": rp["model"]})
            if r.status_code != 200:
                raise JobError(f"could not serve {rp['model']} for benchmarking: {r.text[:200]}")
        registry.set_active(self.home, rp["model"])
        self._log(job, f"[recipe] serving {rp['model']}")
        for argv in commands("bench", rp["bench"], self.home, self.s.clef_url):
            await self._subprocess(job, argv)

    async def _autosave(self, job: dict[str, Any]) -> None:
        while True:
            await asyncio.sleep(3)
            self._save(job)


def _env(name: str) -> str | None:
    import os

    return os.environ.get(name) or None
