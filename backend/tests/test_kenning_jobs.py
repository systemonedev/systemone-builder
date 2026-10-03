from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from systemone_builder.config import Settings
from systemone_builder.kenning import jobs as kj
from systemone_builder.orchestrator.docker_ctl import ContainerInfo


def _dataset(home, name, rows=3):
    d = home / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text("{}\n" * rows)
    (d / f"{name}.manifest.json").write_text(json.dumps({"sources": {"nli": {"rows": rows, "license": "OANC"}}}))


def _validate(kind, params, home, **kw):
    return kj.validate(kind, params, home, pipeline=kw.get("pipeline", False), has_jev_key=kw.get("jev", False))


def test_validation_rejects_unsafe_or_unknown_input(tmp_path):
    _dataset(tmp_path, "clean-v3")
    for kind, params, msg in [
        ("shell", {}, "kind"),
        ("data", {"name": "../etc"}, "lower-case"),
        ("data", {"name": "x; rm -rf /"}, "lower-case"),
        ("data", {"name": "new", "sources": ["amazon", "evil"]}, "subset"),
        ("data", {"name": "new", "per_source": 10**9}, "per_source"),
        ("data", {"name": "new", "synthetic_email": 100}, "triage LLM"),
        ("data", {"name": "clean-v3"}, "already exists"),
        ("label", {"dataset": "missing"}, "no dataset"),
        ("train", {"dataset": "clean-v3", "name": "m", "base": "--help"}, "Hugging Face"),
        ("train", {"dataset": "clean-v3", "name": "m", "lr": "1e-5"}, "lr"),
        ("bench", {"suites": ["modern2", "/etc/passwd"]}, "suites"),
        ("bench", {"engines": ["jev"]}, "TYPESAFE_API_KEY"),
        ("bench", {"engines": ["local"]}, "pipeline"),
    ]:
        with pytest.raises(kj.JobError, match=msg):
            _validate(kind, params, tmp_path)
    with pytest.raises(kj.JobError, match="confirm_paid"):
        _validate("bench", {"engines": ["kenning", "jev"]}, tmp_path, jev=True)
    assert _validate("bench", {"engines": ["kenning", "jev"], "confirm_paid": True}, tmp_path, jev=True)["engines"] == ["kenning", "jev"]


def test_commands_are_argument_lists(tmp_path):
    p = _validate("data", {"name": "mine", "sources": ["nli", "clinc"], "source_rows": {"nli": 2000}}, tmp_path)
    (argv,) = kj.commands("data", p, tmp_path, "http://clef:8000")
    assert argv[:4] == [sys.executable, "-m", "systemone_builder.cli", "data"]
    assert argv[argv.index("--out") + 1] == str(tmp_path / "datasets" / "mine.jsonl")
    assert "nli=2000" in argv and "--synthetic-email" not in argv
    _dataset(tmp_path, "mine")
    lp = _validate("label", {"dataset": "mine", "alpha": 1.0}, tmp_path)
    assert lp["name"] == "mine-clef"
    (argv,) = kj.commands("label", lp, tmp_path, "http://clef:8000")
    assert argv[argv.index("--teacher-url") + 1] == "http://clef:8000" and argv[argv.index("--alpha") + 1] == "1.0"
    bench = kj.commands("bench", _validate("bench", {"suites": ["modern2", "ood"], "n": 20}, tmp_path), tmp_path, "")
    assert [a[a.index("--suite") + 1] for a in bench] == ["modern2", "ood"]
    assert bench[1][bench[1].index("-n") + 1] == "60"  # ood is sized per task


def test_datasets_list_skips_caches_without_manifest(tmp_path):
    _dataset(tmp_path, "clean-v3", rows=5)
    (tmp_path / "datasets" / "synthetic-email-cache.jsonl").write_text("{}\n")
    (listed,) = kj.list_datasets(tmp_path)
    assert listed["name"] == "clean-v3" and listed["rows"] == 5 and listed["sources"]["nli"]["license"] == "OANC"


class FakeDocker:
    def __init__(self, running):
        self.running, self.calls = set(running), []

    async def status(self, name):
        return ContainerInfo(name=name, status="running" if name in self.running else "exited")

    async def stop(self, name, timeout_s=30):
        self.calls.append(("stop", name))
        self.running.discard(name)
        return await self.status(name)

    async def start(self, name):
        self.calls.append(("start", name))
        self.running.add(name)
        return await self.status(name)


def _runtime(tmp_path, running=("systemone-kenning",)):
    s = Settings(_env_file=None, kenning_home=tmp_path, pipeline=False)
    gpu = SimpleNamespace(wait_for_flush=None)
    return SimpleNamespace(settings=s, docker=FakeDocker(running), gpu=gpu, lifecycle=SimpleNamespace(busy=False))


async def test_subprocess_job_logs_progress_and_one_job_at_a_time(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    runner = kj.KenningJobs(rt)
    script = "import time\nfor i in (1, 2): print(f'[kenning-label] {i*8}/16 rows', flush=True); time.sleep(0.05)"
    monkeypatch.setattr(kj, "commands", lambda *a: [[sys.executable, "-c", script, "bench"]])
    job = runner.start("bench", {"suites": ["modern2"]})
    with pytest.raises(kj.JobError, match="another job"):
        runner.start("bench", {})
    await runner._task
    done = runner.get(job["id"])
    assert done["status"] == "succeeded" and done["progress"] == {"done": 16, "total": 16}
    assert "[kenning-label] 16/16 rows" in done["log"]
    assert rt.docker.calls == []  # benchmarks pause nothing


async def test_gpu_jobs_restart_paused_services_even_on_failure(tmp_path):
    rt = _runtime(tmp_path, running=("systemone-kenning", "systemone-clef"))
    runner = kj.KenningJobs(rt)
    job = {"id": "abcdefabcdef"}
    runner.dir.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        async with runner._gpu(job, 0, keep="systemone-clef"):
            assert rt.docker.running == {"systemone-clef"}
            raise RuntimeError("boom")
    assert rt.docker.calls == [("stop", "systemone-kenning"), ("start", "systemone-kenning")]


def test_interrupted_jobs_are_marked_at_startup(tmp_path):
    runner = kj.KenningJobs(_runtime(tmp_path))
    runner._save({"id": "0123456789ab", "kind": "train", "status": "running", "created": 1})
    runner.reconcile()
    assert runner.get("0123456789ab")["status"] == "interrupted"
    with pytest.raises(kj.JobError):
        runner.get("../../etc/passwd")


def test_container_status_survives_a_replaced_image():
    pytest.importorskip("docker")
    from systemone_builder.orchestrator.docker_ctl import DockerController

    class Replaced:
        name, status = "systemone-student", "running"
        attrs = {"Config": {"Image": "systemone/student:latest"}, "State": {"ExitCode": 0},
                 "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["0"]}]}}

        @property
        def image(self):  # what docker-py does once a rebuild removed the image: 404
            raise RuntimeError("No such image: sha256:0da8")

    info = DockerController(client=object())._info(Replaced())
    assert info.status == "running" and info.image == "systemone/student:latest" and info.gpus == [0]
