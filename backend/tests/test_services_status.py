from __future__ import annotations

from systemone_builder.orchestrator.services import summarize_logs

LOADING = """systemone-student  | [serve_student] exec: vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0
Loading safetensors checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  33% Completed | 1/3 [17:35<35:10, 1055.42s/it]
"""

CRASH = """(EngineCore pid=237) INFO 09-27 15:06:32 [shared_offload_region.py:71] Created mmap file /dev/shm/x.mmap (34.36 GB)
(EngineCore pid=237) ERROR 09-27 15:06:32 [core.py:1349] EngineCore failed to start.
(EngineCore pid=237) ERROR 09-27 15:06:32 [core.py:1349] OSError: [Errno 14] Bad address
(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def test_progress_from_loading_log():
    progress, err = summarize_logs(LOADING)
    assert progress == "loading weights 1/3 shards (33%)" and err is None


def test_error_from_crash_log():
    progress, err = summarize_logs(CRASH)
    assert "Engine core initialization failed" in err


def test_compile_and_argparse_errors():
    assert summarize_logs("INFO [monitor.py:53] torch.compile took 13.44 s in total")[0] == "compiled in 13s"
    _, err = summarize_logs("usage: vllm [-h]\nvllm: error: unrecognized arguments: --swap-space=16")
    assert "unrecognized arguments" in err


async def test_kenning_service_status_online_and_offline(monkeypatch):
    import httpx

    from systemone_builder.config import Settings
    from systemone_builder.orchestrator import services

    class Rt:
        settings = Settings(_env_file=None, kenning_url="http://kenning:8000")

    real = httpx.AsyncClient

    def client(answer):
        return lambda **kw: real(transport=httpx.MockTransport(answer), **kw)

    monitor = services.ServiceMonitor.__new__(services.ServiceMonitor)
    monitor.rt = Rt()
    monkeypatch.setattr(httpx, "AsyncClient", client(lambda r: httpx.Response(200, json={"model": "kenning-large-v0.1"})))
    up = await monitor._kenning()
    assert up["state"] == "online" and up["model"] == "kenning-large-v0.1"

    def refuse(r):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", client(refuse))
    down = await monitor._kenning()
    assert down["state"] == "offline"
