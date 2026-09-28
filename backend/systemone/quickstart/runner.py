"""Zero-to-One Quickstart: from ``docker compose up`` to a training loop.

Runs as the one-shot ``quickstart`` compose service (idempotent):

1. wait for the API and the vLLM student (tiny 1.5B model) to be healthy
2. pull the dummy web-automation dataset (``S1_QUICKSTART_DATASET_URL``) or
   generate it locally, and load it as SFT + unseen held-out samples
3. run the first strict train -> hot-reload cycle on GPU 0
4. benchmark the fine-tuned student in the evaluation sandbox
5. print where to go next (dashboard, playground)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import httpx

from systemone.quickstart.webgen import generate

DOMAIN = "computer_use"


class Client:
    def __init__(self, base: str, api_key: str | None) -> None:
        self.c = httpx.Client(base_url=base.rstrip("/") + "/api/v1", timeout=60,
                              headers={"X-API-Key": api_key} if api_key else {})

    def get(self, path: str, **kw: Any) -> Any:
        r = self.c.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: Any = None) -> Any:
        r = self.c.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]}")
        return r.json()


def say(msg: str) -> None:
    print(f"[quickstart {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for(fn, what: str, timeout_s: float, every: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            res = fn()
            if res:
                return res
        except Exception as exc:  # service still booting
            last_err = exc
        time.sleep(every)
    raise TimeoutError(f"timed out waiting for {what} ({last_err})")


def load_dataset(n_train: int, n_heldout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    url = os.environ.get("S1_QUICKSTART_DATASET_URL")
    if url:
        say(f"pulling dummy web-automation dataset from {url}")
        rows = [json.loads(line) for line in httpx.get(url, timeout=120, follow_redirects=True).text.splitlines() if line.strip()]
        split = max(len(rows) - n_heldout, 1)
        return rows[:split], rows[split:]
    say("generating dummy web-automation dataset (procedural, seeded)")
    train = generate(n_train, seed=7)
    seen = {json.dumps(r["state"], sort_keys=True) for r in train}
    held = [r for r in generate(n_heldout * 3, seed=1234) if json.dumps(r["state"], sort_keys=True) not in seen][:n_heldout]
    return train, held


def run(api: str, api_key: str | None = None, n_train: int = 600, n_heldout: int = 50, train: bool = True,
        student_timeout_s: float = 1800) -> int:
    c = Client(api, api_key)
    wait_for(lambda: c.get("/health").get("redis"), "API + Redis", 600)
    say("API is up")
    wait_for(lambda: c.get("/health/deep")["student"].get("ok"), "vLLM student (first start downloads the 1.5B model)", student_timeout_s, 10)
    say("student vLLM is serving")

    # Counters live in Redis, samples on the workspace volume: make sure they agree
    # before deciding the dataset is already loaded.
    c.post(f"/datasets/{DOMAIN}/reconcile")
    stats = c.get(f"/datasets/{DOMAIN}")
    if stats["sft"] >= n_train and stats["heldout"] >= n_heldout:
        say(f"dataset already loaded ({stats['sft']} SFT / {stats['heldout']} held-out) - skipping")
    else:
        tr, ho = load_dataset(n_train, n_heldout)
        added = 0
        for i in range(0, len(tr), 200):
            added += c.post(f"/datasets/{DOMAIN}/sft/bulk", [{"state": r["state"], "action": r["action"]} for r in tr[i:i + 200]])["added"]
        res = c.post(f"/eval/{DOMAIN}/heldout", [{"state": r["state"], "expected": r["action"], "tags": [r.get("scenario", "dummy").split(":")[0]]} for r in ho])
        say(f"loaded {added} SFT samples and {res['added']} unseen held-out samples")

    runs = [r for r in c.get("/training/runs") if r["domain"] == DOMAIN and r["status"] == "succeeded"]
    if train and not runs:
        cfg = c.post("/training/run", {"domain": DOMAIN, "mode": "sft"})
        run_id = cfg["run_id"]
        say(f"training cycle {run_id}: pausing vLLM, flushing GPU 0, Unsloth QLoRA on {cfg['rows']} samples ...")
        last_phase, last_step = None, -1
        while True:
            st = c.get("/training/status")
            if st["phase"] != last_phase:
                say(f"lifecycle -> {st['phase']}")
                last_phase = st["phase"]
            m = c.get(f"/training/runs/{run_id}/metrics")
            if m and m[-1].get("step", -1) != last_step:
                last_step = m[-1]["step"]
                say(f"step {last_step}/{m[-1].get('max_steps')} loss {m[-1].get('loss', float('nan')):.4f}")
            run_rec = next((r for r in c.get("/training/runs") if r["run_id"] == run_id), None)
            if run_rec and run_rec["status"] in ("succeeded", "failed"):
                break
            time.sleep(5)
        if run_rec["status"] != "succeeded":
            say(f"training failed: {run_rec.get('error')}")
            return 1
        say(f"hot-reloaded fine-tuned student (loss {run_rec['result']['train_loss']:.4f})")
    elif runs:
        say(f"student already fine-tuned ({runs[0]['run_id']}) - skipping training")

    rid = c.post("/eval/run", {"domain": DOMAIN, "target": "student", "gates": {"min_samples": min(30, n_heldout)}})["report_id"]
    say(f"benchmarking on held-out data ({rid})")
    rep = wait_for(lambda: (r := c.get(f"/eval/reports/{rid}")).get("status") in ("done", "failed") and r, "evaluation", 1800, 3)
    if rep.get("status") == "done":
        m = rep["metrics"]
        say(f"accuracy {m['accuracy']:.1%} · success {m['success_rate'] or 0:.1%} · hallucination {m['hallucination_ratio']:.1%} · "
            f"p95 latency {rep['latency_ms'].get('p95') or 0:.0f} ms · ready={rep['readiness']['ready_for_deployment']}")
    host = os.environ.get("S1_PUBLIC_HOST", "localhost")
    say(f"done. Dashboard: http://{host}:3000  ·  Playground: http://{host}:3000/playground  ·  API docs: http://{host}:8000/docs")
    return 0


if __name__ == "__main__":
    sys.exit(run(os.environ.get("S1_API", "http://localhost:8000"), os.environ.get("S1_API_KEY")))
