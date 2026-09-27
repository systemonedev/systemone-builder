"""Reference computer-use client: drive a browser with the System-1 reflex model.

    pip install playwright httpx && playwright install chromium
    python playwright_agent.py https://example.internal/login "Sign in as admin" --api http://linux-host:8000

Loop per step:
  1. collect interactive elements (+ bounding boxes) with the collector JS
  2. POST /api/v1/act/computer_use            -> action + execution handle
  3. execute: selector / bbox centre / CLICK_XY coordinates
  4. POST /api/v1/feedback with the post-action observation, so failures
     (error banners, no-ops) feed the state-delta DPO loop
"""

from __future__ import annotations

import argparse
import asyncio
import os

import httpx
from playwright.async_api import Page, async_playwright


async def observe(page: Page, collector: str, goal: str) -> dict:
    elements = await page.evaluate(collector)
    vp = page.viewport_size or {"width": 1280, "height": 800}
    return {"kind": "elements", "data": elements, "url": page.url, "goal": goal, "viewport": [vp["width"], vp["height"]]}


async def execute(page: Page, decision: dict) -> str | None:
    a, h = decision["action"], decision.get("execution") or {}
    try:
        if a["action"] == "CLICK":
            if h.get("selector"):
                await page.click(h["selector"], timeout=3000)
            else:
                await page.mouse.click(h["center"]["x"], h["center"]["y"])
        elif a["action"] == "TYPE":
            text = a.get("text") or ""
            if text == "<SECRET>":
                text = os.environ.get("S1_AGENT_SECRET", "")
            if h.get("selector"):
                await page.fill(h["selector"], text, timeout=3000)
            else:
                await page.mouse.click(h["center"]["x"], h["center"]["y"])
                await page.keyboard.type(text)
        elif a["action"] == "CLICK_XY":
            await page.mouse.click(a["coordinates"]["x"], a["coordinates"]["y"])
        await page.wait_for_load_state("networkidle", timeout=5000)
        return None
    except Exception as exc:  # reported as an execution error -> DPO candidate
        return repr(exc)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("goal")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--session", default="playwright-demo")
    args = ap.parse_args()
    headers = {"X-API-Key": os.environ["S1_API_KEY"]} if os.environ.get("S1_API_KEY") else {}
    async with httpx.AsyncClient(base_url=f"{args.api}/api/v1", headers=headers, timeout=120) as api, async_playwright() as pw:
        collector = (await api.get("/extract/collector.js")).text
        await api.delete(f"/sessions/computer_use/{args.session}")
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.goto(args.url)
        for step in range(args.max_steps):
            obs = await observe(page, collector, args.goal)
            d = (await api.post("/act/computer_use", json={"observation": obs, "session_id": args.session})).json()
            label = d["action"]["action"]
            print(f"step {step}: {label} tier={d['tier']} conf={d['confidence']:.2f} {d['latency_ms']:.0f}ms")
            if d["halted"] or label == "ESCALATE":
                print(f"escalated to the oracle (job {d.get('escalation_id')}) - waiting for the System-2 answer")
                job = (await api.get(f"/routing/escalations/{d['escalation_id']}", params={"wait_s": 300})).json()
                if job.get("status") != "done":
                    print("oracle could not resolve; stopping")
                    break
                d["action"] = job["action"]
                d["execution"] = None
            err = await execute(page, d)
            post = await observe(page, collector, args.goal)
            fb = {"seq": d["seq"], "post_observation": post}
            if err:
                fb.update(outcome="failure", error_message=err)
            res = (await api.post("/feedback", json=fb)).json()
            print(f"  outcome={res['outcome']} {res['delta'].get('error_signals') if res.get('delta') else ''}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
