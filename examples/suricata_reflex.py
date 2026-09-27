"""Reference SecOps client: sub-100ms reflex triage of Suricata EVE events.

    python suricata_reflex.py /var/log/suricata/eve.json --api http://linux-host:8000 [--enforce]

Tails eve.json, sends alert/http events to /api/v1/act/secops and applies
DROP_AND_BLACKLIST_IP decisions with nftables. Without --enforce it only
prints the command (dry run). Escalations are left to the oracle; follow-up
events from the same source are reported as feedback so ALLOW decisions that
were followed by a successful attack become DPO corrections.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import time

import httpx

NFT_SET = "inet filter s1_blacklist"


def tail(path: str):
    with open(path) as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            yield line


def block(ip: str, enforce: bool) -> None:
    ipaddress.ip_address(ip)  # never pass unvalidated input to a shell
    cmd = ["nft", "add", "element", *NFT_SET.split(), "{", ip, "}"]
    print(("  applying: " if enforce else "  dry-run: ") + " ".join(cmd))
    if enforce:
        subprocess.run(cmd, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eve")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--allowlist", default="10.0.0.0/8", help="never block these networks")
    args = ap.parse_args()
    allow = [ipaddress.ip_network(n) for n in args.allowlist.split(",")]
    headers = {"X-API-Key": os.environ["S1_API_KEY"]} if os.environ.get("S1_API_KEY") else {}
    api = httpx.Client(base_url=f"{args.api}/api/v1", headers=headers, timeout=10)
    last_allow: dict[str, int] = {}  # src_ip -> seq of the last ALLOW decision
    for line in tail(args.eve):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_type") not in ("alert", "http", "dns", "tls"):
            continue
        src = ev.get("src_ip")
        if src in last_allow:  # follow-up from a source we allowed: report the delta
            api.post("/feedback", json={"seq": last_allow.pop(src), "post_observation": {"kind": "suricata_eve", "data": ev}})
        d = api.post("/act/secops", json={"observation": {"kind": "suricata_eve", "data": ev}, "session_id": "suricata"}).json()
        a = d["action"]
        print(f"{src} -> {a['verdict']}/{a['immediate_action']} tier={d['tier']} conf={d['confidence']:.2f} {d['latency_ms']:.0f}ms")
        if a["immediate_action"] == "DROP_AND_BLACKLIST_IP":
            for ioc in a.get("target_ioc", []):
                if any(ipaddress.ip_address(ioc) in n for n in allow):
                    print(f"  refusing to block allow-listed {ioc}")
                    api.post("/feedback", json={"seq": d["seq"], "outcome": "failure", "note": f"{ioc} is allow-listed"})
                    continue
                block(ioc, args.enforce)
        elif a["immediate_action"] == "ALLOW" and src:
            last_allow[src] = d["seq"]


if __name__ == "__main__":
    main()
