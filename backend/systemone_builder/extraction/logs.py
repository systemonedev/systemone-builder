"""Log pre-processors for the Cybersecurity (SecOps) domain.

Supported sources:

* ``suricata_eve`` - Suricata EVE JSON (alert / http / dns / tls / flow ...)
* ``syslog``       - RFC3164 / auth.log style lines (sshd, sudo, nginx ...)
* ``json``         - arbitrary JSON log objects (generic fallback)

Each produces a ``state_input`` matching :class:`SecOpsState`. Volatile
per-event metadata (timestamps, flow ids, packet counters, ephemeral source
ports) is dropped so that repeated attacks against the same service render to
identical prompt prefixes.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from systemone_builder.extraction.fuzzy import IP_RE, EntityVault, FuzzyScrubber, ScrubPolicy

EVE_DROP_FIELDS = {
    "timestamp", "flow_id", "pcap_cnt", "tx_id", "in_iface", "community_id", "src_port", "parent_id",
    "packet", "packet_info", "stream", "capture_file", "pcap_filename", "flowbits",
}

# The payload is the signal: keep traversal sequences, encodings etc. verbatim,
# only strip volatile tokens inside.
SECOPS_POLICY = ScrubPolicy(
    drop_fields=EVE_DROP_FIELDS,
    preserve_fields={"src_ip", "dest_ip", "proto", "event_type", "source", "signature_id"},
    normalize_counters=False,
    tokenize_ips=False,
    tokenize_emails=False,
    max_string_len=1024,
)

SYSLOG_RE = re.compile(
    r"^(?:<\d+>)?(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}T\S+)\s+"
    r"(?P<host>\S+)\s+(?P<prog>[\w./-]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$"
)


def _decode_payload(eve: dict[str, Any]) -> str:
    if eve.get("payload_printable"):
        return str(eve["payload_printable"])
    if eve.get("payload"):
        try:
            return base64.b64decode(eve["payload"]).decode("latin-1")
        except Exception:
            return ""
    http = eve.get("http") or {}
    if http.get("url"):
        method = http.get("http_method", "GET")
        proto = http.get("protocol", "HTTP/1.1")
        line = f"{method} {http['url']} {proto}\r\n"
        if http.get("hostname"):
            line += f"Host: {http['hostname']}\r\n"
        if http.get("http_user_agent"):
            line += f"User-Agent: {http['http_user_agent']}\r\n"
        return line + "\r\n"
    dns = eve.get("dns") or {}
    if dns.get("rrname"):
        return f"DNS {dns.get('rrtype', 'A')} {dns['rrname']}"
    tls = eve.get("tls") or {}
    if tls.get("sni"):
        return f"TLS SNI {tls['sni']} JA3 {((tls.get('ja3') or {}).get('hash')) or '-'}"
    return ""


class LogExtractor:
    def __init__(self, scrubber: FuzzyScrubber | None = None, payload_max: int = 512) -> None:
        self.scrubber = scrubber or FuzzyScrubber(SECOPS_POLICY)
        self.payload_max = payload_max

    def _snippet(self, payload: str, vault: EntityVault | None) -> str:
        payload = payload[: self.payload_max]
        # scrub volatile tokens but keep the attack surface (paths, encodings)
        return self.scrubber.scrub_text(payload, vault)

    def from_eve(self, eve: dict[str, Any] | str, vault: EntityVault | None = None) -> dict[str, Any]:
        if isinstance(eve, str):
            eve = json.loads(eve)
        state: dict[str, Any] = {
            "source": "suricata_eve",
            "src_ip": eve.get("src_ip"),
            "payload_snippet": self._snippet(_decode_payload(eve), vault),
            "event_type": eve.get("event_type"),
            "dest_ip": eve.get("dest_ip"),
            "dest_port": eve.get("dest_port"),
            "proto": eve.get("proto"),
        }
        if eve.get("app_proto"):
            state["app_proto"] = eve["app_proto"]
        alert = eve.get("alert") or {}
        if alert:
            state["alert"] = {
                "signature": alert.get("signature"),
                "signature_id": alert.get("signature_id"),
                "category": alert.get("category"),
                "severity": alert.get("severity"),
                "action": alert.get("action"),
            }
        http = eve.get("http") or {}
        if http:
            state["http"] = {
                k: http[k]
                for k in ("hostname", "url", "http_method", "http_user_agent", "status", "http_content_type")
                if k in http
            }
        state = {k: v for k, v in state.items() if v not in (None, "", {})}
        state.setdefault("payload_snippet", "")
        return self.scrubber.scrub(state, vault)

    def from_syslog(self, line: str, vault: EntityVault | None = None) -> dict[str, Any]:
        m = SYSLOG_RE.match(line.strip())
        if not m:
            return self.from_text(line, vault)
        msg = m.group("msg")
        ips = IP_RE.findall(msg)
        state: dict[str, Any] = {
            "source": "syslog",
            "src_ip": ips[0] if ips else None,
            "program": m.group("prog"),
            "host": m.group("host"),
            "payload_snippet": self._snippet(msg, vault),
        }
        return self.scrubber.scrub({k: v for k, v in state.items() if v is not None}, vault)

    def from_text(self, text: str, vault: EntityVault | None = None) -> dict[str, Any]:
        ips = IP_RE.findall(text)
        state = {"source": "text", "src_ip": ips[0] if ips else None, "payload_snippet": self._snippet(text, vault)}
        return self.scrubber.scrub({k: v for k, v in state.items() if v is not None}, vault)

    def from_json(self, obj: dict[str, Any], vault: EntityVault | None = None) -> dict[str, Any]:
        if "event_type" in obj and ("src_ip" in obj or "flow_id" in obj):
            return self.from_eve(obj, vault)
        src = obj.get("src_ip") or obj.get("source_ip") or obj.get("client_ip") or obj.get("remote_addr")
        payload = obj.get("payload_snippet") or obj.get("message") or obj.get("msg") or obj.get("request") or ""
        rest = {k: v for k, v in obj.items() if k not in ("src_ip", "source_ip", "client_ip", "remote_addr", "message", "msg", "request", "payload_snippet")}
        state = {"source": obj.get("source", "json"), "src_ip": src, "payload_snippet": self._snippet(str(payload), vault), **rest}
        state["source"] = str(state["source"])
        return self.scrubber.scrub({k: v for k, v in state.items() if v is not None}, vault)

    def extract(self, raw: Any, fmt: str = "auto", vault: EntityVault | None = None) -> dict[str, Any]:
        if fmt == "suricata_eve":
            return self.from_eve(raw, vault)
        if fmt == "syslog":
            return self.from_syslog(raw, vault)
        if fmt == "json":
            return self.from_json(raw if isinstance(raw, dict) else json.loads(raw), vault)
        if fmt == "text":
            return self.from_text(raw, vault)
        # auto-detect
        if isinstance(raw, dict):
            return self.from_json(raw, vault)
        text = str(raw).strip()
        if text.startswith("{"):
            try:
                return self.from_json(json.loads(text), vault)
            except json.JSONDecodeError:
                pass
        return self.from_syslog(text, vault)
