"""Built-in domains for the two primary example specializations."""

from __future__ import annotations

from systemone.contracts.computer_use import COMPUTER_USE_ACTIONS
from systemone.contracts.secops import SECOPS_ACTIONS
from systemone.domains.spec import DomainSpec, ExtractorConfig, FactoryConfig
from systemone.extraction.logs import EVE_DROP_FIELDS

COMPUTER_USE_STATE_SCHEMA = {
    "type": "object",
    "required": ["url", "viewport_tree"],
    "properties": {
        "url": {"type": "string"},
        "temporal_buffer": {"type": "array", "items": {"type": "string"}},
        "viewport_tree": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "role"],
                "properties": {
                    "id": {"type": "integer"},
                    "role": {"type": "string"},
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    "states": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

COMPUTER_USE_ACTION_SCHEMA = {
    "type": "object",
    "required": ["confidence_score", "action"],
    "properties": {
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "action": {"type": "string"},
        "target_id": {"type": ["integer", "null"]},
        "coordinates": {
            "type": ["object", "null"],
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
        "text": {"type": ["string", "null"]},
        "supported_actions": {"type": "array", "items": {"type": "string"}},
    },
}

SECOPS_STATE_SCHEMA = {
    "type": "object",
    "required": ["source", "payload_snippet"],
    "properties": {
        "source": {"type": "string"},
        "src_ip": {"type": ["string", "null"]},
        "payload_snippet": {"type": "string"},
    },
}

SECOPS_ACTION_SCHEMA = {
    "type": "object",
    "required": ["confidence_score", "verdict", "immediate_action", "target_ioc"],
    "properties": {
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict": {"type": "string", "enum": ["BENIGN", "SUSPICIOUS", "MALICIOUS"]},
        "immediate_action": {"type": "string"},
        "target_ioc": {"type": "array", "items": {"type": "string"}},
        "supported_actions": {"type": "array", "items": {"type": "string"}},
    },
}

COMPUTER_USE = DomainSpec(
    id="computer_use",
    name="Real-Time Computer-Use (GUI/DOM)",
    description="Reflex GUI automation over accessibility trees with pixel-coordinate fallbacks.",
    kind="computer_use",
    state_schema=COMPUTER_USE_STATE_SCHEMA,
    action_schema=COMPUTER_USE_ACTION_SCHEMA,
    supported_actions=list(COMPUTER_USE_ACTIONS),
    action_field="action",
    threshold=0.70,
    system_prompt=(
        "You are a System-1 GUI reflex agent. Given the current page state, emit exactly one next action "
        "as JSON. Use CLICK/TYPE with a target_id from viewport_tree whenever the element is present. "
        "Use CLICK_XY with pixel coordinates only when the target has no accessible node. Use ESCALATE "
        "when the goal is ambiguous or the page is in an unexpected state. confidence_score is your "
        "calibrated probability that the action is correct."
    ),
    few_shots=[
        {
            "state": {
                "url": "https://dashboard.internal/auth",
                "viewport_tree": [
                    {"id": 1, "role": "textbox", "name": "Username"},
                    {"id": 2, "role": "textbox", "name": "Password"},
                    {"id": 3, "role": "button", "name": "Sign in"},
                ],
                "temporal_buffer": ["TYPE(1, 'admin')"],
            },
            "action": {"confidence_score": 0.93, "action": "CLICK", "target_id": 2},
        }
    ],
    prompt_key_order=["url", "viewport_tree", "temporal_buffer"],
    extractor=ExtractorConfig(type="dom", temporal_buffer_len=8),
    factory=FactoryConfig(
        scenarios=[
            "log into an internal admin dashboard",
            "fill and submit a multi-field web form",
            "navigate a settings menu to toggle an option",
            "search a table and open a matching record",
            "dismiss a cookie banner or modal dialog",
            "recover from an inline validation error",
        ]
    ),
    source="builtin",
)

SECOPS = DomainSpec(
    id="secops",
    name="Cybersecurity Reflex Triage (SecOps)",
    description="Sub-100ms triage of IDS/log events into verdicts and containment actions.",
    kind="secops",
    state_schema=SECOPS_STATE_SCHEMA,
    action_schema=SECOPS_ACTION_SCHEMA,
    supported_actions=list(SECOPS_ACTIONS),
    action_field="immediate_action",
    threshold=0.95,
    system_prompt=(
        "You are a System-1 SecOps reflex triage agent. Classify the event as BENIGN, SUSPICIOUS or "
        "MALICIOUS and choose one immediate_action. Only list IOCs that literally appear in the event. "
        "Choose ESCALATE when evidence is insufficient for an automated containment decision. "
        "confidence_score is your calibrated probability that the verdict and action are correct."
    ),
    few_shots=[
        {
            "state": {
                "source": "suricata_eve",
                "event_type": "http",
                "dest_ip": "10.0.0.5",
                "dest_port": 80,
                "src_ip": "192.168.1.150",
                "payload_snippet": "GET /../../../../etc/passwd HTTP/1.1\r\n\r\n",
            },
            "action": {
                "confidence_score": 0.98,
                "verdict": "SUSPICIOUS",
                "immediate_action": "DROP_AND_BLACKLIST_IP",
                "target_ioc": ["192.168.1.150"],
            },
        }
    ],
    prompt_key_order=["source", "event_type", "proto", "app_proto", "dest_ip", "dest_port", "alert", "http", "src_ip", "payload_snippet"],
    extractor=ExtractorConfig(
        type="log",
        drop_fields=sorted(EVE_DROP_FIELDS),
        preserve_fields=["src_ip", "dest_ip", "proto", "event_type", "source", "signature_id"],
        normalize_counters=False,
        tokenize_emails=False,
        max_string_len=1024,
    ),
    factory=FactoryConfig(
        scenarios=[
            "path traversal attempt against a web server",
            "SQL injection in a query string",
            "SSH brute force from a single source",
            "benign health-check and monitoring traffic",
            "log4shell JNDI lookup in a header",
            "suspicious DNS tunneling query",
            "benign software update download",
        ]
    ),
    source="builtin",
)

BUILTIN_DOMAINS = {d.id: d for d in (COMPUTER_USE, SECOPS)}
