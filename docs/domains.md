# Domains, contracts & templates

A **domain** (`DomainSpec`) bundles everything needed to distill one specialization:

* state and action JSON schemas
* the action vocabulary and `action_field`
* the confidence threshold (plus an optional triage threshold)
* the reflex system prompt and few-shot examples
* `prompt_key_order` for prefix stability
* the extractor configuration (`dom` / `log` / `passthrough`, dropped fields, custom volatile
  regexes, entity tokenization)
* synthetic-factory parameters (scenarios, samples per scenario, judge minimum score)

## Built-in data contracts

**Computer-use**, with spatial fallbacks. `CLICK` and `TYPE` need a `target_id` that exists in
`viewport_tree`; `TYPE` also needs `text`. `CLICK_XY` needs `coordinates`.

```json
{"state_input": {"url": "https://dashboard.internal/auth",
                 "temporal_buffer": ["CLICK(12)", "TYPE(2, 'admin')"],
                 "viewport_tree": [{"id": 3, "role": "textbox", "name": "Password"}]},
 "action_output": {"confidence_score": 0.92, "action": "CLICK_XY", "target_id": null,
                   "coordinates": {"x": 450, "y": 820},
                   "supported_actions": ["CLICK", "TYPE", "CLICK_XY", "ESCALATE"]}}
```

**Cybersecurity reflex.** `DROP_AND_BLACKLIST_IP` needs valid IP IOCs, and every IOC must
literally appear in the observed state.

```json
{"state_input": {"source": "suricata_eve", "src_ip": "192.168.1.150",
                 "payload_snippet": "GET /../../../../etc/passwd HTTP/1.1\r\n\r\n"},
 "action_output": {"confidence_score": 0.98, "verdict": "SUSPICIOUS",
                   "immediate_action": "DROP_AND_BLACKLIST_IP", "target_ioc": ["192.168.1.150"],
                   "supported_actions": ["DROP_AND_BLACKLIST_IP", "ESCALATE", "ALLOW"]}}
```

## Observation kinds

| kind | payload |
|---|---|
| `html` | raw page HTML (+ `url`) |
| `ax_tree` | Playwright `accessibility.snapshot()` / CDP-style tree |
| `elements` | output of `GET /api/v1/extract/collector.js` run with `page.evaluate` (bboxes + selectors) |
| `screenshot` | base64 PNG; vision-parsed on the Mac |
| `suricata_eve`, `syslog`, `json`, `text`, `log` | security telemetry |
| `state` | an already-structured `state_input` (scrub only) |

An observation can also carry `goal`, `temporal_buffer` and `viewport`. For computer-use,
the temporal buffer is kept per `session_id` automatically.

## Starter templates

| id | kind | notes |
|---|---|---|
| `computer_use` | computer_use | web GUI automation, τ 0.70 |
| `secops` | secops | Suricata/log triage, τ 0.95 |
| `desktop_vision` | computer_use | screenshot-only GUIs, bbox elements, CLICK_XY |
| `auth_log_bruteforce` | secops | sshd/sudo/PAM brute-force reflex |

List them with `systemone templates` or `GET /api/v1/templates`, and install one with
`POST /api/v1/templates/{id}/install`.

## Prompt-to-Workflow

`POST /api/v1/workflows/generate {"prompt": "..."}` does the following:

1. Classifies the prompt as GUI, security or custom, and picks the closest starter template.
2. Asks the teacher for a full `DomainSpec`.
3. Validates it: schemas, `ESCALATE` present, few-shots valid against the schemas, and regexes
   compile. Errors are fed back to the teacher for up to two repair rounds.

The result is stored as a draft. Review and edit it in the dashboard, then activate it with
`POST /api/v1/workflows/activate`. Activation registers the domain and, optionally, starts
seed synthesis.
