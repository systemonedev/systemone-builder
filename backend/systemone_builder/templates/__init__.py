"""Domain-specific starter templates (Module H).

The two primary example domains (``computer_use``, ``secops``) are always
installed. Additional starter templates ship as JSON files in this package
and can be installed as-is or customized with the Prompt-to-Workflow engine.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from systemone_builder.domains.builtin import BUILTIN_DOMAINS
from systemone_builder.domains.spec import DomainSpec


def _json_templates() -> dict[str, DomainSpec]:
    out: dict[str, DomainSpec] = {}
    for entry in resources.files(__package__).iterdir():
        if entry.name.endswith(".json"):
            spec = DomainSpec.model_validate({**json.loads(entry.read_text()), "source": "template"})
            out[spec.id] = spec
    return out


def all_templates() -> dict[str, DomainSpec]:
    return {**BUILTIN_DOMAINS, **_json_templates()}


def list_templates() -> list[dict[str, Any]]:
    return [
        {"id": t.id, "name": t.name, "kind": t.kind, "description": t.description, "threshold": t.threshold,
         "supported_actions": t.supported_actions, "extractor": t.extractor.type, "scenarios": len(t.factory.scenarios),
         "builtin": t.id in BUILTIN_DOMAINS}
        for t in all_templates().values()
    ]


def load_template(template_id: str) -> DomainSpec:
    try:
        return all_templates()[template_id]
    except KeyError:
        raise KeyError(f"unknown template {template_id!r}") from None
