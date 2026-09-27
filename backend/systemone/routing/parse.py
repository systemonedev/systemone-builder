"""Robust extraction of the action_output JSON object from model text."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Return (chain_of_thought, remainder)."""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():]
    if "</think>" in text:  # opening tag consumed by the chat template
        cot, rest = text.split("</think>", 1)
        return cot.strip(), rest
    return None, text


def _balanced_objects(text: str) -> list[tuple[int, int]]:
    spans, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse (action_output, chain_of_thought) from raw model output."""
    cot, rest = split_reasoning(text)
    candidates = [m.group(1) for m in _FENCE_RE.finditer(rest)]
    candidates += [rest[a:b] for a, b in reversed(_balanced_objects(rest))]
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "action_output" in obj and isinstance(obj["action_output"], dict):
                obj = obj["action_output"]
            return obj, cot
    return None, cot
