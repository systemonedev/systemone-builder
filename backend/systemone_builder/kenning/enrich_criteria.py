"""Ask a local teacher to write richer criteria for the generated noul questions.

:mod:`systemone_builder.kenning.criteria` attaches criteria built from the one-line label
descriptions the generators already hold ("no time pressure"). Those are correct but terse, while
the criteria real callers send are concrete and name the near misses:

    true:  The customer says they will post publicly, contact the press, or dispute the charge.
    false: No such threat, or it is hypothetical, quoted from someone else, or already in the past.

That *false* side is what teaches discrimination. A bare negation ("it is not a threat") carries
nothing the stem did not already say.

This writes several variants per question template to ``criteria_variants.json``, once, offline. The
generators then sample a variant per row, so data generation stays deterministic and needs no
teacher at run time. Length is capped deliberately: the hypothesis shares the model's token budget
with the state, and over-long options are what starved the premise on long documents.

    python -m systemone_builder.kenning.enrich_criteria --base-url http://HOST:11434/v1 \
        --model qwen3.8:27b --variants 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.kenning.synthetic_tasks import TASKS

VARIANTS_FILE = Path(__file__).with_name("criteria_variants.json")
MAX_WORDS = 25          # per side; keeps the hypothesis from crowding out the state
SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"true": {"type": "string"}, "false": {"type": "string"}},
                "required": ["true", "false"],
            },
        }
    },
    "required": ["variants"],
}

PROMPT = """You write decision criteria for one yes/no question used by an automated classifier.

Question: {question}
"Yes" means: {yes}
"No" means: {no}

Write {n} alternative phrasings of these criteria. Each phrasing has a "true" side and a "false"
side.

Rules:
- The "true" side states concretely what must hold for the answer to be yes.
- The "false" side must name the near misses: cases that resemble yes but are not. Never write a
  bare negation such as "it is not that" or "no, it is something else".
- At most {max_words} words per side. Be concrete, not abstract.
- Describe the boundary, never one specific example.
- The phrasings must differ in wording, not in meaning.

Worked example for a different question:
  Question: Does the customer threaten to escalate publicly?
  true:  The customer says they will post publicly, contact the press, or dispute the charge.
  false: No such threat, or it is hypothetical, quoted from someone else, or already in the past.
"""


def templates() -> list[dict[str, Any]]:
    """Every noul question shape the generators can emit, with the descriptions we already have."""
    out: list[dict[str, Any]] = []
    for task in TASKS:
        for attr in task.attrs:
            names = list(attr.labels)
            instr = attr.question[0]
            if set(names) == {"yes", "no"}:
                out.append({"key": f"{task.name}.{attr.name}", "kind": "yes_no", "question": instr,
                            "yes": attr.labels["yes"], "no": attr.labels["no"]})
                continue
            for label in names:
                others = [n for n in names if n != label][:4]
                desc = attr.labels[label]
                out.append({
                    "key": f"{task.name}.{attr.name}.{label}", "kind": "label",
                    "question": f"{instr.rstrip('?')} - is the answer \"{label}\"?",
                    "yes": f"it is {label}" + (f" ({desc})" if desc else ""),
                    "no": "it is one of the other options: " + ", ".join(others),
                })
    return out


def _ok(side: str) -> bool:
    s = " ".join(str(side).split())
    if not s or len(s.split()) > MAX_WORDS:
        return False
    low = s.lower().rstrip(".")
    # a bare negation teaches nothing the stem did not already say
    return not (low.startswith("it is not") and "," not in low and " but " not in low)


def valid(variant: dict[str, Any]) -> bool:
    t, f = variant.get("true"), variant.get("false")
    return bool(t) and bool(f) and _ok(t) and _ok(f) and str(t).strip() != str(f).strip()


async def enrich(base_url: str, model: str, n_variants: int, concurrency: int = 4,
                 api_key: str | None = None, timeout: float = 600,
                 limit: int | None = None, reasoning_effort: str | None = "none") -> dict[str, Any]:
    tmpls = templates()[:limit] if limit else templates()
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    done = 0

    async def one(client: httpx.AsyncClient, t: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        nonlocal done
        body: dict[str, Any] = {
            "model": model, "temperature": 0.7, "max_tokens": 900,
            "messages": [{"role": "user", "content": PROMPT.format(
                question=t["question"], yes=t["yes"], no=t["no"], n=n_variants,
                max_words=MAX_WORDS)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "criteria", "schema": SCHEMA}}}
        if reasoning_effort:
            # A reasoning model otherwise spends the whole budget thinking and returns no content.
            body["reasoning_effort"] = reasoning_effort
        variants: list[dict[str, str]] = []
        async with sem:
            try:
                r = await client.post("/chat/completions", json=body)
                payload = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
                for v in payload.get("variants") or []:
                    if valid(v):
                        variants.append({"true": " ".join(str(v["true"]).split()),
                                         "false": " ".join(str(v["false"]).split())})
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                print(f"  ! {t['key']}: {type(exc).__name__}", file=sys.stderr)
        done += 1
        print(f"  [{done}/{len(tmpls)}] {t['key']}: {len(variants)} variant(s)", file=sys.stderr)
        return t["key"], {"kind": t["kind"], "variants": variants}

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                 headers=headers) as client:
        pairs = await asyncio.gather(*(one(client, t) for t in tmpls))
    return {k: v for k, v in pairs if v["variants"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", required=True, help="OpenAI-compatible teacher, e.g. http://host:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", type=Path, default=VARIANTS_FILE)
    p.add_argument("--limit", type=int, default=None, help="only the first N templates (for a trial run)")
    p.add_argument("--reasoning-effort", default="none",
                   help='sent as reasoning_effort; a reasoning teacher otherwise returns no content. '
                        'Pass "" to omit the field for providers that reject it.')
    a = p.parse_args(argv)

    got = asyncio.run(enrich(a.base_url, a.model, a.variants, a.concurrency, limit=a.limit,
                             reasoning_effort=a.reasoning_effort or None))
    total = sum(len(v["variants"]) for v in got.values())
    doc = {"version": 1, "teacher": a.model, "max_words": MAX_WORDS,
           "note": "Written offline by a local teacher; see kenning/enrich_criteria.py.",
           "criteria": got}
    a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(got)} templates, {total} variants -> {a.out}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
