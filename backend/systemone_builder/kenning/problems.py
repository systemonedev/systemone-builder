"""Problem specs: train Kenning for your own decision problem.

A problem spec is a small JSON file that says what the state looks like (one text, or named fields)
and which questions to ask about it, with every possible answer described:

    {
      "name": "computer_use",
      "title": "Computer-use agent: check the next action",
      "writes": "one step of a computer-use agent working in a web browser",
      "state": {"fields": {"goal": "the user's task", "screen": "...", "proposed_action": "..."}},
      "questions": [
        {"id": "advances_goal", "ask": ["Does the proposed action move the task forward?"],
         "labels": {"yes": "it is a sensible next step", "no": "it is unrelated or a step backwards"}},
        {"id": "risk", "ask": ["How risky is the proposed action?"], "ordinal": true,
         "labels": {"low": "...", "medium": "...", "high": "..."}}
      ]
    }

A teacher LLM writes one case *for* randomly drawn answers ("write a step where the action is risky
and does not advance the goal"), so every label is known by construction. Each case becomes a
training row with varied phrasings (noul, choice, score, "is it X?"). A share of the cases is held out
with fixed questions, so the trained model can be benchmarked on the problem it was trained for.

Labels by construction are only as good as the teacher's writing: run the Label job (Clef) on the
dataset to check agreement, and drop or fix sources where it is low.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from systemone_builder.kenning.synthetic_tasks import STYLES, Attr, questions_for

BUILTIN_DIR = Path(__file__).parent / "problems"
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_TEXT = 4000


@dataclass
class Problem:
    name: str
    title: str
    writes: str
    fields: dict[str, str]          # field -> description; one "text" field for a plain-text state
    attrs: list[Attr]
    text_key: str | None = None     # set when the state is one text: {text_key: "..."}
    styles: list[str] = field(default_factory=lambda: list(STYLES))
    description: str = ""
    constraints: list[tuple[dict[str, list[str]], dict[str, list[str]]]] = field(default_factory=list)

    def allows(self, labels: dict[str, str]) -> bool:
        """False when the labels break a constraint (if all of `when` hold, `then` must hold)."""
        for when, then in self.constraints:
            if all(labels[k] in v for k, v in when.items()) and not all(labels[k] in v for k, v in then.items()):
                return False
        return True

    def draw(self, rng: random.Random, fix: dict[str, str] | None = None) -> dict[str, str]:
        """Random labels that satisfy the constraints; ``fix`` pins some questions' answers."""
        for _ in range(400):
            labels = {a.name: fix[a.name] if fix and a.name in fix else rng.choice(list(a.labels)) for a in self.attrs}
            if self.allows(labels):
                return labels
        raise ValueError(f"{self.name}: the constraints rule out (almost) every combination of labels"
                         + (f" with {fix}" if fix else ""))

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description,
                "state": [self.text_key] if self.text_key else list(self.fields),
                "questions": {a.name: list(a.labels) for a in self.attrs}}


def parse_problem(d: Any) -> Problem:
    """A validated Problem from a spec dict (raises ValueError with a readable message)."""
    if not isinstance(d, dict):
        raise ValueError("a problem spec is a JSON object")
    name = d.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError("name: lowercase letters, digits, '_' or '-', up to 64 characters")
    writes = d.get("writes")
    if not isinstance(writes, str) or len(writes) < 5:
        raise ValueError("writes: describe what the teacher should write, e.g. 'a customer support ticket'")
    state = d.get("state") or {}
    text_key = None
    if isinstance(state.get("text"), str) and NAME.match(state["text"]):
        text_key, fields = state["text"], {state["text"]: writes}
    elif isinstance(state.get("fields"), dict) and 1 <= len(state["fields"]) <= 12 and all(
            isinstance(k, str) and NAME.match(k) and isinstance(v, str) for k, v in state["fields"].items()):
        fields = dict(state["fields"])
    else:
        raise ValueError('state: {"text": "<key>"} for one text, or {"fields": {"<name>": "<description>", ...}} '
                         "with 1-12 fields")
    qs = d.get("questions")
    if not isinstance(qs, list) or not 1 <= len(qs) <= 8:
        raise ValueError("questions: a list of 1-8 questions")
    attrs: list[Attr] = []
    for i, q in enumerate(qs):
        where = f"questions[{i}]"
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not NAME.match(q["id"]):
            raise ValueError(f"{where}.id: lowercase letters, digits, '_' or '-'")
        ask = q.get("ask")
        ask = [ask] if isinstance(ask, str) else ask
        if not isinstance(ask, list) or not ask or not all(isinstance(a, str) and a.strip() for a in ask):
            raise ValueError(f"{where}.ask: the question, or a list of phrasings")
        labels = q.get("labels")
        if not isinstance(labels, dict) or not 2 <= len(labels) <= 10 or not all(
                isinstance(k, str) and k.strip() and isinstance(v, (str, type(None))) for k, v in labels.items()):
            raise ValueError(f"{where}.labels: 2-10 answers, each mapped to a description (or null)")
        ordinal = bool(q.get("ordinal"))
        if ordinal and set(labels) == {"yes", "no"}:
            raise ValueError(f"{where}: a yes/no question can't be ordinal")
        attrs.append(Attr(q["id"], ask, {k: v or "" for k, v in labels.items()}, ordinal=ordinal))
    if len({a.name for a in attrs}) != len(attrs):
        raise ValueError("questions: ids must be unique")
    styles = d.get("styles") or list(STYLES)
    if not isinstance(styles, list) or not all(isinstance(s, str) for s in styles):
        raise ValueError("styles: a list of strings")
    known = {a.name: a.labels for a in attrs}
    constraints = []
    for i, c in enumerate(d.get("constraints") or []):
        parts = []
        for side in ("if", "then"):
            cond = c.get(side) if isinstance(c, dict) else None
            if not isinstance(cond, dict) or not cond:
                raise ValueError(f'constraints[{i}]: {{"if": {{question: answer(s)}}, "then": {{question: answer(s)}}}}')
            norm = {}
            for k, v in cond.items():
                vals = [v] if isinstance(v, str) else v
                if k not in known or not isinstance(vals, list) or not all(x in known[k] for x in vals):
                    raise ValueError(f"constraints[{i}].{side}: {k!r} must be a question id and its values its answers")
                norm[k] = vals
            parts.append(norm)
        constraints.append((parts[0], parts[1]))
    problem = Problem(name, str(d.get("title") or name), writes, fields, attrs, text_key, styles,
                      str(d.get("description") or ""), constraints)
    problem.draw(random.Random(0))  # fails early when the constraints leave nothing to draw
    return problem


def builtin_problems() -> dict[str, Problem]:
    return {p.name: p for p in (parse_problem(json.loads(f.read_text(encoding="utf-8")))
                                for f in sorted(BUILTIN_DIR.glob("*.json")))}


def load_problem(ref: str, user_dir: Path | None = None) -> Problem:
    """A built-in problem by name, a spec saved in user_dir, or a path to a spec file."""
    if user_dir and (user_dir / f"{ref}.json").exists():
        return parse_problem(json.loads((user_dir / f"{ref}.json").read_text(encoding="utf-8")))
    builtin = builtin_problems()
    if ref in builtin:
        return builtin[ref]
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return parse_problem(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"no problem named {ref!r} (built in: {', '.join(builtin)})")


# ------------------------------------------------------------------ writing cases
def _prompt(p: Problem, labels: dict[str, str], style: str) -> str:
    want = "; ".join(f"{a.name}: {labels[a.name]}" + (f" ({a.labels[labels[a.name]]})" if a.labels[labels[a.name]] else "")
                     for a in p.attrs)
    if p.text_key:
        shape = "Return JSON with the text in 'text'."
    else:
        shape = "Return JSON with these fields, each a string: " + "; ".join(f"'{k}': {v}" for k, v in p.fields.items()) + "."
    return (f"Write {p.writes}. It must clearly have these properties - {want}. Style: {style}. Make it realistic "
            f"and specific. Do not name the properties or labels in the content itself. {shape}")


def _schema(p: Problem) -> dict[str, Any]:
    keys = ["text"] if p.text_key else list(p.fields)
    return {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys}


def _to_state(p: Problem, out: Any) -> dict[str, Any] | None:
    if not isinstance(out, dict):
        return None
    if p.text_key:
        text = out.get("text")
        return {p.text_key: text.strip()[:MAX_TEXT]} if isinstance(text, str) and len(text.strip()) >= 15 else None
    state = {k: out.get(k) for k in p.fields}
    if not all(isinstance(v, str) and v.strip() for v in state.values()):
        return None
    return {k: v.strip()[:MAX_TEXT] for k, v in state.items()}


def _check_prompt(p: Problem, state: dict[str, Any]) -> str:
    qs = "\n".join(f"- {a.name}: {a.question[0]} Answers: "
                   + "; ".join(n + (f" ({d})" if d else "") for n, d in a.labels.items()) for a in p.attrs)
    return (f"Here is {p.writes}:\n{json.dumps(state, indent=1, ensure_ascii=False)}\n\nAnswer each question with "
            f"exactly one of its answers.\n{qs}\nReturn JSON mapping each question id to its answer.")


async def _chat(client: httpx.AsyncClient, model: str, prompt: str, schema: dict[str, Any], seed: int,
                temperature: float, max_tokens: int, extra: dict[str, Any] | None = None) -> Any:
    body = {"model": model, "temperature": temperature, "top_p": 0.95, "seed": seed, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
            **(extra or {})}
    r = await client.post("/chat/completions", json=body)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


async def write_cases(p: Problem, n: int, seed: int, base_url: str, model: str, api_key: str | None = None,
                      concurrency: int = 16, max_tokens: int = 700, check: bool = True,
                      stats: dict[str, int] | None = None, extra: dict[str, Any] | None = None,
                      balance: bool = False) -> list[dict[str, Any]]:
    """Up to n distinct label-conditioned cases ({"state", "labels"}) written by an OpenAI-compatible teacher.

    With check, the teacher then answers the questions about each case without seeing the intended
    labels, and a case is kept only when every answer matches (a blind agreement filter). ``extra`` is
    added to every request body, e.g. ``{"reasoning_effort": "none"}`` to turn off a reasoning model's
    thinking on servers that support it. ``balance`` draws the first question's answers uniformly (so a
    decision like keep/drop isn't dominated by one class the constraints make common).
    """
    rng = random.Random(f"problem-{p.name}-{seed}")
    primary = p.attrs[0].name
    primary_vals = list(p.attrs[0].labels)

    def _draw() -> dict[str, str]:
        return p.draw(rng, {primary: rng.choice(primary_vals)}) if balance else p.draw(rng)
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    schema = _schema(p)
    check_schema = {"type": "object", "properties": {a.name: {"type": "string", "enum": list(a.labels)} for a in p.attrs},
                    "required": [a.name for a in p.attrs]}
    counts = stats if stats is not None else {}
    for k in ("asked", "written", "disagreed", "kept"):
        counts.setdefault(k, 0)

    async def one(client: httpx.AsyncClient, i: int, labels: dict[str, str], style: str) -> dict[str, Any] | None:
        counts["asked"] += 1
        async with sem:
            try:
                state = _to_state(p, await _chat(client, model, _prompt(p, labels, style), schema,
                                                 seed * 100_003 + i, 0.9, max_tokens, extra))
                if not state:
                    return None
                counts["written"] += 1
                if check:
                    seen = await _chat(client, model, _check_prompt(p, state), check_schema, seed * 100_019 + i, 0.0,
                                       200, extra)
                    if not isinstance(seen, dict) or any(seen.get(k) != v for k, v in labels.items()):
                        counts["disagreed"] += 1
                        return None
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError, AttributeError):
                return None
        counts["kept"] += 1
        return {"state": state, "labels": labels}

    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    max_attempts = 5 * n  # a teacher that keeps under 20% is the wrong teacher for this problem
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=300, headers=headers) as client:
        while len(cases) < n and counts["asked"] < max_attempts:
            rate = max(counts["kept"] / counts["asked"], 0.2) if counts["asked"] else 1.0
            batch = min(math.ceil((n - len(cases)) / rate * 1.1), max_attempts - counts["asked"])
            jobs = [(_draw(), rng.choice(p.styles)) for _ in range(batch)]
            start = counts["asked"]
            out = await asyncio.gather(*(one(client, start + j, *job) for j, job in enumerate(jobs)))
            for x in out:
                key = json.dumps(x["state"], sort_keys=True) if x else ""
                if x and key not in seen:
                    seen.add(key)
                    cases.append(x)
    return cases[:n]


# ------------------------------------------------------------------ rows and hold-out
def training_rows(p: Problem, cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"problem-rows-{p.name}-{seed}")
    return [{"state": c["state"], **questions_for(p.attrs, c["labels"], rng)} for c in cases]


def canonical_questions(p: Problem) -> dict[str, dict[str, Any]]:
    """One fixed question per attribute, for the hold-out benchmark."""
    out: dict[str, dict[str, Any]] = {}
    for a in p.attrs:
        if set(a.labels) == {"yes", "no"}:
            out[a.name] = {"type": "noul", "instructions": a.question[0]}
        elif a.ordinal:
            out[a.name] = {"type": "score", "instructions": a.question[0],
                           "criteria": [f"{n}: {d}" if d else n for n, d in a.labels.items()]}
        else:
            out[a.name] = {"type": "choice", "instructions": a.question[0],
                           "criteria": {n: d or None for n, d in a.labels.items()}}
    return out


def holdout_items(p: Problem, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Benchmark items ({id, state, labels}) in the format `systemone bench --suite <file>` reads."""
    items = []
    for i, c in enumerate(cases):
        labels: dict[str, Any] = {}
        for a in p.attrs:
            truth = c["labels"][a.name]
            labels[a.name] = (int(truth == "yes") if set(a.labels) == {"yes", "no"}
                              else list(a.labels).index(truth) if a.ordinal else truth)
        items.append({"id": f"{p.name}-{i}", "state": c["state"], "labels": labels})
    return items


def split(cases: list[Any], share: float, seed: int, key: str) -> tuple[list[Any], list[Any]]:
    """(train, holdout): a seeded share held out, at least 20 items when there are enough cases."""
    if share <= 0 or len(cases) < 40:
        return list(cases), []
    idx = list(range(len(cases)))
    random.Random(f"holdout-{key}-{seed}").shuffle(idx)
    k = min(max(20, round(len(cases) * share)), len(cases) // 2)
    hold = set(idx[:k])
    return [c for i, c in enumerate(cases) if i not in hold], [c for i, c in enumerate(cases) if i in hold]
