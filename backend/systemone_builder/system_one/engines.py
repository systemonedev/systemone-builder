"""Engines that answer the System One contract.

* :class:`JevEngine` - TypeSafe's hosted Jev, the reference System One model.
* :class:`LocalLogprobEngine` - the local baseline. Each question becomes one
  forward pass of a local model whose answer is a single label token (option
  letter, level digit, Yes/No); the answer distribution is read from that
  token's probabilities instead of being generated. No text is produced, the
  readout is argmax over a distribution, and questions run in parallel. It is
  a stand-in until a dedicated typed-head System One model is trained.
* :class:`LLMJsonEngine` - the System 2 baseline: a generative LLM writes the
  answers as JSON with a self-reported confidence, the way agents usually
  "decide" today. It is what System One models are meant to beat.
"""

from __future__ import annotations

import abc
import asyncio
import json
import math
import string
import time
from typing import Any

import httpx

from systemone_builder.adapters.base import ModelAdapter
from systemone_builder.system_one.contract import (
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    choice_answer,
    describe,
    noul_answer,
    render_state,
    score_answer,
)

# vLLM's default --max-logprobs; option labels must fit in the top-k readout.
LOCAL_MAX_OPTIONS = 20
CHOICE_LABELS = string.ascii_uppercase[:LOCAL_MAX_OPTIONS]


class EngineError(RuntimeError):
    pass


class SystemOneEngine(abc.ABC):
    name: str = "engine"

    @abc.abstractmethod
    async def answer(self, req: SystemOneRequest) -> SystemOneResponse: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass


# ------------------------------------------------------------------- Jev
class JevEngine(SystemOneEngine):
    """Anything speaking Jev's ``POST {base_url}/v1/systemone``: TypeSafe Jev itself,
    or the local System One model server (``require_key=False``)."""

    def __init__(self, api_key: str | None, base_url: str = "https://api.typesafe.ai", model: str = "jev-latest",
                 timeout_s: float = 30.0, require_key: bool = True, name: str | None = None) -> None:
        self.model = model
        self.name = name or f"jev ({model})"
        self._key = api_key
        self._require_key = require_key
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
        )

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        if self._require_key and not self._key:
            raise EngineError("no TypeSafe API key: set TYPESAFE_API_KEY in .env")
        body = {
            "state": req.state,
            "model": req.model or self.model,
            "questions": {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()},
        }
        t0 = time.perf_counter()
        try:
            r = await self.client.post("/v1/systemone", json=body)
        except httpx.HTTPError as exc:
            raise EngineError(f"{self.name} unreachable: {exc!r}") from exc
        latency = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise EngineError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        out = SystemOneResponse.model_validate(r.json())
        out.latency_ms = latency
        return out

    async def aclose(self) -> None:
        await self.client.aclose()


# ---------------------------------------------------------------- local
SYSTEM_PROMPT = (
    "You are a System One decision model. Read the state, then answer the question with exactly one token: "
    "the label requested, nothing else."
)


def local_prompt(state: str, q: Question) -> tuple[str, list[str], list[Any]]:
    """Prompt for one question, the label tokens to read, and the matching options."""
    if q.type == "noul":
        crit = describe(q.criteria)
        text = f"Question: {q.instructions}\n" + (f"Criteria: {crit}\n" if crit else "") + "\nAnswer Yes or No."
        return text, ["Yes", "No"], ["yes", "no"]
    if q.type == "choice":
        options = list(q.criteria)  # type: ignore[arg-type]
        if len(options) > LOCAL_MAX_OPTIONS:
            raise EngineError(f"local engine reads at most {LOCAL_MAX_OPTIONS} choice options (got {len(options)})")
        labels = list(CHOICE_LABELS[: len(options)])
        lines = [f"{lb}) {o}" + (f": {describe(d)}" if describe(d) else "") for lb, (o, d) in zip(labels, q.criteria.items())]  # type: ignore[union-attr]
        text = f"Question: {q.instructions}\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with the option letter only."
        return text, labels, options
    levels = list(q.criteria)  # type: ignore[arg-type]
    labels = [str(i) for i in range(len(levels))]
    lines = [f"{i}) {describe(lv)}" for i, lv in enumerate(levels)]
    text = f"Question: {q.instructions}\nLevels:\n" + "\n".join(lines) + "\n\nAnswer with the level number only."
    return text, labels, levels


def label_mass(top_logprobs: list[dict[str, Any]], labels: list[str]) -> list[float]:
    """Probability of each label among the top-k next tokens.

    Tokenizer variants of one label (``Yes``, `` Yes``, ``yes``) are summed.
    A label missing from the top-k gets 0.
    """
    want = {lb.lower(): i for i, lb in enumerate(labels)}
    mass = [0.0] * len(labels)
    seen: set[str] = set()
    for t in top_logprobs:
        tok = t.get("token", "")
        if tok in seen:
            continue
        seen.add(tok)
        i = want.get(tok.strip().lower())
        if i is not None:
            mass[i] += math.exp(float(t.get("logprob", -1e9)))
    return mass


class LocalLogprobEngine(SystemOneEngine):
    """Answers each question from one forward pass of an OpenAI-compatible model."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0, concurrency: int = 16,
                 api_key: str | None = None) -> None:
        self.model = model
        self.name = f"local ({model})"
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
        )

    async def _one(self, state: str, qid: str, q: Question) -> tuple[str, Any, int]:
        text, labels, options = local_prompt(state, q)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"State:\n{state}\n\n{text}"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 0,
            "logprobs": True,
            "top_logprobs": LOCAL_MAX_OPTIONS,
        }
        async with self.sem:
            try:
                r = await self.client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                raise EngineError(f"local model unreachable: {exc!r}") from exc
        if r.status_code != 200:
            raise EngineError(f"local model HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("local model returned no logprobs (needs an OpenAI-compatible server with logprobs)") from exc
        mass = label_mass(top, labels)
        if q.type == "noul":
            ans: Any = noul_answer(mass[0], mass[1])
        elif q.type == "choice":
            ans = choice_answer(options, mass)
        else:
            ans = score_answer(options, mass)
        return qid, ans, int((data.get("usage") or {}).get("prompt_tokens") or 0)

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        state = render_state(req.state)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(self._one(state, qid, q) for qid, q in req.questions.items()))
        return SystemOneResponse(
            model=f"logprob/{self.model}",
            answers={qid: a for qid, a, _ in results},
            usage=Usage(input_tokens=sum(n for *_, n in results), output_tokens=len(results)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def aclose(self) -> None:
        await self.client.aclose()


# -------------------------------------------------------- LLM (System 2)
def llm_schema(questions: dict[str, Question]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for qid, q in questions.items():
        if q.type == "noul":
            props[qid] = {"type": "object", "properties": {"probability_yes": {"type": "number"}},
                          "required": ["probability_yes"]}
        elif q.type == "choice":
            props[qid] = {"type": "object", "properties": {"choice": {"type": "string", "enum": list(q.criteria)},  # type: ignore[arg-type]
                                                         "confidence": {"type": "number"}},
                          "required": ["choice", "confidence"]}
        else:
            props[qid] = {"type": "object", "properties": {"level": {"type": "integer", "minimum": 0,
                                                                     "maximum": len(q.criteria) - 1},  # type: ignore[arg-type]
                                                         "confidence": {"type": "number"}},
                          "required": ["level", "confidence"]}
    return {"type": "object", "properties": props, "required": list(props)}


def _clamp(x: Any) -> float:
    return min(max(float(x), 0.0), 1.0)


class LLMJsonEngine(SystemOneEngine):
    """A generative LLM writing the answers as JSON (the usual agent approach)."""

    def __init__(self, adapter: ModelAdapter, max_tokens: int = 512, name: str | None = None) -> None:
        self.adapter = adapter
        self.max_tokens = max_tokens
        self.name = name or f"llm ({adapter.model})"

    async def aclose(self) -> None:
        await self.adapter.aclose()

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        spec = []
        for qid, q in req.questions.items():
            if q.type == "noul":
                spec.append(f"- {qid}: {q.instructions} -> probability_yes (0..1)")
            elif q.type == "choice":
                opts = "; ".join(f"{o}" + (f" ({describe(d)})" if describe(d) else "") for o, d in q.criteria.items())  # type: ignore[union-attr]
                spec.append(f"- {qid}: {q.instructions} -> choice, one of: {opts}; plus confidence (0..1)")
            else:
                lv = "; ".join(f"{i}={describe(x)}" for i, x in enumerate(q.criteria))  # type: ignore[arg-type]
                spec.append(f"- {qid}: {q.instructions} -> level, one of: {lv}; plus confidence (0..1)")
        messages = [
            {"role": "system", "content": "You are an automated decision classifier. Return ONLY valid JSON matching the schema."},
            {"role": "user", "content": f"State:\n{render_state(req.state)}\n\nAnswer each question:\n" + "\n".join(spec)},
        ]
        t0 = time.perf_counter()
        gen = await self.adapter.generate(messages, json_schema=llm_schema(req.questions), max_tokens=self.max_tokens,
                                          temperature=0.0)
        latency = (time.perf_counter() - t0) * 1000
        try:
            data = json.loads(gen.text)
            answers: dict[str, Any] = {}
            for qid, q in req.questions.items():
                a = data[qid]
                if q.type == "noul":
                    p = _clamp(a["probability_yes"])
                    answers[qid] = noul_answer(p, 1 - p)
                    continue
                options = list(q.criteria)  # type: ignore[arg-type]
                # the stated answer must stay the most likely one
                c = max(_clamp(a["confidence"]), 1 / len(options))
                pick = options.index(a["choice"]) if q.type == "choice" else int(a["level"])
                if not 0 <= pick < len(options):
                    raise ValueError(f"{qid}: answer out of range")
                rest = (1 - c) / (len(options) - 1)
                probs = [c if i == pick else rest for i in range(len(options))]
                answers[qid] = choice_answer(options, probs) if q.type == "choice" else score_answer(options, probs)
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineError(f"LLM output failed the schema: {exc}") from exc
        return SystemOneResponse(
            model=f"llm/{self.adapter.model}", answers=answers,
            usage=Usage(input_tokens=gen.prompt_tokens or 0, output_tokens=gen.completion_tokens or 0),
            latency_ms=latency,
        )


# -------------------------------------------------------- Direct Kenning
class DirectKenningEngine(SystemOneEngine):
    """In-process Kenning cross-encoder engine (no HTTP container needed)."""

    def __init__(self, path: str, device: str | None = None, max_length: int | None = None) -> None:
        from pathlib import Path as _P

        from systemone_builder.kenning.model import read_config

        self._path = path
        self._device = device
        self._max_length = max_length
        self._kenning: Any = None
        cfg = read_config(path) if _P(path).is_dir() else {}
        name = cfg.get("name") or _P(path).name or path
        self.name = f"kenning ({name})"

    def _ensure_loaded(self) -> Any:
        if self._kenning is None:
            from systemone_builder.kenning.model import Kenning

            self._kenning = Kenning(path=self._path, device=self._device, max_length=self._max_length)
        return self._kenning

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        k = await asyncio.to_thread(self._ensure_loaded)
        return await asyncio.to_thread(k.answer, req)

    async def aclose(self) -> None:
        if self._kenning is not None:
            import gc

            import torch

            self._kenning = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DirectKenningXLEngine(SystemOneEngine):
    """In-process decoder LM (KenningXL) single-pass logit readout engine."""

    def __init__(self, path: str, device: str | None = None, deliberate: bool = False) -> None:
        from pathlib import Path as _P

        self._path = path
        self._device = device
        self._deliberate = deliberate
        self._xl: Any = None
        short = _P(path).name or path.split("/")[-1]
        self.name = f"xl ({short}{'-deliberate' if deliberate else ''})"

    def _ensure_loaded(self) -> Any:
        if self._xl is None:
            from systemone_builder.kenning.xl import KenningXL

            self._xl = KenningXL(path=self._path, device=self._device)
        return self._xl

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        xl = await asyncio.to_thread(self._ensure_loaded)
        return await asyncio.to_thread(xl.system_one, req, self._deliberate)

    async def aclose(self) -> None:
        if self._xl is not None:
            import gc

            import torch

            self._xl = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DirectClefEngine(SystemOneEngine):
    """In-process Cloudflare Clef engine (no HTTP container needed)."""

    def __init__(self, repo: str = "Cloudflare/clef-flash") -> None:
        self._repo = repo
        self._clef: Any = None
        self.name = f"clef ({repo.split('/')[-1]})"

    def _ensure_loaded(self) -> Any:
        if self._clef is None:
            import os

            from systemone_builder.kenning import clef_serve

            os.environ["CLEF_MODEL"] = self._repo
            if not clef_serve._state.get("model"):
                clef_serve.load()
            self._clef = clef_serve
        return self._clef

    def _sync_answer(self, req: SystemOneRequest) -> SystemOneResponse:
        clef = self._ensure_loaded()
        body = {
            "state": req.state,
            "model": req.model or clef._state.get("repo", "clef-flash"),
            "questions": {k: clef._clef_question(q.model_dump(exclude_none=True)) for k, q in req.questions.items()},
        }
        out = clef._answer(body)
        return SystemOneResponse.model_validate(out)

    async def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        return await asyncio.to_thread(self._sync_answer, req)

    async def aclose(self) -> None:
        if self._clef is not None:
            import gc

            import torch

            self._clef._state.clear()
            self._clef = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



