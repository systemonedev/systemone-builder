"""Two-stage cascade inference wrapper for Kenning cross-encoder models.

Pass 1 runs only the primary binary gate question (2 candidate pairs, ~18-35 ms).
Pass 2 runs the remaining diagnostic questions (choice/score/secondary nouls)
only when Pass 1's probability falls in the uncertain band ``lo < p < hi``.
"""

from __future__ import annotations

import time
from typing import Any

from systemone_builder.kenning.model import Kenning
from systemone_builder.system_one.contract import (
    NoulAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


class CascadeKenning:
    """Two-stage cascade over a single Kenning cross-encoder."""

    def __init__(
        self,
        path: str,
        device: str | None = None,
        max_length: int | None = None,
        gate_qid: str = "is_phish",
        lo: float = 0.05,
        hi: float = 0.92,
    ) -> None:
        self.kenning = Kenning(path=path, device=device, max_length=max_length)
        self.name = f"cascade/{self.kenning.name}"
        self.gate_qid = gate_qid
        self.lo = lo
        self.hi = hi

    def _pick_gate(self, req: SystemOneRequest) -> str | None:
        if self.gate_qid in req.questions and req.questions[self.gate_qid].type == "noul":
            return self.gate_qid
        for qid, q in req.questions.items():
            if q.type == "noul":
                return qid
        return None

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        gate = self._pick_gate(req)
        if gate is None or len(req.questions) <= 1:
            resp = self.kenning.answer(req)
            resp.model = self.name
            return resp

        t0 = time.perf_counter()
        pass1_req = SystemOneRequest(state=req.state, model=req.model, questions={gate: req.questions[gate]})
        r1 = self.kenning.answer(pass1_req)
        answers: dict[str, Any] = dict(r1.answers)
        in_toks = r1.usage.input_tokens
        out_toks = r1.usage.output_tokens

        gate_ans = NoulAnswer.model_validate(answers[gate])
        if self.lo < gate_ans.noul < self.hi:
            rest_qs = {k: q for k, q in req.questions.items() if k != gate}
            pass2_req = SystemOneRequest(state=req.state, model=req.model, questions=rest_qs)
            r2 = self.kenning.answer(pass2_req)
            answers.update(r2.answers)
            in_toks += r2.usage.input_tokens
            out_toks += r2.usage.output_tokens

        return SystemOneResponse(
            model=self.name,
            answers=answers,
            usage=Usage(input_tokens=in_toks, output_tokens=out_toks),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
