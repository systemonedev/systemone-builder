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
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    choice_answer,
    noul_answer,
    score_answer,
)


def _fast_path_answer(q: Question, p_gate: float, state: Any = None) -> Any:
    """Deterministic structural fast-path answer when Pass 1 gate is already outside ``(lo, hi)``."""
    is_phish = p_gate >= 0.5
    p_hi = min(max(p_gate if is_phish else (1.0 - p_gate), 0.5), 0.9999)
    links = state.get("links") or [] if isinstance(state, dict) else []
    attachments = state.get("attachments") or [] if isinstance(state, dict) else []
    signals = state.get("signals") or {} if isinstance(state, dict) else {}

    if q.type == "noul":
        instr = q.instructions.lower()
        if "identity" in instr or "brand" in instr:
            return noul_answer(0.88, 0.12)
        if "urgen" in instr or "time pressure" in instr:
            return noul_answer(0.85 if is_phish else 0.15, 0.15 if is_phish else 0.85)
        return noul_answer(p_gate, 1.0 - p_gate)
    if q.type == "choice":
        opts = list(q.criteria)  # type: ignore[arg-type]
        pick: str | None = None
        if "quarantine" in opts and "deliver" in opts:
            pick = "quarantine" if is_phish else "deliver"
        elif "phishing" in opts and "legitimate" in opts:
            pick = "phishing" if is_phish else "legitimate"
        elif "click_link" in opts and "open_attachment" in opts:
            if attachments:
                pick = "open_attachment"
            elif links:
                pick = "click_link"
            else:
                pick = "reply_with_info" if (is_phish and "reply_with_info" in opts) else ("none" if "none" in opts else opts[0])
        elif "credentials" in opts and "nothing" in opts:
            if not is_phish:
                pick = "document_access" if (signals.get("is_saas_notification") and "document_access" in opts) else "nothing"
            else:
                if signals.get("risky_attachment") and "software_install" in opts:
                    pick = "software_install"
                elif signals.get("is_saas_notification") and "document_access" in opts:
                    pick = "document_access"
                else:
                    pick = "credentials"
        if pick is not None:
            rest = (1.0 - p_hi) / max(len(opts) - 1, 1)
            probs = [p_hi if o == pick else rest for o in opts]
            return choice_answer(opts, probs)
        return choice_answer(opts, [1.0 / len(opts)] * len(opts))
    levels = list(q.criteria)  # type: ignore[arg-type]
    return score_answer(levels, [1.0 / len(levels)] * len(levels))


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
        rest_qs = {k: q for k, q in req.questions.items() if k != gate}
        if self.lo < gate_ans.noul < self.hi:
            pass2_req = SystemOneRequest(state=req.state, model=req.model, questions=rest_qs)
            r2 = self.kenning.answer(pass2_req)
            answers.update(r2.answers)
            in_toks += r2.usage.input_tokens
            out_toks += r2.usage.output_tokens
        else:
            for k, q in rest_qs.items():
                answers[k] = _fast_path_answer(q, gate_ans.noul, state=req.state)

        return SystemOneResponse(
            model=self.name,
            answers=answers,
            usage=Usage(input_tokens=in_toks, output_tokens=out_toks),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
