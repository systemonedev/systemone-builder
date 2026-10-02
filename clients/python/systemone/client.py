"""HTTP clients for servers speaking the System One wire format (``POST /v1/systemone``).

    from systemone import Client, Noul, Choice

    with Client("http://localhost:8093") as client:          # a Kenning server
        r = client.system_one(state={"ticket": "I was charged twice."},
                              questions={"billing": Noul("Is this about billing?"),
                                         "team": Choice("Which team?", {"billing": None, "technical": None})})
    r.nouls["billing"].noul, r.choices["team"].choice

``base_url`` is the server root: ``http://localhost:8093`` for a Kenning server,
``http://localhost:8090/api`` for SystemOne Builder's API (with ``api_key``).
Both default to ``SYSTEMONE_BASE_URL`` / ``SYSTEMONE_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

from systemone.types import Question, Response, question_dict

DEFAULT_BASE_URL = "http://localhost:8093"


class SystemOneError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _headers(api_key: str | None) -> dict[str, str]:
    # Builder's API reads X-API-Key; servers that use bearer tokens read Authorization.
    return {"X-API-Key": api_key, "Authorization": f"Bearer {api_key}"} if api_key else {}


def _body(state: Any, questions: Mapping[str, Question | dict[str, Any]], model: str | None) -> dict[str, Any]:
    if not questions:
        raise ValueError("ask at least one question")
    body: dict[str, Any] = {"state": state, "questions": {k: question_dict(q) for k, q in questions.items()}}
    if model:
        body["model"] = model
    return body


def _parse(r: httpx.Response) -> Response:
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise SystemOneError(r.status_code, str(detail)[:500])
    return Response.from_wire(r.json())


class Client:
    """Synchronous client. Reuse one instance: it keeps connections alive."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0,
                 model: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("SYSTEMONE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout,
                                  headers=_headers(api_key or os.environ.get("SYSTEMONE_API_KEY")))

    def system_one(self, state: Any, questions: Mapping[str, Question | dict[str, Any]],
                   model: str | None = None) -> Response:
        return _parse(self._http.post("/v1/systemone", json=_body(state, questions, model or self.model)))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncClient:
    """Asynchronous client for asyncio code."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0,
                 model: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("SYSTEMONE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout,
                                       headers=_headers(api_key or os.environ.get("SYSTEMONE_API_KEY")))

    async def system_one(self, state: Any, questions: Mapping[str, Question | dict[str, Any]],
                         model: str | None = None) -> Response:
        return _parse(await self._http.post("/v1/systemone", json=_body(state, questions, model or self.model)))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
