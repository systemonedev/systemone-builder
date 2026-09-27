from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, WebSocket, status

from systemone.runtime import Runtime


def get_rt(request: Request) -> Runtime:
    return request.app.state.rt


def _check_key(expected: str | None, provided: str | None) -> bool:
    return expected is None or (provided is not None and secrets.compare_digest(expected, provided))


async def require_api_key(request: Request) -> None:
    """LAN auth: when ``S1_API_KEY`` is set every request needs ``X-API-Key``."""
    expected = request.app.state.rt.settings.api_key
    if not _check_key(expected, request.headers.get("x-api-key")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


def ws_authorized(ws: WebSocket) -> bool:
    expected = ws.app.state.rt.settings.api_key
    return _check_key(expected, ws.headers.get("x-api-key") or ws.query_params.get("api_key"))
