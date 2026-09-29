"""FastAPI application factory for the LAN REST API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from systemone import __version__
from systemone.api.deps import require_api_key
from systemone.api.routes import byom, dashboard, extraction, feedback, replay, routing, system, training
from systemone.config import Settings, get_settings
from systemone.runtime import Runtime

API_PREFIX = "/api/v1"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
WEAK_KEYS = {"", "change-me", "changeme", "secret", "password"}

log = logging.getLogger("systemone.api")


def check_exposure(settings: Settings) -> None:
    """Refuse to serve beyond loopback without a real API key.

    The API holds the Docker socket (root-equivalent on the Docker host/VM), so
    publishing it on the LAN with no or a placeholder key would hand that out.
    """
    key = settings.api_key or ""
    weak = key.strip().lower() in WEAK_KEYS or len(key) < 16
    if settings.bind_addr not in LOOPBACK and weak:
        raise RuntimeError(
            f"S1_BIND_ADDR={settings.bind_addr} exposes the API beyond this host but S1_API_KEY is "
            "missing or weak. Set S1_API_KEY to a random secret (openssl rand -hex 32) or bind to 127.0.0.1."
        )
    if weak:
        log.warning("S1_API_KEY is missing or weak; acceptable only while the API is bound to loopback")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    check_exposure(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rt = Runtime(settings)
        app.state.rt = rt
        await rt.startup()
        try:
            yield
        finally:
            await rt.shutdown()

    app = FastAPI(title="systemone-builder", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system.public_router, prefix=API_PREFIX)
    app.include_router(dashboard.ws_router, prefix=API_PREFIX)  # auth checked in-handler
    auth = [Depends(require_api_key)]
    for module in (system, replay, extraction, routing, training, feedback, dashboard, byom):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=auth)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    # Single worker: the lifecycle lock, oracle waiters and WebSocket bus are in-process.
    uvicorn.run(create_app(s), host=s.api_host, port=s.api_port, log_level="info", ws_ping_interval=20)


if __name__ == "__main__":
    main()
