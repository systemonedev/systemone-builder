"""FastAPI application factory for the LAN REST API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from systemone import __version__
from systemone.api.deps import require_api_key
from systemone.api.routes import extraction, replay, routing, system, training
from systemone.config import Settings, get_settings
from systemone.runtime import Runtime

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

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
    auth = [Depends(require_api_key)]
    for module in (system, replay, extraction, routing, training):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=auth)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    uvicorn.run(create_app(s), host=s.api_host, port=s.api_port, log_level="info")


if __name__ == "__main__":
    main()
