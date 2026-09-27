"""Tests run against real services only.

Redis-backed tests need a reachable Redis (``S1_TEST_REDIS_URL``, default
``redis://localhost:6379/15``) and are skipped otherwise. Hardware paths
(NVML, Docker, vLLM, Ollama) are exercised on the target rig, not here.
"""

from __future__ import annotations

import os

import pytest

from systemone.datastore.store import create_redis

TEST_REDIS_URL = os.environ.get("S1_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def redis_client():
    client = create_redis(TEST_REDIS_URL)
    try:
        await client.ping()
    except Exception:
        pytest.skip(f"Redis not reachable at {TEST_REDIS_URL}")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()
