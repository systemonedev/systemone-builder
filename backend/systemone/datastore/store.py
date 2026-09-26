"""Redis connection factory (120GB RAM datastore) and a namespaced JSON store."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis


def create_redis(url: str) -> Any:
    # A blocking pool makes bursts of concurrent requests wait for a free
    # connection instead of failing with MaxConnectionsError.
    pool = aioredis.BlockingConnectionPool.from_url(url, max_connections=128, timeout=10, health_check_interval=30)
    return aioredis.Redis(connection_pool=pool)


class JsonStore:
    """Namespaced helpers for JSON documents, hashes, queues and counters."""

    def __init__(self, client: Any, namespace: str = "s1") -> None:
        self.r = client
        self.ns = namespace

    def key(self, *parts: str) -> str:
        return ":".join((self.ns, *parts))

    @staticmethod
    def _load(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)

    # documents -------------------------------------------------------------
    async def get(self, *key: str, default: Any = None) -> Any:
        val = self._load(await self.r.get(self.key(*key)))
        return default if val is None else val

    async def set(self, value: Any, *key: str, ttl: int | None = None) -> None:
        await self.r.set(self.key(*key), json.dumps(value), ex=ttl)

    async def delete(self, *key: str) -> None:
        await self.r.delete(self.key(*key))

    # hashes ----------------------------------------------------------------
    async def hset(self, name: tuple[str, ...], field: str, value: Any) -> None:
        await self.r.hset(self.key(*name), field, json.dumps(value))

    async def hget(self, name: tuple[str, ...], field: str) -> Any:
        return self._load(await self.r.hget(self.key(*name), field))

    async def hdel(self, name: tuple[str, ...], field: str) -> None:
        await self.r.hdel(self.key(*name), field)

    async def hgetall(self, name: tuple[str, ...]) -> dict[str, Any]:
        raw = await self.r.hgetall(self.key(*name))
        return {(k.decode() if isinstance(k, bytes) else k): self._load(v) for k, v in raw.items()}

    # queues ----------------------------------------------------------------
    async def push(self, name: tuple[str, ...], value: Any) -> int:
        return int(await self.r.rpush(self.key(*name), json.dumps(value)))

    async def pop_many(self, name: tuple[str, ...], n: int) -> list[Any]:
        k = self.key(*name)
        pipe = self.r.pipeline()
        pipe.lrange(k, 0, n - 1)
        pipe.ltrim(k, n, -1)
        items, _ = await pipe.execute()
        return [self._load(x) for x in items]

    async def peek(self, name: tuple[str, ...], start: int = 0, end: int = -1) -> list[Any]:
        return [self._load(x) for x in await self.r.lrange(self.key(*name), start, end)]

    async def qlen(self, name: tuple[str, ...]) -> int:
        return int(await self.r.llen(self.key(*name)))

    # counters --------------------------------------------------------------
    async def incr(self, *key: str, by: int = 1) -> int:
        return int(await self.r.incrby(self.key(*key), by))

    async def counter(self, *key: str) -> int:
        raw = await self.r.get(self.key(*key))
        return int(raw) if raw is not None else 0

    async def memory_info(self) -> dict[str, Any]:
        info = await self.r.info("memory")
        return {
            "used_memory": info.get("used_memory"),
            "used_memory_human": info.get("used_memory_human"),
            "maxmemory": info.get("maxmemory"),
            "maxmemory_human": info.get("maxmemory_human"),
            "maxmemory_policy": info.get("maxmemory_policy"),
        }
