"""Rolling replay buffer backed by the 120GB RAM datastore (Module A).

The buffer keeps the last ``capacity`` actions (10,000 by default) and their
state deltas as a continuous, monotonically-sequenced log. Each record is
stored in a Redis hash keyed by sequence number and the ordering lives in a
sorted set, so the dashboard scrubber can seek to any position in O(log n).
Appends and trimming happen atomically in a Lua script.
"""

from __future__ import annotations

import abc
import json
from typing import Any

from systemone.contracts.replay import Outcome, ReplayRecord


class ReplayBuffer(abc.ABC):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    @abc.abstractmethod
    async def append(self, record: ReplayRecord) -> ReplayRecord: ...

    @abc.abstractmethod
    async def get(self, seq: int) -> ReplayRecord | None: ...

    @abc.abstractmethod
    async def update(self, seq: int, **fields: Any) -> ReplayRecord | None: ...

    @abc.abstractmethod
    async def range(self, start_seq: int | None = None, limit: int = 100, reverse: bool = False) -> list[ReplayRecord]:
        """Records with ``seq >= start_seq`` (or ``<=`` when ``reverse``)."""

    @abc.abstractmethod
    async def bounds(self) -> tuple[int | None, int | None]:
        """(oldest seq, newest seq) currently retained."""

    @abc.abstractmethod
    async def size(self) -> int: ...

    @abc.abstractmethod
    async def clear(self) -> None: ...

    async def latest(self, limit: int = 100) -> list[ReplayRecord]:
        return await self.range(None, limit=limit, reverse=True)

    async def at_offset(self, offset: int) -> ReplayRecord | None:
        """Scrubber helper: ``offset`` 0 = newest, ``size-1`` = oldest."""
        lo, hi = await self.bounds()
        if hi is None:
            return None
        seq = hi - offset
        # sequences are dense inside the window, but guard anyway
        recs = await self.range(seq, limit=1, reverse=True)
        return recs[0] if recs and recs[0].seq is not None and recs[0].seq >= (lo or 0) else None

    async def outcome_counts(self) -> dict[str, int]:
        counts = {o.value: 0 for o in Outcome}
        for rec in await self.latest(self.capacity):
            counts[rec.outcome.value] += 1
        return counts

    async def stats(self) -> dict[str, Any]:
        lo, hi = await self.bounds()
        return {"size": await self.size(), "capacity": self.capacity, "oldest_seq": lo, "newest_seq": hi}


# Lua keeps append+trim atomic across concurrent API workers.
_APPEND_LUA = """
local seq = redis.call('INCR', KEYS[1])
redis.call('HSET', KEYS[2], seq, ARGV[1])
redis.call('ZADD', KEYS[3], seq, seq)
local cap = tonumber(ARGV[2])
local n = redis.call('ZCARD', KEYS[3])
if n > cap then
  local old = redis.call('ZRANGE', KEYS[3], 0, n - cap - 1)
  for _, s in ipairs(old) do redis.call('HDEL', KEYS[2], s) end
  redis.call('ZREMRANGEBYRANK', KEYS[3], 0, n - cap - 1)
end
return seq
"""


class RedisReplayBuffer(ReplayBuffer):
    def __init__(self, client: Any, capacity: int = 10_000, namespace: str = "s1") -> None:
        super().__init__(capacity)
        self.r = client
        self.k_seq = f"{namespace}:replay:seq"
        self.k_data = f"{namespace}:replay:data"
        self.k_index = f"{namespace}:replay:index"
        self._script = None

    async def append(self, record: ReplayRecord) -> ReplayRecord:
        payload = record.model_dump(mode="json", exclude={"seq"})
        if self._script is None:
            self._script = self.r.register_script(_APPEND_LUA)
        seq = await self._script(keys=[self.k_seq, self.k_data, self.k_index], args=[json.dumps(payload), self.capacity])
        return record.model_copy(update={"seq": int(seq)})

    @staticmethod
    def _decode(seq: Any, raw: Any) -> ReplayRecord | None:
        # The payload is stored without its seq (the hash field is the seq).
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        rec = ReplayRecord.model_validate_json(raw)
        return rec.model_copy(update={"seq": int(seq)})

    async def get(self, seq: int) -> ReplayRecord | None:
        return self._decode(seq, await self.r.hget(self.k_data, str(seq)))

    async def update(self, seq: int, **fields: Any) -> ReplayRecord | None:
        rec = await self.get(seq)
        if rec is None:
            return None
        rec = rec.model_copy(update=fields)
        # Only write if still retained (avoid resurrecting trimmed records).
        if await self.r.zscore(self.k_index, str(seq)) is not None:
            await self.r.hset(self.k_data, str(seq), rec.model_dump_json(exclude={"seq"}))
        return rec

    async def range(self, start_seq: int | None = None, limit: int = 100, reverse: bool = False) -> list[ReplayRecord]:
        if reverse:
            hi = "+inf" if start_seq is None else start_seq
            seqs = await self.r.zrevrangebyscore(self.k_index, hi, "-inf", start=0, num=limit)
        else:
            lo = "-inf" if start_seq is None else start_seq
            seqs = await self.r.zrangebyscore(self.k_index, lo, "+inf", start=0, num=limit)
        if not seqs:
            return []
        raws = await self.r.hmget(self.k_data, seqs)
        return [r for r in (self._decode(s, x) for s, x in zip(seqs, raws)) if r is not None]

    async def bounds(self) -> tuple[int | None, int | None]:
        first = await self.r.zrange(self.k_index, 0, 0)
        last = await self.r.zrange(self.k_index, -1, -1)
        if not first:
            return None, None
        return int(first[0]), int(last[0])

    async def size(self) -> int:
        return int(await self.r.zcard(self.k_index))

    async def clear(self) -> None:
        await self.r.delete(self.k_data, self.k_index)
