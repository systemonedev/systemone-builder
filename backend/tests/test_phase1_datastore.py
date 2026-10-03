from __future__ import annotations

import asyncio

from systemone_builder.contracts.replay import Outcome, ReplayRecord
from systemone_builder.datastore.replay_buffer import RedisReplayBuffer
from systemone_builder.datastore.store import JsonStore
from systemone_builder.telemetry.bus import EventBus


def rec(i: int) -> ReplayRecord:
    return ReplayRecord(domain="computer_use", state={"i": i}, action={"action": "CLICK"})


async def test_replay_buffer_rolls_at_capacity(redis_client):
    buf = RedisReplayBuffer(redis_client, capacity=50)
    for i in range(120):
        await buf.append(rec(i))
    assert await buf.size() == 50
    assert await buf.bounds() == (71, 120)
    latest = await buf.latest(3)
    assert [r.seq for r in latest] == [120, 119, 118]
    assert latest[0].state == {"i": 119} and latest[0].route_path == []
    assert await buf.get(10) is None
    assert [r.seq for r in await buf.range(100, limit=3)] == [100, 101, 102]
    assert (await buf.at_offset(0)).seq == 120
    assert (await buf.at_offset(49)).seq == 71
    await buf.update(115, outcome=Outcome.FAILURE, delta={"added": ["error"]})
    assert (await buf.get(115)).delta == {"added": ["error"]}
    assert (await buf.outcome_counts())["failure"] == 1


async def test_replay_buffer_concurrent_appends_unique_seqs(redis_client):
    buf = RedisReplayBuffer(redis_client, capacity=100)
    out = await asyncio.gather(*(buf.append(rec(i)) for i in range(300)))
    assert len({r.seq for r in out}) == 300
    assert await buf.size() == 100


async def test_json_store_queue_and_hash(redis_client):
    st = JsonStore(redis_client, "t")
    for i in range(5):
        await st.push(("q",), {"i": i})
    assert [x["i"] for x in await st.pop_many(("q",), 3)] == [0, 1, 2]
    assert await st.qlen(("q",)) == 2
    await st.hset(("h",), "a", {"x": 1})
    assert await st.hgetall(("h",)) == {"a": {"x": 1}}
    assert (await st.memory_info())["used_memory"] > 0


async def test_event_bus_fanout():
    bus = EventBus()
    q = bus.subscribe()
    bus.publish("routing", "escalate", tier="triage")
    ev = await q.get()
    assert ev["channel"] == "routing" and ev["data"]["tier"] == "triage"


def test_api_refuses_lan_bind_without_real_key():
    import pytest

    from systemone_builder.api.app import check_exposure
    from systemone_builder.config import Settings

    check_exposure(Settings(bind_addr="127.0.0.1", api_key=None))  # loopback: warning only
    with pytest.raises(RuntimeError, match="S1_API_KEY"):
        check_exposure(Settings(bind_addr="0.0.0.0", api_key="change-me"))
    with pytest.raises(RuntimeError):
        check_exposure(Settings(bind_addr="0.0.0.0", api_key=None))
    check_exposure(Settings(bind_addr="0.0.0.0", api_key="3f1c9a0b7d2e4f6a8b1c3d5e7f9a0b2c"))
