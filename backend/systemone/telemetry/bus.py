"""In-process pub/sub event bus.

Every subsystem publishes structured events (inference, routing, lifecycle,
training, factory, dpo, eval). The WebSocket gateway fans them out to the
dashboard; tests subscribe directly.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any


class EventBus:
    def __init__(self, history: int = 500, queue_size: int = 1000) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._queue_size = queue_size
        self.history: deque[dict[str, Any]] = deque(maxlen=history)

    def publish(self, channel: str, type_: str, **data: Any) -> dict[str, Any]:
        event = {"channel": channel, "type": type_, "ts": time.time(), "data": data}
        self.history.append(event)
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # slow consumer: drop its oldest event rather than block producers
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        return event

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_size)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    def recent(self, channel: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        items = [e for e in self.history if channel is None or e["channel"] == channel]
        return items[-limit:]
