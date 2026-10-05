"""Message bus.

Demo profile: an in-process asyncio fan-out bus. Production profile swaps in a
Kafka/Redpanda implementation with the same two methods, keyed by entity so one
entity's signals stay ordered within a partition.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Protocol

log = logging.getLogger(__name__)

TOPIC_RAW = "raw.docs"
TOPIC_SIGNALS = "enriched.signals"
TOPIC_DEAD_LETTER = "dead.letter"


class Bus(Protocol):
    async def publish(self, topic: str, message: Any) -> None: ...

    def subscribe(self, topic: str, maxsize: int | None = None) -> asyncio.Queue[Any]: ...

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None: ...


class InProcessBus:
    """Fan-out bus: every subscriber of a topic gets every message.

    Slow subscribers with a bounded queue lose their *oldest* message rather than
    blocking the producer; the drop is counted so it shows on the health page.
    """

    def __init__(self, default_maxsize: int = 10_000) -> None:
        self._subs: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)
        self._default_maxsize = default_maxsize
        self.published: dict[str, int] = defaultdict(int)
        self.dropped: dict[str, int] = defaultdict(int)

    async def publish(self, topic: str, message: Any) -> None:
        self.published[topic] += 1
        for q in list(self._subs.get(topic, ())):
            if q.full():
                try:
                    q.get_nowait()
                    self.dropped[topic] += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(message)

    def subscribe(self, topic: str, maxsize: int | None = None) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize or self._default_maxsize)
        self._subs[topic].append(q)
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        if queue in self._subs.get(topic, []):
            self._subs[topic].remove(queue)

    def lag(self) -> dict[str, int]:
        """Messages waiting per topic, summed over subscribers (consumer lag)."""
        return {t: sum(q.qsize() for q in qs) for t, qs in self._subs.items()}
