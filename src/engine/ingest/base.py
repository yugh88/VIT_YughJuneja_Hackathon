"""Connector contract, circuit breaker and per-source health.

A connector only knows how to turn one HTTP poll into ``RawDoc`` objects. Retries,
back-off, de-duplication of already-seen items, fallback to the recorded replay
twin and health accounting all live in :class:`SourceRunner`, so every source gets
identical resilience for free.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import httpx

from engine.schema import RawDoc, SourceType

log = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures, then waits with
    exponential back-off (``base_backoff_s`` doubling to ``max_backoff_s``) before
    letting a single trial request through (half-open)."""

    def __init__(
        self,
        failure_threshold: int = 2,
        base_backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self._clock = clock
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self.opened_count = 0
        self._backoff_s = base_backoff_s
        self._open_until = 0.0

    def allow(self) -> bool:
        if self.state is BreakerState.OPEN:
            if self._clock() >= self._open_until:
                self.state = BreakerState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self._backoff_s = self.base_backoff_s

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_count += 1
            self._open_until = self._clock() + self._backoff_s
            self._backoff_s = min(self._backoff_s * 2, self.max_backoff_s)

    @property
    def retry_in_s(self) -> float:
        if self.state is not BreakerState.OPEN:
            return 0.0
        return max(0.0, self._open_until - self._clock())


@dataclass
class SourceHealth:
    name: str
    source_type: str
    configured_mode: str  # "live" | "replay" | "replay-only"
    active_mode: str = "starting"  # "live" | "replay" | "fallback" | "down"
    docs_total: int = 0
    polls_ok: int = 0
    polls_failed: int = 0
    last_doc_at: float | None = None
    last_ok_at: float | None = None
    last_error: str | None = None
    note: str | None = None
    _recent: deque[float] = field(default_factory=lambda: deque(maxlen=50_000), repr=False)

    def record_doc(self, now: float | None = None) -> None:
        now = now or time.time()
        self.docs_total += 1
        self.last_doc_at = now
        self._recent.append(now)

    def docs_per_min(self, now: float | None = None) -> int:
        now = now or time.time()
        while self._recent and self._recent[0] < now - 60:
            self._recent.popleft()
        return len(self._recent)

    def as_dict(self, breaker: CircuitBreaker | None = None) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["docs_per_min"] = self.docs_per_min()
        if breaker is not None:
            d["breaker"] = {
                "state": breaker.state.value,
                "consecutive_failures": breaker.consecutive_failures,
                "opened_count": breaker.opened_count,
                "retry_in_s": round(breaker.retry_in_s, 1),
            }
        return d


class Source(ABC):
    """A live, keyless connector. Subclasses implement :meth:`fetch` only."""

    name: str
    source_type: SourceType
    poll_interval_s: float = 120.0

    def __init__(self, name: str, poll_interval_s: float | None = None, **_: Any) -> None:
        self.name = name
        if poll_interval_s is not None:
            self.poll_interval_s = poll_interval_s

    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        """One poll. Raise on transport/parse failure; return [] when nothing new."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources held between polls (none by default)."""


class SeenCache:
    """Bounded LRU of doc ids so a feed re-polled every minute is not re-emitted."""

    def __init__(self, maxsize: int = 100_000) -> None:
        self._d: OrderedDict[str, None] = OrderedDict()
        self.maxsize = maxsize

    def add(self, key: str) -> bool:
        """Return True if the key is new."""
        if key in self._d:
            self._d.move_to_end(key)
            return False
        self._d[key] = None
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._d)


Emit = Callable[[RawDoc], Any]


class ReplayFeed(Protocol):
    """A recorded twin: yields (seconds to wait, doc) and re-stamps docs on emission."""

    def __iter__(self) -> Iterator[tuple[float, RawDoc]]: ...

    def stamp(self, doc: RawDoc) -> RawDoc: ...


class SourceRunner:
    """Drives one source: live polling with breaker, or replay, or live-with-fallback.

    While the breaker is open the runner serves the source's replay twin, so the
    dashboard never goes quiet; once the breaker lets a trial through and it
    succeeds, the source is live again.
    """

    def __init__(
        self,
        source: Source | None,
        name: str,
        source_type: SourceType,
        emit: Emit,
        replay_factory: Callable[[], ReplayFeed] | None,
        live: bool,
        client: httpx.AsyncClient | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        if live and (source is None or client is None):
            raise ValueError(f"{name}: live mode needs a connector and an HTTP client")
        self.source = source
        self.name = name
        self.emit = emit
        self.client = client
        self.live = live
        self.breaker = breaker or CircuitBreaker()
        self.replay_factory = replay_factory
        self._replay_feed: ReplayFeed | None = None
        self._replay_iter: Iterator[tuple[float, RawDoc]] | None = None
        self._pending: tuple[float, RawDoc] | None = None
        self.seen = SeenCache()
        self.killed = False  # set by the "kill a source" resilience demo
        self.replay_covered_by: str | None = None
        configured = "replay-only" if source is None else ("live" if live else "replay")
        self.health = SourceHealth(name=name, source_type=source_type.value, configured_mode=configured)

    async def _emit(self, doc: RawDoc) -> None:
        if not self.seen.add(doc.doc_id):
            return
        self.health.record_doc()
        res = self.emit(doc)
        if asyncio.iscoroutine(res):
            await res

    async def _replay_for(self, seconds: float | None) -> bool:
        """Emit replayed docs for ``seconds`` (forever if None). False if no twin exists.

        The feed yields (delay, doc) pairs and never sleeps itself, so stopping
        mid-wait (when the live source recovers) loses nothing.
        """
        if self.replay_factory is None:
            return False
        deadline = None if seconds is None else time.monotonic() + seconds
        while True:
            if self._pending is None:
                if self._replay_feed is None:
                    self._replay_feed = self.replay_factory()
                    self._replay_iter = iter(self._replay_feed)
                try:
                    delay, doc = next(self._replay_iter)  # type: ignore[arg-type]
                except StopIteration:
                    self._replay_feed = None
                    if deadline is None:
                        return True  # twin exhausted and not looping
                    await asyncio.sleep(max(0.0, deadline - time.monotonic()))
                    return True
                self._pending = (time.monotonic() + delay, doc)
            due, doc = self._pending
            now = time.monotonic()
            if deadline is not None and due > deadline:
                await asyncio.sleep(max(0.0, deadline - now))
                return True
            if due > now:
                await asyncio.sleep(due - now)
            self._pending = None
            assert self._replay_feed is not None
            await self._emit(self._replay_feed.stamp(doc))

    async def _poll_once(self) -> None:
        assert self.source is not None and self.client is not None
        if self.killed:
            raise ConnectionError("source killed by operator (resilience demo)")
        docs = await self.source.fetch(self.client)
        for d in docs:
            await self._emit(d)

    async def run(self) -> None:
        if not self.live:
            self.health.active_mode = "replay"
            ok = await self._replay_for(None)
            if not ok:
                self.health.active_mode = "covered" if self.replay_covered_by else "down"
                self.health.note = (
                    f"no separate recording; its articles are in the {self.replay_covered_by} replay"
                    if self.replay_covered_by
                    else "no replay twin recorded"
                )
            return

        assert self.source is not None
        while True:
            if self.breaker.allow():
                try:
                    await self._poll_once()
                    self.breaker.record_success()
                    self.health.polls_ok += 1
                    self.health.last_ok_at = time.time()
                    self.health.last_error = None
                    self.health.active_mode = "live"
                    await asyncio.sleep(self.source.poll_interval_s)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.breaker.record_failure()
                    self.health.polls_failed += 1
                    self.health.last_error = f"{type(exc).__name__}: {exc}"[:300]
                    log.warning("source %s poll failed: %s", self.name, self.health.last_error)
            # Breaker open (or a failure below threshold): serve the twin meanwhile.
            wait = max(self.breaker.retry_in_s, 1.0)
            if self.breaker.state is BreakerState.OPEN and self.replay_factory is not None:
                self.health.active_mode = "fallback"
                await self._replay_for(wait)
            else:
                if self.breaker.state is BreakerState.OPEN:
                    self.health.active_mode = "down"
                await asyncio.sleep(wait)

    async def aclose(self) -> None:
        if self.source is not None:
            with contextlib.suppress(Exception):
                await self.source.aclose()
