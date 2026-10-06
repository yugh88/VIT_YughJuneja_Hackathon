"""Circuit breaker and fallback: a failing source must never silence the dashboard."""

from __future__ import annotations

import asyncio

import httpx

from conftest import make_doc, write_twin
from engine.ingest.base import BreakerState, CircuitBreaker, Source, SourceRunner
from engine.ingest.replay import ReplayReader
from engine.schema import RawDoc, SourceType


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_breaker_opens_backs_off_exponentially_and_recovers() -> None:
    clock = FakeClock()
    b = CircuitBreaker(failure_threshold=2, base_backoff_s=1, max_backoff_s=60, clock=clock)
    b.record_failure()
    assert b.state is BreakerState.CLOSED and b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN and not b.allow()
    assert b.retry_in_s == 1
    waits = []
    for _ in range(8):  # each failed trial doubles the wait, capped at 60 s
        clock.t += b.retry_in_s
        assert b.allow() and b.state is BreakerState.HALF_OPEN
        b.record_failure()
        waits.append(b.retry_in_s)
    assert waits == [2, 4, 8, 16, 32, 60, 60, 60]
    clock.t += 60
    assert b.allow()
    b.record_success()
    assert b.state is BreakerState.CLOSED and b.consecutive_failures == 0


class DownSource(Source):
    source_type = SourceType.NEWS
    poll_interval_s = 0.01

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        raise httpx.ConnectError("simulated outage")


async def test_failing_live_source_falls_back_to_replay_twin(tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_twin(tmp_path, "rss", [make_doc(i) for i in range(10)])
    got: list[RawDoc] = []
    async with httpx.AsyncClient() as client:
        runner = SourceRunner(
            source=DownSource("rss"),
            name="rss",
            source_type=SourceType.NEWS,
            emit=got.append,
            replay_factory=lambda: ReplayReader.for_source(tmp_path, "rss", speed=1e6, max_gap_s=0.001),  # type: ignore[arg-type,return-value]
            live=True,
            client=client,
            breaker=CircuitBreaker(failure_threshold=1, base_backoff_s=0.2),
        )
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.15)
        task.cancel()
    assert runner.health.active_mode == "fallback"
    assert runner.health.polls_failed >= 1 and "simulated outage" in (runner.health.last_error or "")
    assert len(got) >= 5 and all(d.replayed for d in got)


async def test_runner_deduplicates_repeated_poll_items() -> None:
    class Repeating(Source):
        source_type = SourceType.NEWS
        poll_interval_s = 0.0

        async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
            return [make_doc(1), make_doc(2)]

    got: list[RawDoc] = []
    async with httpx.AsyncClient() as client:
        runner = SourceRunner(Repeating("rss"), "rss", SourceType.NEWS, got.append, None, True, client)
        for _ in range(3):
            await runner._poll_once()
    assert len(got) == 2 and runner.health.docs_total == 2
