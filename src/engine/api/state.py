"""Process-wide runtime state wired up by the FastAPI lifespan."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engine.aggregate.aggregator import Aggregator
from engine.aggregate.store import SignalStore
from engine.api.metrics import Metrics
from engine.bus import TOPIC_DEAD_LETTER, TOPIC_RAW, TOPIC_SIGNALS, InProcessBus
from engine.config import RunMode, Settings
from engine.ingest.manager import IngestManager
from engine.schema import RawDoc

if TYPE_CHECKING:
    from engine.nlp.pipeline import Pipeline

log = logging.getLogger(__name__)


@dataclass
class RawDocMonitor:
    """Keeps the most recent raw docs and per-language counts for the dashboard."""

    maxlen: int = 2000
    recent: deque[RawDoc] = field(default_factory=deque)
    languages: Counter[str] = field(default_factory=Counter)
    total: int = 0

    def add(self, doc: RawDoc) -> None:
        self.recent.append(doc)
        while len(self.recent) > self.maxlen:
            self.recent.popleft()
        self.total += 1
        if doc.language:
            self.languages[doc.language] += 1


@dataclass
class AppState:
    settings: Settings
    bus: InProcessBus
    ingest: IngestManager
    metrics: Metrics
    monitor: RawDocMonitor
    started_at: float = field(default_factory=time.time)
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    store: SignalStore = field(default_factory=lambda: SignalStore(None))
    aggregator: Aggregator = field(default_factory=Aggregator)
    rebalancer: Any = None  # modules.rebalancer.engine.Rebalancer (Module A)
    stress: Any = None  # modules.stress.engine.StressEngine (Module B)
    pipeline: Pipeline | None = None
    pipeline_error: str | None = None
    dead_letters: int = 0

    @classmethod
    def build(cls, settings: Settings) -> AppState:
        bus = InProcessBus(settings.bus_queue_size)
        scale = settings.replay_speed if settings.mode is RunMode.REPLAY else 1.0
        return cls(
            settings=settings,
            bus=bus,
            ingest=IngestManager(settings, bus),
            metrics=Metrics(),
            monitor=RawDocMonitor(maxlen=settings.recent_buffer),
            store=SignalStore(
                settings.output_dir if settings.write_files else None, buffer=settings.signal_buffer
            ),
            aggregator=Aggregator(time_scale=scale),
        )

    async def _consume_raw(self) -> None:
        q = self.bus.subscribe(TOPIC_RAW)
        while True:
            doc: RawDoc = await q.get()
            self.monitor.add(doc)
            self.metrics.docs_ingested.labels(source=doc.source).inc()

    async def load_pipeline(self) -> None:
        """Load models off the event loop; on failure the API still serves ingestion."""
        from engine.nlp.pipeline import get_pipeline

        scale = self.settings.replay_speed if self.settings.mode is RunMode.REPLAY else 1.0
        try:
            pipe = await asyncio.to_thread(get_pipeline, self.settings.zero_shot, scale)
            await asyncio.to_thread(pipe.reset_state)
            self.pipeline = pipe
            log.info("NLP pipeline ready: %s", self.pipeline.modes)
        except Exception as exc:  # pragma: no cover - environment specific
            self.pipeline_error = f"{type(exc).__name__}: {exc}"
            log.exception("NLP pipeline failed to load")

    @property
    def time_scale(self) -> float:
        return self.aggregator.time_scale

    async def _run_batch(self, batch: list[RawDoc]) -> None:
        assert self.pipeline is not None
        try:
            traces = await asyncio.to_thread(
                self.pipeline.process_batch, batch, True, self.settings.zero_shot_stream
            )
        except Exception:
            if len(batch) > 1:  # isolate the poison doc, keep the rest
                for d in batch:
                    await self._run_batch([d])
                return
            self.dead_letters += 1
            self.metrics.dead_letters.labels(stage="nlp").inc()
            await self.bus.publish(TOPIC_DEAD_LETTER, batch[0])
            log.exception("doc %s sent to dead-letter queue", batch[0].doc_id)
            return
        for tr in traces:
            if tr.dropped == "duplicate":
                self.metrics.duplicates.inc()
            for sig in tr.signals:
                if not self.store.add(sig):
                    continue  # idempotent: a signal id already seen is never counted twice
                self.aggregator.add(sig, author=tr.doc.author, text=tr.clean.text)
                await self._modules_on_signal(sig)
                self.metrics.signals_emitted.labels(event=sig.event.primary.value).inc()
                self.metrics.pipeline_latency.observe(sig.latency_ms / 1000)
                await self.bus.publish(TOPIC_SIGNALS, sig)

    async def _nlp_worker(self) -> None:
        await self.load_pipeline()
        if self.pipeline is None:
            return
        q = self.bus.subscribe(TOPIC_RAW, maxsize=self.settings.bus_queue_size)
        loop = asyncio.get_running_loop()
        size = self.settings.nlp_batch_size
        while True:
            batch = [await q.get()]
            deadline = loop.time() + 0.1  # micro-batch: up to `size` docs or 100 ms
            while len(batch) < size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    # asyncio.timeout, not wait_for: on 3.11 wait_for can swallow a cancellation
                    # that races with a completed get(), which hung shutdown.
                    async with asyncio.timeout(remaining):
                        batch.append(await q.get())
                except TimeoutError:
                    break
            await self._run_batch(batch)

    def init_modules(self) -> None:
        from modules.rebalancer.engine import Rebalancer
        from modules.stress.engine import StressEngine

        try:
            self.rebalancer = Rebalancer(self.aggregator, time_scale=self.time_scale)
        except FileNotFoundError:
            log.warning("Module A disabled: data/reference/universe.csv missing")
        try:
            self.stress = StressEngine(time_scale=self.time_scale)
        except FileNotFoundError:
            log.warning("Module B disabled: synthetic book or shock library missing")

    async def _modules_on_signal(self, sig: Any) -> None:
        if self.rebalancer is not None:
            self.rebalancer.on_signal(sig)
        if self.stress is not None and sig.impact.score > self.stress.min_impact:
            # Monte Carlo repricing takes ~0.2 s: keep it off the event loop.
            await asyncio.to_thread(self.stress.on_signal, sig)

    async def _module_timer(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.rebalancer is not None:
                self.rebalancer.on_timer(time.time())

    async def _flush_files(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await asyncio.to_thread(self.store.flush)
            except OSError:
                log.exception("signal sink write failed")

    async def _refresh_gauges(self) -> None:
        while True:
            for topic, lag in self.bus.lag().items():
                self.metrics.bus_lag.labels(topic=topic).set(lag)
            for h in self.ingest.health():
                up = (
                    1.0
                    if h["active_mode"] in {"live", "replay", "fallback"} and h["docs_per_min"] > 0
                    else 0.0
                )
                self.metrics.source_up.labels(source=h["name"], mode=h["active_mode"]).set(up)
            await asyncio.sleep(5)

    async def start(self) -> None:
        self.tasks.append(asyncio.create_task(self._consume_raw(), name="raw-monitor"))
        self.tasks.append(asyncio.create_task(self._refresh_gauges(), name="gauges"))
        self.tasks.append(asyncio.create_task(self._flush_files(), name="sink"))
        self.init_modules()
        self.tasks.append(asyncio.create_task(self._module_timer(), name="modules"))
        if self.settings.pipeline_enabled:
            self.tasks.append(asyncio.create_task(self._nlp_worker(), name="nlp"))
        await self.ingest.start()

    async def stop(self) -> None:
        await self.ingest.stop()
        for t in self.tasks:
            t.cancel()
        _, stuck = await asyncio.wait(self.tasks, timeout=5)
        for t in stuck:
            log.warning("task %s did not stop within 5 s", t.get_name())
        await asyncio.to_thread(self.store.close)
