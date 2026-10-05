"""Builds connectors from ``sources.yaml`` and runs them concurrently on the bus."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import httpx
import yaml

from engine.bus import TOPIC_RAW, Bus
from engine.config import RunMode, Settings
from engine.ingest.base import Source, SourceRunner
from engine.ingest.gdelt import GdeltDocSource, GdeltGkgSource
from engine.ingest.replay import ReplayReader, twin_paths
from engine.ingest.rss import EdgarSource, RedditRssSource, RssSource
from engine.ingest.social import HackerNewsSource, MastodonSource
from engine.schema import RawDoc, SourceType

log = logging.getLogger(__name__)

CONNECTORS: dict[str, type[Source]] = {
    "gdelt_gkg": GdeltGkgSource,
    "gdelt_doc": GdeltDocSource,
    "rss": RssSource,
    "edgar": EdgarSource,
    "reddit": RedditRssSource,
    "mastodon": MastodonSource,
    "hackernews": HackerNewsSource,
}


def load_source_config(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return {k: dict(v or {}) for k, v in cfg.items()}


def build_source(name: str, cfg: dict[str, Any]) -> Source | None:
    """Instantiate a connector; ``None`` for replay-only sources."""
    kind = cfg.get("kind", name)
    if kind == "replay_only":
        return None
    try:
        cls = CONNECTORS[kind]
    except KeyError as exc:
        raise ValueError(f"source {name!r}: unknown kind {kind!r}") from exc
    params = {
        k: v for k, v in cfg.items() if k not in {"kind", "live", "note", "source_type", "replay_covered_by"}
    }
    return cls(name=name, **params)


def make_http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=settings.http_timeout_s,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=20),
    )


class IngestManager:
    """Owns one :class:`SourceRunner` per configured source and publishes to ``raw.docs``."""

    def __init__(self, settings: Settings, bus: Bus, mode: RunMode | None = None) -> None:
        self.settings = settings
        self.bus = bus
        self.mode = mode or settings.mode
        self.runners: dict[str, SourceRunner] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None

    async def _publish(self, doc: RawDoc) -> None:
        await self.bus.publish(TOPIC_RAW, doc)

    def _replay_factory(self, name: str):
        if not twin_paths(self.settings.replay_dir, name):
            return None
        s = self.settings
        return lambda: ReplayReader.for_source(
            s.replay_dir, name, speed=s.replay_speed, max_gap_s=s.replay_max_gap_s, loop=s.replay_loop
        )

    def build(self) -> None:
        cfg = load_source_config(self.settings.sources_file)
        enabled = set(self.settings.enabled_sources) if self.settings.enabled_sources else None
        live_mode = self.mode is RunMode.LIVE
        if live_mode:
            self._client = make_http_client(self.settings)
        for name, scfg in cfg.items():
            if enabled is not None and name not in enabled:
                continue
            source = build_source(name, scfg)
            stype = source.source_type if source else SourceType(scfg.get("source_type", "news"))
            live = live_mode and source is not None and scfg.get("live", True)
            runner = SourceRunner(
                source=source,
                name=name,
                source_type=stype,
                emit=self._publish,
                replay_factory=self._replay_factory(name),
                live=bool(live),
                client=self._client,
            )
            if scfg.get("note"):
                runner.health.note = scfg["note"]
            runner.replay_covered_by = scfg.get("replay_covered_by")
            self.runners[name] = runner

    async def start(self) -> None:
        if not self.runners:
            self.build()
        for name, runner in self.runners.items():
            self._tasks.append(asyncio.create_task(runner.run(), name=f"source:{name}"))
        log.info("ingest started in %s mode with %d sources", self.mode.value, len(self.runners))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        for r in self.runners.values():
            await r.aclose()
        if self._client is not None:
            await self._client.aclose()

    def health(self) -> list[dict[str, Any]]:
        return [r.health.as_dict(r.breaker if r.live else None) for r in self.runners.values()]

    def set_killed(self, name: str, killed: bool) -> bool:
        r = self.runners.get(name)
        if r is None:
            return False
        r.killed = killed
        return True
