"""Replay recorder: polls live sources and writes their twins to ``data/replay/``.

Examples::

    python -m engine.ingest.record --minutes 20                    # all live sources
    python -m engine.ingest.record --sources rss,mastodon --minutes 5
    python -m engine.ingest.record --gdelt-backfill-hours 6        # last 6 h of GDELT windows
    python -m engine.ingest.record --gdelt-window 2022-02-24T00:00 --hours 24 --name scenario_2022-02-24

Recording is the only step that needs the internet; everything after it runs offline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

from engine.config import get_settings
from engine.ingest.base import SeenCache
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import build_source, load_source_config, make_http_client
from engine.ingest.replay import ReplayWriter
from engine.schema import RawDoc

log = logging.getLogger("record")


def _windows(start: datetime, hours: float) -> list[str]:
    start = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)
    n = int(hours * 4)
    return [(start + timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S") for i in range(n)]


async def record_gdelt_windows(
    start: datetime, hours: float, name: str, max_docs: int, concurrency: int = 4
) -> int:
    s = get_settings()
    src = GdeltGkgSource(name=name, translation=True, max_docs_per_file=max_docs)
    writer = ReplayWriter(s.replay_dir, name)
    seen = SeenCache()
    sem = asyncio.Semaphore(concurrency)
    async with make_http_client(s) as client:

        async def one(stamp: str) -> list[RawDoc]:
            async with sem:
                try:
                    return await src.fetch_window(client, stamp)
                except Exception as exc:
                    log.warning("window %s failed: %s", stamp, exc)
                    return []

        stamps = _windows(start, hours)
        for i in range(0, len(stamps), concurrency * 4):
            batch = await asyncio.gather(*(one(st) for st in stamps[i : i + concurrency * 4]))
            for docs in batch:
                for d in docs:
                    if seen.add(d.doc_id):
                        writer.write(
                            d.model_copy(update={"source": name if name.startswith("gdelt") else d.source})
                        )
            log.info(
                "gdelt %s: %d/%d windows, %d docs",
                name,
                min(i + concurrency * 4, len(stamps)),
                len(stamps),
                writer.count,
            )
    writer.close()
    return writer.count


async def record_live(source_names: list[str] | None, minutes: float) -> dict[str, int]:
    s = get_settings()
    cfg = load_source_config(s.sources_file)
    counts: dict[str, int] = {}
    async with make_http_client(s) as client:

        async def run_one(name: str) -> None:
            src = build_source(name, cfg[name])
            if src is None:
                return
            writer = ReplayWriter(s.replay_dir, name)
            seen = SeenCache()
            deadline = time.monotonic() + minutes * 60
            # Record at least twice as often as the normal poll so short sessions still collect data.
            interval = min(src.poll_interval_s, max(30.0, minutes * 60 / 4))
            try:
                while True:
                    try:
                        for d in await src.fetch(client):
                            if seen.add(d.doc_id):
                                writer.write(d)
                    except Exception as exc:
                        log.warning("%s: poll failed: %s", name, exc)
                    log.info("%s: %d docs so far", name, writer.count)
                    if time.monotonic() + interval > deadline:
                        break
                    await asyncio.sleep(interval)
            finally:
                writer.close()
                counts[name] = writer.count
                if writer.count == 0:
                    writer.path.unlink(missing_ok=True)

        names = source_names or [
            n for n, c in cfg.items() if c.get("kind") not in {"replay_only", "gdelt_gkg"}
        ]
        await asyncio.gather(*(run_one(n) for n in names if n in cfg))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", help="comma-separated source names (default: all live, except gdelt_gkg)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument(
        "--gdelt-backfill-hours", type=float, default=0.0, help="record the last N hours of GDELT GKG"
    )
    ap.add_argument("--gdelt-window", help="UTC start (ISO) of a historical GDELT window to record")
    ap.add_argument("--hours", type=float, default=24.0, help="length of --gdelt-window")
    ap.add_argument("--name", default="gdelt_gkg", help="replay twin name for GDELT recordings")
    ap.add_argument("--max-docs-per-file", type=int, default=400)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.gdelt_window:
        start = datetime.fromisoformat(args.gdelt_window).replace(tzinfo=UTC)
        n = asyncio.run(record_gdelt_windows(start, args.hours, args.name, args.max_docs_per_file))
        print(f"{args.name}: {n} docs")
        return
    if args.gdelt_backfill_hours:
        start = datetime.now(UTC) - timedelta(hours=args.gdelt_backfill_hours + 0.5)
        n = asyncio.run(
            record_gdelt_windows(start, args.gdelt_backfill_hours, args.name, args.max_docs_per_file)
        )
        print(f"{args.name}: {n} docs")
        if not args.sources:
            return
    names = args.sources.split(",") if args.sources else None
    counts = asyncio.run(record_live(names, args.minutes))
    for k, v in sorted(counts.items()):
        print(f"{k}: {v} docs")


if __name__ == "__main__":
    main()
