"""Signal store and file sinks.

* In memory: the most recent signals for the API, with cursor pagination
  (cursor = sequence number of the last item you saw).
* Idempotent: a signal id seen before is ignored, so replays and at-least-once
  delivery never double-count.
* Files (the problem statement's "writing to a file"): every signal is appended to
  ``data/output/signals.jsonl``; each completed hour is also written to
  ``data/output/parquet/signals_YYYYMMDDHH.parquet``. Both are queryable with DuckDB:
  ``duckdb -c "select entity.id, avg(sentiment.score) from 'data/output/signals.jsonl' group by 1"``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engine.schema import EventClass, Signal

log = logging.getLogger(__name__)


@dataclass
class Page:
    items: list[Signal]
    next_cursor: int | None


def _flat(s: Signal) -> dict[str, Any]:
    return {
        "signal_id": s.signal_id, "published_at": s.published_at, "ingested_at": s.ingested_at,
        "source": s.source.name, "outlet": s.source.outlet, "source_type": s.source.type.value,
        "language": s.language, "entity_id": s.entity.id, "entity_name": s.entity.name,
        "sector": s.entity.sector, "country": s.entity.country, "sentiment": s.sentiment.score, "sentiment_human": s.sentiment.score_human,
        "sentiment_label": s.sentiment.label, "sentiment_confidence": s.sentiment.confidence,
        "event": s.event.primary.value, "event_confidence": s.event.confidence, "impact": s.impact.score,
        "impact_low": s.impact.low, "impact_high": s.impact.high, "impact_mode": s.impact.mode,
        "novelty": s.novelty, "syndication_count": s.syndication_count, "title": s.evidence.title,
        "url": s.evidence.url, "latency_ms": s.latency_ms,
    }  # fmt: skip


class SignalStore:
    def __init__(self, output_dir: Path | None, buffer: int = 20_000, max_ids: int = 1_000_000) -> None:
        self.output_dir = output_dir
        self._recent: deque[tuple[int, Signal]] = deque(maxlen=buffer)
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._max_ids = max_ids
        self._seq = 0
        self._pending: list[Signal] = []
        self._hour_rows: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.duplicates = 0
        self.written = 0
        if output_dir is not None:
            (output_dir / "parquet").mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._recent)

    def add(self, s: Signal) -> bool:
        with self._lock:
            if s.signal_id in self._ids:
                self.duplicates += 1
                return False
            self._ids[s.signal_id] = None
            if len(self._ids) > self._max_ids:
                self._ids.popitem(last=False)
            self._seq += 1
            self._recent.append((self._seq, s))
            self._pending.append(s)
            return True

    def query(
        self,
        entity: str | None = None,
        event: EventClass | None = None,
        min_impact: int = 1,
        language: str | None = None,
        since: float | None = None,
        cursor: int | None = None,
        limit: int = 50,
    ) -> Page:
        """Newest first. Pass ``next_cursor`` back as ``cursor`` for the next page."""
        out: list[Signal] = []
        last_seq: int | None = None
        for seq, s in reversed(self._recent):
            if cursor is not None and seq >= cursor:
                continue
            if entity and s.entity.id != entity:
                continue
            if event and s.event.primary != event:
                continue
            if s.impact.score < min_impact or (language and s.language != language):
                continue
            if since is not None and s.published_at.timestamp() < since:
                continue
            out.append(s)
            last_seq = seq
            if len(out) >= limit:
                break
        return Page(out, last_seq if len(out) >= limit else None)

    def latest(self, n: int = 1000) -> list[Signal]:
        return [s for _, s in list(self._recent)[-n:]]

    def flush(self) -> int:
        """Append pending signals to JSONL; roll completed hours to Parquet. Called off the event loop."""
        if self.output_dir is None:
            return 0
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        with open(self.output_dir / "signals.jsonl", "a", encoding="utf-8") as fh:
            for s in pending:
                fh.write(s.model_dump_json() + "\n")
        for s in pending:
            hour = s.ingested_at.strftime("%Y%m%d%H")
            self._hour_rows.setdefault(hour, []).append(_flat(s))
        current = max(self._hour_rows)
        for hour in [h for h in self._hour_rows if h < current]:
            self._write_parquet(hour)
        self.written += len(pending)
        return len(pending)

    def _write_parquet(self, hour: str) -> None:
        assert self.output_dir is not None
        rows = self._hour_rows.pop(hour)
        path = self.output_dir / "parquet" / f"signals_{hour}.parquet"
        df = pd.DataFrame(rows)
        if path.exists():  # restart within the same hour: merge, keep ids unique
            df = pd.concat([pd.read_parquet(path), df]).drop_duplicates("signal_id")
        df.to_parquet(path, index=False)

    def close(self) -> None:
        self.flush()
        for hour in list(self._hour_rows):
            self._write_parquet(hour)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
