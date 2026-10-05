"""Recorded replay twins.

Every live source writes what it sees to ``data/replay/<source>/*.jsonl.gz`` when
run under the recorder. In replay mode the same files are played back, paced by
the original publication gaps divided by ``speed`` (gaps are clamped so the demo
never stalls), optionally looping with fresh ids.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from engine.schema import RawDoc, utcnow

log = logging.getLogger(__name__)

_CACHE: dict[tuple[str, ...], list[RawDoc]] = {}


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def load_docs(paths: list[Path]) -> list[RawDoc]:
    """Read and time-sort docs from JSONL files; bad lines are skipped and counted."""
    key = tuple(str(p) for p in paths)
    if key in _CACHE:
        return _CACHE[key]
    docs: list[RawDoc] = []
    bad = 0
    for p in paths:
        with _open_text(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(RawDoc.model_validate_json(line))
                except ValueError:
                    bad += 1
    if bad:
        log.warning("replay: skipped %d unreadable lines in %s", bad, key)
    docs.sort(key=lambda d: d.published_at)
    _CACHE[key] = docs
    return docs


def twin_paths(replay_dir: Path, name: str) -> list[Path]:
    d = replay_dir / name
    if not d.is_dir():
        return []
    return sorted([*d.glob("*.jsonl"), *d.glob("*.jsonl.gz")])


def spread_batches(ts: list[float], default_window_s: float = 900.0) -> list[float]:
    """Spread docs that share one timestamp evenly up to the next distinct timestamp.

    GDELT stamps every article in a 15-minute file with the file time; replaying
    those literally would dump hundreds of docs in one instant. ``ts`` must be sorted.
    """
    out = list(ts)
    i, n = 0, len(ts)
    while i < n:
        j = i
        while j + 1 < n and ts[j + 1] == ts[i]:
            j += 1
        if j > i:
            window = (ts[j + 1] - ts[i]) if j + 1 < n else default_window_s
            k = j - i + 1
            for m in range(k):
                out[i + m] = ts[i] + window * m / k
        i = j + 1
    return out


class ReplayReader:
    """Iterates (delay_seconds, doc) pairs. Implements the ``ReplayFeed`` protocol."""

    def __init__(
        self,
        paths: list[Path],
        speed: float = 30.0,
        max_gap_s: float = 2.0,
        loop: bool = True,
        rebase_time: bool = True,
    ) -> None:
        if not paths:
            raise FileNotFoundError("replay twin has no files")
        self.paths = paths
        self.speed = speed
        self.max_gap_s = max_gap_s
        self.loop = loop
        self.rebase_time = rebase_time

    @classmethod
    def for_source(cls, replay_dir: Path, name: str, **kw: object) -> ReplayReader | None:
        paths = twin_paths(replay_dir, name)
        return cls(paths, **kw) if paths else None  # type: ignore[arg-type]

    @property
    def docs(self) -> list[RawDoc]:
        return load_docs(self.paths)

    def __iter__(self) -> Iterator[tuple[float, RawDoc]]:
        docs = self.docs
        if not docs:
            return
        times = spread_batches([d.published_at.timestamp() for d in docs])
        lap = 0
        while True:
            for i, doc in enumerate(docs):
                gap = 0.0 if i == 0 else (times[i] - times[i - 1]) / self.speed
                delay = min(max(gap, 0.0), self.max_gap_s)
                if lap:
                    doc = doc.model_copy(update={"doc_id": f"{doc.doc_id}~r{lap}"})
                yield delay, doc
            if not self.loop:
                return
            lap += 1

    def stamp(self, doc: RawDoc) -> RawDoc:
        """Mark a doc as replayed at emission time. With ``rebase_time`` the doc is
        re-dated to now (original kept in ``meta``) so the live views stay live."""
        now = utcnow()
        update: dict[str, object] = {"ingested_at": now, "replayed": True}
        if self.rebase_time:
            update["published_at"] = now
            update["meta"] = {**doc.meta, "original_published_at": doc.published_at.isoformat()}
        return doc.model_copy(update=update)


class ReplayWriter:
    """Appends docs for one source to a gzip JSONL file (used by the recorder)."""

    def __init__(self, replay_dir: Path, name: str, stamp: str | None = None) -> None:
        stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = replay_dir / name / f"{stamp}.jsonl.gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "at", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.count = 0

    def write(self, doc: RawDoc) -> None:
        rec = json.loads(doc.model_dump_json())
        rec["replayed"] = False
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> None:
        self._fh.close()
