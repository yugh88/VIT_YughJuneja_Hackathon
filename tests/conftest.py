from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine.config import RunMode, Settings
from engine.schema import RawDoc, SourceType

FIXTURES = Path(__file__).parent / "fixtures"


def make_doc(i: int, source: str = "rss", minutes: int = 1, **kw: object) -> RawDoc:
    base = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    fields: dict[str, object] = {
        "doc_id": RawDoc.make_id(source, f"doc-{i}"),
        "source": source,
        "source_type": SourceType.NEWS,
        "outlet": "example.com",
        "url": f"https://example.com/{i}",
        "title": f"Headline number {i}",
        "published_at": base + timedelta(minutes=minutes * i),
    }
    fields.update(kw)
    return RawDoc(**fields)  # type: ignore[arg-type]


def write_twin(replay_dir: Path, name: str, docs: list[RawDoc]) -> Path:
    p = replay_dir / name / "twin.jsonl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for d in docs:
            fh.write(d.model_dump_json() + "\n")
    return p


@pytest.fixture
def replay_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temp data dir with two small replay twins."""
    data = tmp_path / "data"
    (data / "reference").mkdir(parents=True)
    (data / "reference" / "sources.yaml").write_text(
        "rss:\n  kind: rss\n  feeds: []\ntweets:\n  kind: replay_only\n  source_type: social\n"
    )
    names = ["Apple", "Microsoft", "JPMorgan", "Boeing", "Tesla"]
    write_twin(
        data / "replay",
        "rss",
        [make_doc(i, "rss", title=f"{names[i % 5]} shares move on report number {i}") for i in range(20)],
    )
    write_twin(
        data / "replay",
        "tweets",
        [
            make_doc(i, "tweets", source_type=SourceType.SOCIAL, title="", text=f"$AAPL tweet {i}")
            for i in range(20)
        ],
    )
    return Settings(
        mode=RunMode.REPLAY,
        data_dir=data,
        sources_file=data / "reference" / "sources.yaml",
        replay_speed=1000.0,
        replay_max_gap_s=0.01,
        zero_shot=False,
    )


@pytest.fixture(scope="session")
def pipeline():  # type: ignore[no-untyped-def]
    """The real pipeline: fine-tuned ONNX models when present, fallbacks otherwise.

    The zero-shot NLI fallback is off to keep the suite fast.
    """
    from engine.nlp.pipeline import Pipeline

    return Pipeline(zero_shot=False)
