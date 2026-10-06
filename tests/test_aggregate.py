"""Aggregation math, the signal store, and impact labels without look-ahead."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine.aggregate.aggregator import PRIOR_WEIGHT, Aggregator
from engine.aggregate.store import SignalStore, read_jsonl
from engine.impact.features import PriceBook, trading_day_for
from engine.schema import (
    EntityRef,
    Event,
    EventClass,
    Evidence,
    Impact,
    Sentiment,
    Signal,
    SourceRef,
    SourceType,
)

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def sig(i: int, score: float, minutes: float = 0, stype: SourceType = SourceType.NEWS, novelty: float = 1.0,
        entity: str = "AAPL", conf: float = 1.0) -> Signal:  # fmt: skip
    return Signal(
        signal_id=f"sig_{i}", doc_id=f"d{i}", published_at=T0 + timedelta(minutes=minutes), ingested_at=T0,
        source=SourceRef(name="t", outlet="o", type=stype, credibility=1.0), language="en",
        entity=EntityRef(id=entity, name=entity), sentiment=Sentiment(score=score, label="positive", confidence=conf),
        event=Event(primary=EventClass.OTHER, confidence=0.9), impact=Impact(score=5, low=4, high=6),
        novelty=novelty, syndication_count=1, evidence=Evidence(title=f"t{i}"), model_versions={}, latency_ms=0,
    )  # fmt: skip


def test_decay_half_life_and_stale_signal_returns_to_zero() -> None:
    agg = Aggregator()
    agg.add(sig(1, 0.8))
    t = T0.timestamp()
    assert agg.score("AAPL", t) == 0.8 * 1 / (PRIOR_WEIGHT + 1)
    # one news half-life (12 h) later the evidence weight has halved...
    assert math.isclose(agg.evidence_weight("AAPL", t + 12 * 3600), 0.5, rel_tol=1e-9)
    # ...and with no new news the score drifts back toward 0 instead of freezing
    assert agg.score("AAPL", t + 72 * 3600) < 0.05
    # social decays 6x faster (2 h half-life); 30x replay compresses every clock
    agg.add(sig(2, -0.5, stype=SourceType.SOCIAL, entity="TSLA"))
    assert math.isclose(agg.evidence_weight("TSLA", t + 2 * 3600), 0.5, rel_tol=1e-9)
    fast = Aggregator(time_scale=30)
    fast.add(sig(3, 0.8))
    assert math.isclose(fast.evidence_weight("AAPL", t + 12 * 3600 / 30), 0.5, rel_tol=1e-9)


def test_novelty_weighting_and_bot_guard() -> None:
    agg = Aggregator()
    agg.add(sig(1, 0.9))  # fresh story, full weight
    for i in range(14):  # 14 near-copies of a negative wire story barely move it (novelty 0.05)
        agg.add(sig(100 + i, -0.9, minutes=1, novelty=0.05))
    assert agg.score("AAPL", T0.timestamp() + 120) > 0
    bots = Aggregator()
    for i in range(25):  # one account pushing $TSLA 25 times in an hour
        bots.add(
            sig(i, 0.9, minutes=i, stype=SourceType.SOCIAL, entity="TSLA"),
            author="pumper",
            text="$TSLA to the moon",
        )
    assert bots.bot_suppressed == 5  # posts 21-25 get weight 0


def test_store_is_idempotent_paginates_and_writes_files(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    for i in range(30):
        assert store.add(sig(i, 0.1, minutes=i))
    assert not store.add(sig(5, 0.1, minutes=5))  # replayed copy ignored
    page1 = store.query(limit=20)
    page2 = store.query(cursor=page1.next_cursor, limit=20)
    ids = [s.signal_id for s in page1.items + page2.items]
    assert len(ids) == 30 == len(set(ids)) and page2.next_cursor is None
    assert ids[0] == "sig_29"  # newest first
    store.close()
    assert len(read_jsonl(tmp_path / "signals.jsonl")) == 30
    assert (
        pd.concat(pd.read_parquet(p) for p in (tmp_path / "parquet").glob("*.parquet"))["signal_id"].nunique()
        == 30
    )


def test_impact_label_timing_has_no_look_ahead() -> None:
    days = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(0)
    prices = pd.concat([
        pd.DataFrame({"date": days, "ticker": t, "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))),
                      "volume": 1e6}) for t in ("AAA", "SPY")
    ])  # fmt: skip
    pb = PriceBook(prices)
    # before the 20:00 UTC US close -> same day; after the close -> next trading day (the next open)
    assert trading_day_for(datetime(2024, 6, 5, 15, 0, tzinfo=UTC), 20.0, pb.dates) == pd.Timestamp(
        "2024-06-05"
    )
    assert trading_day_for(datetime(2024, 6, 5, 21, 0, tzinfo=UTC), 20.0, pb.dates) == pd.Timestamp(
        "2024-06-06"
    )
    assert trading_day_for(datetime(2024, 6, 8, 12, 0, tzinfo=UTC), 20.0, pb.dates) == pd.Timestamp(
        "2024-06-10"
    )  # Sat
    # price features as of d0 ignore d0 itself and everything after it
    d0 = pd.Timestamp("2024-06-05")
    before = pb.vol_20d("AAA", d0)
    pb.close.loc[pb.close.index >= d0, "AAA"] *= 3  # a huge move on/after d0...
    pb.returns = pb.close.pct_change(fill_method=None)
    assert pb.vol_20d("AAA", d0) == before  # ...cannot leak into the features
