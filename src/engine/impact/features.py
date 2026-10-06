"""Impact features, shared by training (``training.impact_data``) and runtime.

One definition, two callers: the historical builder and the live pipeline both
produce the same ``FEATURES`` vector, so the model never sees a feature computed
differently at training and serving time. Stream statistics (burst, novelty,
outlets) come from :class:`EntityActivity`; price features from :class:`PriceBook`.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from engine.nlp.dedupe import shingles
from engine.schema import EventClass

EVENTS = [e.value for e in EventClass]
SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
    "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities", "Sovereign",
]  # fmt: skip
SOURCE_TYPES = ["news", "social", "filing", "central_bank"]
KINDS = ["equity", "bank", "corporate", "sovereign"]

# Sovereign risk tier of the entity's country: 1 developed / AA+, 2 investment-grade EM, 3 high-yield or frontier.
COUNTRY_TIER = {
    **dict.fromkeys("US GB DE FR JP CH NL CA AU SG SE NO FI AT BE IE NZ KR TW HK IT ES IL".split(), 1),
    **dict.fromkeys("CN IN MX CL PL SA AE QA ID MY TH PH PT GR HU RO PE CO".split(), 2),
}
SEVERITY = re.compile(
    r"\b(default\w*|bankrupt\w*|insolven\w*|sanction\w*|war|invasion|fraud\w*|collapse\w*|crash\w*|plunge\w*|"
    r"plummet\w*|soar\w*|surge\w*|record|downgrad\w*|halt\w*|seiz\w*|emergency|crisis|missile\w*|scandal|probe|"
    r"recall\w*|ransomware|breach|warning|slump\w*|tumbl\w*|skyrocket\w*)\b",
    re.I,
)

# Numeric features, then categoricals (encoded as integer codes; the model treats them as categorical).
NUMERIC = [
    "abs_sentiment", "sentiment", "confidence", "event_confidence", "severity_hits", "credibility",
    "log_syndication", "outlets_window", "burst_z", "novelty", "vol_20d", "log_dollar_volume_20d",
    "country_tier", "goldstein", "abs_goldstein", "log_num_mentions", "log_num_sources", "avg_tone",
    "has_gdelt_event", "market_open", "day_of_week", "earnings_window",
]  # fmt: skip
CATEGORICAL = ["event", "sector", "kind", "source_type"]
FEATURES = NUMERIC + CATEGORICAL


def country_tier(cc: str | None) -> int:
    return COUNTRY_TIER.get(cc or "", 3)


def _code(value: str | None, vocab: list[str]) -> int:
    return vocab.index(value) if value in vocab else len(vocab)


def load_market_map(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def yf_ticker(entity_id: str, kind: str, market_map: dict[str, Any]) -> str | None:
    """Yahoo ticker for an entity, or None (sovereigns use their country index)."""
    if entity_id in market_map["tickers"]:
        return str(market_map["tickers"][entity_id])
    if kind == "equity":
        return entity_id
    return None


def country_info(cc: str | None, market_map: dict[str, Any]) -> dict[str, Any]:
    return market_map["countries"].get(cc or "", market_map["countries"]["default"])


def market_open(ts: datetime, close_utc: float) -> bool:
    """Approximate regular session: weekdays, the 6.5 hours before the local close."""
    if ts.weekday() >= 5:
        return False
    h = ts.hour + ts.minute / 60
    return close_utc - 6.5 <= h < close_utc


# ----------------------------------------------------------------------------- stream statistics
@dataclass
class ActivityStats:
    burst_z: float
    novelty: float
    outlets_window: int
    syndication: int
    baseline_windows: int = 0  # windows of history behind burst_z (few = unreliable)
    baseline_active: int = 0  # of those, windows in which this entity was mentioned at all


@dataclass
class EntityActivity:
    """Per-entity mention rates, outlet diversity and novelty over a rolling history.

    * burst_z: mentions in the current 15-minute window vs. the mean/std of the
      entity's per-window counts over the previous 7 days;
    * novelty: 1 - max Jaccard similarity to the entity's docs in the last 72 h
      (at most its 300 most recent, which bounds the cost per signal);
    * outlets_window: distinct outlets naming the entity in the current window.

    ``time_scale`` > 1 compresses all windows for accelerated replay (30x replay:
    a 15-minute window lasts 30 real seconds).
    """

    window_s: float = 900.0
    baseline_s: float = 7 * 86400.0
    novelty_s: float = 72 * 3600.0
    time_scale: float = 1.0
    _counts: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    _outlets: dict[str, dict[int, set[str]]] = field(default_factory=lambda: defaultdict(dict))
    # Novelty compares against the entity's most recent docs only (bounded cost per signal).
    _docs: dict[str, deque[tuple[float, frozenset[str]]]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=300))
    )
    _observed: deque[int] = field(default_factory=deque)  # windows the tracker has seen any doc in

    def _bucket(self, t: float) -> int:
        return int(t // (self.window_s / self.time_scale))

    def observe(
        self, entity: str, t: float, outlet: str, text: str, lang: str = "en", update: bool = True
    ) -> ActivityStats:
        b = self._bucket(t)
        n_base = int(self.baseline_s / self.window_s)
        if update and (not self._observed or self._observed[-1] != b):
            self._observed.append(b)
        while self._observed and self._observed[0] < b - n_base:
            self._observed.popleft()
        # Baseline = this entity's count in every *observed* window of the last 7 days. On a
        # continuous stream that is every window; on sampled history only the sampled ones.
        counts = self._counts[entity]
        history = [counts.get(k, 0) for k in self._observed if k != b] or [0]
        mean, std = float(np.mean(history)), float(np.std(history))
        current = counts.get(b, 0) + 1
        burst = (current - mean) / max(std, 0.5)

        sh = frozenset(shingles(text, lang, k=3))
        docs = self._docs[entity]
        horizon = t - self.novelty_s / self.time_scale
        while docs and docs[0][0] < horizon:
            docs.popleft()
        best = 0.0
        for _, other in docs:
            if sh and other:
                best = max(best, len(sh & other) / len(sh | other))
        outlets = self._outlets[entity].get(b, set()) | {outlet}
        if update:
            counts[b] = current
            self._outlets[entity][b] = outlets
            docs.append((t, sh))
            for old in [k for k in counts if k < b - n_base]:
                del counts[old]
                self._outlets[entity].pop(old, None)
        return ActivityStats(
            burst_z=round(burst, 3),
            novelty=round(1.0 - best, 3),
            outlets_window=len(outlets),
            syndication=current,
            baseline_windows=len(self._observed),
            baseline_active=sum(1 for k in self._observed if k != b and counts.get(k, 0) > 0),
        )


# ----------------------------------------------------------------------------- price book
class PriceBook:
    """Daily close/volume per Yahoo ticker; answers 'as of date d' questions with no look-ahead."""

    def __init__(self, prices: pd.DataFrame) -> None:
        # long format: date, ticker, close, volume
        p = prices.dropna(subset=["close"]).sort_values(["ticker", "date"])
        self.close = p.pivot(index="date", columns="ticker", values="close").sort_index()
        self.volume = p.pivot(index="date", columns="ticker", values="volume").reindex(self.close.index)
        self.returns = self.close.pct_change(fill_method=None)
        self.dates = self.close.index

    @classmethod
    def load(cls, path: Path) -> PriceBook:
        return cls(pd.read_parquet(path))

    def has(self, ticker: str | None) -> bool:
        return ticker is not None and ticker in self.close.columns

    def _before(self, ticker: str, d: pd.Timestamp, n: int) -> pd.Series:
        s = self.returns[ticker]
        s = s[s.index < d].dropna()
        return s.iloc[-n:]

    def vol_20d(self, ticker: str | None, d: pd.Timestamp) -> float:
        if not self.has(ticker):
            return float("nan")
        r = self._before(ticker, d, 20)  # type: ignore[arg-type]
        return float(r.std()) if len(r) >= 10 else float("nan")

    def log_dollar_volume_20d(self, ticker: str | None, d: pd.Timestamp) -> float:
        if not self.has(ticker):
            return float("nan")
        c = self.close[ticker][self.close.index < d].iloc[-20:]
        v = self.volume[ticker][self.volume.index < d].iloc[-20:]
        dv = (c * v).dropna()
        return float(math.log1p(dv.mean())) if len(dv) >= 5 and dv.mean() > 0 else float("nan")

    def latest_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[-1])


def build_feature_row(
    *,
    sentiment: float,
    confidence: float,
    event: str,
    event_confidence: float,
    text: str,
    credibility: float,
    syndication: int,
    activity: ActivityStats,
    vol_20d: float,
    log_dollar_volume_20d: float,
    country: str | None,
    sector: str | None,
    kind: str,
    source_type: str,
    gdelt_event: dict[str, Any] | None,
    published_at: datetime,
    close_utc: float,
    earnings_window: float = float("nan"),
) -> dict[str, float]:
    gd = gdelt_event or {}
    ts = published_at.astimezone(UTC)
    gold = gd.get("goldstein")
    return {
        "abs_sentiment": abs(sentiment),
        "sentiment": sentiment,
        "confidence": confidence,
        "event_confidence": event_confidence,
        "severity_hits": float(min(len(SEVERITY.findall(text)), 5)),
        "credibility": credibility,
        "log_syndication": math.log1p(max(syndication, activity.syndication) - 1),
        "outlets_window": float(activity.outlets_window),
        "burst_z": float(np.clip(activity.burst_z, -5, 50)),
        "novelty": activity.novelty,
        "vol_20d": vol_20d,
        "log_dollar_volume_20d": log_dollar_volume_20d,
        "country_tier": float(country_tier(country)),
        "goldstein": float(gold) if gold is not None else np.nan,
        "abs_goldstein": abs(float(gold)) if gold is not None else np.nan,
        "log_num_mentions": math.log1p(gd.get("num_mentions", 0)) if gd else np.nan,
        "log_num_sources": math.log1p(gd.get("num_sources", 0)) if gd else np.nan,
        "avg_tone": float(gd["avg_tone"]) if "avg_tone" in gd else np.nan,
        "has_gdelt_event": 1.0 if gd else 0.0,
        "market_open": 1.0 if market_open(ts, close_utc) else 0.0,
        "day_of_week": float(ts.weekday()),
        "earnings_window": earnings_window,
        "event": float(_code(event, EVENTS)),
        "sector": float(_code(sector, SECTORS)),
        "kind": float(_code(kind, KINDS)),
        "source_type": float(_code(source_type, SOURCE_TYPES)),
    }


def session_day(ts: datetime, close_utc: float) -> pd.Timestamp:
    """Live stand-in for ``trading_day_for`` without a price calendar: the weekday session
    whose close can reflect news at ``ts`` (after the close or at the weekend -> next weekday).
    Holidays are ignored; the earnings window is +/-1 day (+/-3 around weekends) anyway."""
    ts = ts.astimezone(UTC)
    day = pd.Timestamp(ts.date())
    if ts.hour + ts.minute / 60 >= close_utc:
        day = day + timedelta(days=1)
    while day.weekday() >= 5:
        day = day + timedelta(days=1)
    return day


def trading_day_for(ts: datetime, close_utc: float, dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first trading day whose close can reflect news published at ``ts``.

    News before the local close maps to that day; news after the close (or on a
    non-trading day) maps to the next trading day, i.e. the next open.
    """
    ts = ts.astimezone(UTC)
    day = pd.Timestamp(ts.date())
    if ts.hour + ts.minute / 60 >= close_utc:
        day = day + timedelta(days=1)
    idx = dates.searchsorted(day)
    return pd.Timestamp(dates[idx]) if idx < len(dates) else None


class EarningsCalendar:
    """Scheduled earnings dates per ticker (data/reference/earnings_dates.parquet, from yfinance)."""

    def __init__(self, path: Path) -> None:
        self.by_ticker: dict[str, np.ndarray] = {}
        if path.exists():
            df = pd.read_parquet(path)
            for t, g in df.groupby("ticker"):
                self.by_ticker[str(t)] = np.sort(pd.to_datetime(g["date"]).to_numpy(dtype="datetime64[D]"))

    def window(self, ticker: str | None, day: pd.Timestamp | datetime) -> float:
        """1.0 if an earnings date falls within +/-1 calendar day of ``day`` (weekend-adjusted:
        +/-3 days on Mondays/Fridays), 0.0 if not, NaN when the ticker has no calendar."""
        dates = self.by_ticker.get(ticker or "")
        if dates is None or not len(dates):
            return float("nan")
        d = np.datetime64(pd.Timestamp(day).date(), "D")
        span = 3 if pd.Timestamp(day).weekday() in (0, 4) else 1
        i = np.searchsorted(dates, d)
        near = [dates[k] for k in (i - 1, i) if 0 <= k < len(dates)]
        return 1.0 if any(abs(int((x - d).astype(int))) <= span for x in near) else 0.0
