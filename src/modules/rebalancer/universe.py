"""Module A universe: 20 S&P 100 names across 11 sectors, with float-adjusted base weights.

    python -m modules.rebalancer.universe     # refresh data/reference/universe.csv (yfinance, keyless)

Base weight = float-adjusted market cap (float shares x price), capped at 10% per name
(excess redistributed pro rata, repeated until no name exceeds the cap).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import get_settings

log = logging.getLogger(__name__)
UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC", "GS",
            "XOM", "CVX", "JNJ", "UNH", "PFE", "PG", "KO", "WMT", "CAT", "BA"]  # fmt: skip
BASE_CAP = 0.10


def universe_path() -> Path:
    return get_settings().reference_dir / "universe.csv"


def load_universe() -> pd.DataFrame:
    """ticker, name, sector, float_shares (committed snapshot)."""
    return pd.read_csv(universe_path()).set_index("ticker").loc[UNIVERSE]


def capped(weights: pd.Series, cap: float) -> pd.Series:
    w = weights / weights.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        w[free] += excess * w[free] / w[free].sum()
    return w


def base_weights(prices: pd.Series | None = None) -> pd.Series:
    """Float-adjusted cap weights at the given prices (latest snapshot prices by default), capped at 10%."""
    u = load_universe()
    px = prices if prices is not None else u["last_price"]
    return capped((u["float_shares"] * px.reindex(u.index)).astype(float), BASE_CAP)


def refresh() -> Path:
    import yfinance as yf

    from engine.nlp.entities import EntityLinker

    linker = EntityLinker()
    rows = []
    for t in UNIVERSE:
        info = yf.Ticker(t).info
        e = linker.entities[t]
        rows.append(
            {
                "ticker": t,
                "name": e.name,
                "sector": e.sector,
                "float_shares": float(info.get("floatShares") or info.get("sharesOutstanding") or np.nan),
                "last_price": float(info.get("currentPrice") or info.get("regularMarketPrice") or np.nan),
                "as_of": pd.Timestamp.utcnow().date().isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(universe_path(), index=False)
    log.info("universe: %s", df[["ticker", "sector", "float_shares"]].to_string())
    return universe_path()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh()
