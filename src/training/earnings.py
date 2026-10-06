"""Earnings calendar (keyless, yfinance) for the impact model's `earnings_window` feature.

    python -m training.earnings      # -> data/reference/earnings_dates.parquet (committed, small)

Earnings dates are scheduled and announced weeks ahead, so "is d0 within one trading day
of an earnings date" is known at publication time: not look-ahead. Releases after the US
close (yfinance times them at 16:00 ET) count toward the next session, which the +/-1 day
window already covers.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from engine.config import get_settings
from engine.impact.features import load_market_map, yf_ticker
from engine.nlp.entities import EntityLinker

log = logging.getLogger("earnings")


def main() -> None:
    import yfinance as yf

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    mm = load_market_map(get_settings().reference_dir / "market_map.yaml")
    rows = []
    for e in EntityLinker().entities.values():
        tick = yf_ticker(e.id, e.kind, mm)
        if not tick:
            continue
        try:
            d = yf.Ticker(tick).get_earnings_dates(limit=48)
        except Exception as exc:
            log.info("%s: %s", tick, type(exc).__name__)
            continue
        if d is None or d.empty:
            continue
        for ts in d.index:
            rows.append(
                {"ticker": tick, "date": pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).normalize()}
            )
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates().sort_values(["ticker", "date"])
    out = get_settings().reference_dir / "earnings_dates.parquet"
    df.to_parquet(out, index=False)
    log.info("earnings dates: %d rows for %d tickers -> %s", len(df), df["ticker"].nunique(), out)


if __name__ == "__main__":
    main()
