"""Shock library calibrated from measured market moves over each anchor episode (keyless).

    python -m modules.stress.calibrate   # -> data/reference/shocks.yaml + factor_history.parquet

Factors: equity (SPY), financials (XLF), EM equity (EEM), oil (WTI, FRED DCOILWTICO), USD (FRED
DTWEXBGS broad index), UST 10y (FRED DGS10), VIX, and credit spreads from long-history proxies:
IG = Moody's Baa yield - 10y UST (FRED DBAA - DGS10); HY and EM = spread changes implied by the
excess return of HYG over IEI and EMB over IEF (duration-matched). FRED now serves the ICE BofA
OAS indices for the last 3 years only, which would miss the 2022-2023 anchors. Each event class's shock is the cumulative move of
every factor over its anchor window. Classes without a market-wide anchor (M&A, cyber /
operational) keep the plan's idiosyncratic starting values, marked as such.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import httpx
import pandas as pd
import yaml

from engine.config import get_settings

log = logging.getLogger("calibrate_shocks")
FRED = {"oil": "DCOILWTICO", "usd": "DTWEXBGS", "rates": "DGS10", "baa": "DBAA"}
YF = {"equity": "SPY", "financials": "XLF", "em_equity": "EEM", "vix": "^VIX"}
# (bond ETF, duration; treasury ETF, duration) for spread proxies; durations ~ fund-reported averages
SPREAD_PROXY = {"hy": ("HYG", 3.3, "IEI", 4.3), "em_spread": ("EMB", 7.0, "IEF", 7.3)}
PCT = {"equity", "financials", "em_equity", "oil", "usd"}  # cumulative % move; others: level change
BP = {"rates": 100, "ig": 100, "hy": 100, "em_spread": 100}  # FRED quotes these in percent -> bp
ANCHORS = {
    "Geopolitical": ("Russia-Ukraine invasion", "2022-02-15", "2022-03-08"),
    "Macroeconomic": ("2022 Fed tightening cycle", "2022-01-03", "2022-06-16"),
    "Credit Event": ("SVB failure and US regional-bank stress", "2023-03-08", "2023-03-17"),
}
IDIOSYNCRATIC = {  # no market-wide anchor; plan section 9 starting values
    "Merger/Acquisition": {"equity": -0.0, "named_equity": -0.03, "named_spread_bp": 40, "named_notches": 1},
    "Cyber/Technology Risk": {"named_equity": -0.10, "sector_spread_bp": 20},
    "Operational/Supply Chain": {"named_equity": -0.10, "sector_spread_bp": 20},
}
EXTRAS = {  # targeting add-ons applied on top of the measured factor moves (plan section 9)
    "Geopolitical": {"affected_country_spread_bp": 300, "em_fx_pct": -0.10},
    "Macroeconomic": {},
    "Credit Event": {"named_notches": 3},
}


def fred(series: str, start: str = "2018-01-01") -> pd.Series:
    r = httpx.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series, "cosd": start},
        timeout=60,
        headers={"User-Agent": get_settings().user_agent},
    )
    df = pd.read_csv(io.StringIO(r.raise_for_status().text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def factor_levels() -> pd.DataFrame:
    import yfinance as yf

    cols = {k: fred(v) for k, v in FRED.items()}
    etfs = sorted({t for v in SPREAD_PROXY.values() for t in (v[0], v[2])})
    px = yf.download(list(YF.values()) + etfs, start="2018-01-01", auto_adjust=True, progress=False)["Close"]
    for k, t in YF.items():
        cols[k] = px[t].dropna()
    df = pd.DataFrame(cols).sort_index().ffill()
    df["ig"] = df.pop("baa") - df["rates"]  # Moody's Baa - 10y, in percent
    for k, (bond, d_b, tsy, d_t) in SPREAD_PROXY.items():
        r_b, r_t = px[bond].pct_change(), px[tsy].pct_change()
        excess = r_b - (d_b / d_t) * r_t
        df[k] = (-excess / d_b * 100).cumsum().reindex(df.index).ffill()  # spread "level" in percent units
    return df.dropna()


def move(levels: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    a = levels.loc[:start].iloc[-1]
    b = levels.loc[:end].iloc[-1]
    out = {}
    for f in levels.columns:
        if f in PCT:
            out[f] = round(float(b[f] / a[f] - 1), 4)
        else:
            out[f] = round(float((b[f] - a[f]) * BP.get(f, 1)), 2)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lv = factor_levels()
    ref = get_settings().reference_dir
    lib: dict[str, Any] = {"units": {f: ("fraction" if f in PCT else "bp" if f in BP else "points") for f in lv.columns},
                           "sources": {**{k: f"FRED {v}" for k, v in FRED.items()}, **{k: f"Yahoo {v}" for k, v in YF.items()}},
                           "classes": {}}  # fmt: skip
    for cls, (name, s, e) in ANCHORS.items():
        lib["classes"][cls] = {
            "anchor": name,
            "window": [s, e],
            "measured": move(lv, s, e),
            "extras": EXTRAS[cls],
        }
    for cls, v in IDIOSYNCRATIC.items():
        lib["classes"][cls] = {
            "anchor": "none (idiosyncratic; plan starting values)",
            "measured": {},
            "extras": v,
        }
    (ref / "shocks.yaml").write_text(yaml.safe_dump(lib, sort_keys=False, allow_unicode=True))
    # 3 years of daily factor changes for Monte Carlo correlations
    hist = lv.loc[lv.index[-1] - pd.Timedelta(days=3 * 365) :]
    chg = pd.DataFrame(
        {f: hist[f].pct_change() if f in PCT else hist[f].diff() * BP.get(f, 1) for f in hist.columns}
    )
    chg.dropna().to_parquet(ref / "factor_history.parquet")
    for cls, v in lib["classes"].items():
        log.info("%-26s %s", cls, v["measured"] or v["extras"])


if __name__ == "__main__":
    main()
