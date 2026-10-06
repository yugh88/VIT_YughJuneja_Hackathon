"""Build the impact training set from real market reactions (no look-ahead).

    python -m training.impact_data prices      # daily prices, 2018-06 -> today (yfinance, keyless)
    python -m training.impact_data gdelt       # one GDELT 15-min window per weekday, 2019 -> 2026 (resumable)
    python -m training.impact_data features    # NLP + stream features + abnormal-return labels

Label for a (entity, news time t0):
    d0  = first trading day whose close can reflect the news (after-close news -> next open)
    AR  = sum over d0, d0+1 of (r - beta * r_market); beta from the 120 trading days before d0
    z   = |AR| / sigma60, sigma60 = std of market-model residuals over the 60 days before d0
Sovereign entities use max(|index move| / sigma60, |VIX change| / sigma60) instead.
Everything used to build a feature is dated strictly before d0; only the label looks forward.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.config import get_settings
from engine.impact.features import (
    EarningsCalendar,
    EntityActivity,
    PriceBook,
    build_feature_row,
    country_info,
    load_market_map,
    trading_day_for,
    yf_ticker,
)
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import make_http_client
from engine.nlp.entities import EntityLinker
from engine.schema import RawDoc, SourceType
from training.data import raw_dir

log = logging.getLogger("impact_data")
START, END = date(2019, 1, 1), date(2026, 8, 31)  # 2019 H1 windows lack page titles (see credit_labels)
PRICE_START = "2018-06-01"


def ref_dir() -> Path:
    return get_settings().reference_dir


# ----------------------------------------------------------------------------- prices
def all_tickers(linker: EntityLinker, mm: dict[str, Any]) -> list[str]:
    ticks = {yf_ticker(e.id, e.kind, mm) for e in linker.entities.values()}
    ticks |= {c["index"] for c in mm["countries"].values()} | {mm["volatility_index"]}
    return sorted(t for t in ticks if t)


def download_prices() -> Path:
    import yfinance as yf

    mm = load_market_map(ref_dir() / "market_map.yaml")
    tickers = all_tickers(EntityLinker(), mm)
    frames = []
    for i in range(0, len(tickers), 40):
        chunk = tickers[i : i + 40]
        d = yf.download(chunk, start=PRICE_START, auto_adjust=True, progress=False, threads=True)
        close = d["Close"].stack().rename("close")
        vol = d["Volume"].stack().rename("volume")
        part = pd.concat([close, vol], axis=1).reset_index()
        part.columns = ["date", "ticker", "close", "volume"]
        frames.append(part)
        log.info("prices %d/%d tickers", min(i + 40, len(tickers)), len(tickers))
    prices = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    out = raw_dir() / "prices.parquet"
    prices.to_parquet(out)
    missing = sorted(set(tickers) - set(prices["ticker"]))
    log.info("prices: %d rows, %d tickers; missing: %s", len(prices), prices["ticker"].nunique(), missing)
    # Small committed snapshot (last 90 trading days) for runtime price features in the offline demo.
    snap = prices.sort_values("date").groupby("ticker").tail(90)
    snap.to_parquet(ref_dir() / "price_snapshot.parquet")
    return out


# ----------------------------------------------------------------------------- GDELT sample
def windows() -> list[str]:
    """One 15-minute window per weekday, alternating 14:30 UTC (US session) and 20:30 UTC (after the close)."""
    out, d, k = [], START, 0
    while d <= END:
        if d.weekday() < 5:
            hhmm = "1430" if k % 2 == 0 else "2030"
            out.append(d.strftime("%Y%m%d") + hhmm + "00")
            k += 1
        d += timedelta(days=1)
    return out


async def fetch_windows(
    concurrency: int = 6, offset_hours: float = 0.0, subdir: str = "impact_windows"
) -> None:
    """Fetch the sampled windows (or, with ``offset_hours=-1``, the window one hour before each)."""
    out_dir = raw_dir() / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    shift = timedelta(hours=offset_hours)
    stamps = [(datetime.strptime(w, "%Y%m%d%H%M%S") + shift).strftime("%Y%m%d%H%M%S") for w in windows()]
    todo = [w for w in stamps if not (out_dir / f"{w}.parquet").exists()]
    log.info("%d windows to fetch (%d done) into %s", len(todo), len(stamps) - len(todo), subdir)
    linker = EntityLinker()
    src = GdeltGkgSource(translation=False, max_docs_per_file=None)  # type: ignore[arg-type]
    sem = asyncio.Semaphore(concurrency)
    async with make_http_client(get_settings()) as client:

        async def one(stamp: str) -> None:
            async with sem:
                docs: list[RawDoc] = []
                for attempt in range(3):
                    try:
                        docs = await src.fetch_window(client, stamp)
                        break
                    except Exception as exc:
                        log.warning("%s attempt %d: %s", stamp, attempt + 1, exc)
                        await asyncio.sleep(2 * (attempt + 1))
            rows, sov = [], 0
            for d in docs:
                ents = linker.link(d.title).entities()
                companies = [e.id for e in ents if e.kind != "sovereign"]
                sovs = [e.id for e in ents if e.kind == "sovereign"]
                if not companies and not (sovs and sov < 15):
                    continue
                sov += 0 if companies else 1
                ev = d.meta.get("gdelt_event") or {}
                rows.append(
                    {
                        "window": stamp,
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "outlet": d.outlet,
                        "url": d.url,
                        "entities": companies or sovs,
                        "themes": d.meta.get("gdelt_themes", []),
                        "tone": d.meta.get("gdelt_tone", {}).get("tone"),
                        "gdelt_event": {
                            k: ev.get(k)
                            for k in ("cameo_root", "goldstein", "num_mentions", "num_sources", "avg_tone")
                        }
                        if ev
                        else None,
                    }
                )
            pd.DataFrame(rows).to_parquet(out_dir / f"{stamp}.parquet")

        for i in range(0, len(todo), 60):
            await asyncio.gather(*(one(s) for s in todo[i : i + 60]))
            log.info("windows %d/%d", min(i + 60, len(todo)), len(todo))


# ----------------------------------------------------------------------------- labels
def abnormal_z(pb: PriceBook, ticker: str, bench: str, d0: pd.Timestamp) -> float | None:
    r, m = pb.returns[ticker], pb.returns[bench]
    both = pd.concat([r, m], axis=1, keys=["r", "m"]).dropna()
    before = both[both.index < d0].iloc[-120:]
    after = both[both.index >= d0].iloc[:2]
    if len(before) < 80 or len(after) < 2 or after.index[0] != d0:
        return None
    cov = np.cov(before["r"], before["m"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
    resid = (before["r"] - beta * before["m"]).iloc[-60:]
    sigma = float(resid.std())
    if not sigma or math.isnan(sigma):
        return None
    ar = float((after["r"] - beta * after["m"]).sum())
    return abs(ar) / sigma


def sovereign_z(pb: PriceBook, index: str, vix: str, d0: pd.Timestamp) -> float | None:
    if not (pb.has(index) and pb.has(vix)):
        return None
    r = pb.returns[index].dropna()
    before, after = r[r.index < d0].iloc[-60:], r[r.index >= d0].iloc[:2]
    if len(before) < 40 or len(after) < 2:
        return None
    zi = abs(float(after.sum())) / float(before.std())
    v = pb.close[vix].dropna()
    dv = v.diff()
    vb = dv[dv.index < d0].iloc[-60:]
    pre = v[v.index < d0]
    post = v[v.index >= d0].iloc[:2]
    if len(vb) < 40 or len(post) < 2 or pre.empty:
        return zi
    zv = abs(float(post.iloc[-1] - pre.iloc[-1])) / float(vb.std())
    return max(zi, zv)


_Z_CACHE: dict[tuple[str, str, Any], float | None] = {}


def cached_z(kind: str, pb: PriceBook, a: str, b: str, d0: pd.Timestamp) -> float | None:
    """Labels depend only on (ticker, benchmark, day); many rows share them."""
    key = (a, b, d0)
    if key not in _Z_CACHE:
        _Z_CACHE[key] = sovereign_z(pb, a, b, d0) if kind == "sovereign" else abnormal_z(pb, a, b, d0)
    return _Z_CACHE[key]


def build_features() -> Path:
    from engine.nlp.pipeline import Pipeline, credibility

    mm = load_market_map(ref_dir() / "market_map.yaml")
    files = sorted((raw_dir() / "impact_windows").glob("*.parquet"))
    docs = pd.concat([pd.read_parquet(f) for f in files if f.stat().st_size > 0], ignore_index=True)
    docs = docs.dropna(subset=["title"]).sort_values("window").reset_index(drop=True)
    log.info("docs: %d from %d windows", len(docs), len(files))
    pb = PriceBook.load(raw_dir() / "prices.parquet")
    pipe = Pipeline(zero_shot=False)
    activity = EntityActivity()
    cal = EarningsCalendar(ref_dir() / "earnings_dates.parquet")
    # Denser burst sampling: an extra window 1 h before each sampled window feeds the activity
    # tracker (mention counts, outlets, novelty history) but produces no training rows.
    burst_dir = raw_dir() / "impact_burst"
    pre: dict[str, pd.DataFrame] = {}
    if burst_dir.exists():
        for f in burst_dir.glob("*.parquet"):
            if f.stat().st_size:
                pre[f.stem] = pd.read_parquet(f)
    log.info("burst pre-windows available: %d", len(pre))
    rows: list[dict[str, Any]] = []
    for w, g in docs.groupby("window", sort=True):
        ts = datetime.strptime(str(w), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        pw = (ts - timedelta(hours=1)).strftime("%Y%m%d%H%M%S")
        if pw in pre:
            pts = (ts - timedelta(hours=1)).timestamp()
            for r in pre[pw].itertuples():
                for eid in r.entities:
                    activity.observe(str(eid), pts, r.outlet, r.title, "en")
        raws = [
            RawDoc(
                doc_id=r.doc_id,
                source="gdelt_gkg",
                source_type=SourceType.NEWS,
                outlet=r.outlet,
                url=r.url,
                title=r.title,
                language="en",
                published_at=ts,
                meta={"gdelt_event": r.gdelt_event, "gdelt_themes": list(r.themes)}
                if r.gdelt_event is not None
                else {"gdelt_themes": list(r.themes)},
            )
            for r in g.itertuples()
        ]
        for tr in pipe.process_batch(raws, dedupe=True, zero_shot=False):
            if tr.event is None:
                continue
            for sig in tr.signals:
                e = pipe.linker.entities[sig.entity.id]
                st = activity.observe(e.id, ts.timestamp(), tr.doc.outlet, tr.clean.text, "en")
                info = country_info(e.country, mm)
                tick = yf_ticker(e.id, e.kind, mm)
                d0 = trading_day_for(ts, info["close_utc"], pb.dates)
                if d0 is None:
                    continue
                if e.kind == "sovereign":
                    z = cached_z("sovereign", pb, info["index"], mm["volatility_index"], d0)
                elif pb.has(tick) and pb.has(info["index"]):
                    z = cached_z("equity", pb, tick, info["index"], d0)  # type: ignore[arg-type]
                else:
                    z = None
                feats = build_feature_row(
                    sentiment=sig.sentiment.score,
                    confidence=sig.sentiment.confidence,
                    event=tr.event.primary,
                    event_confidence=tr.event.confidence,
                    text=tr.clean.text,
                    credibility=credibility(tr.doc),
                    syndication=tr.syndication_count,
                    activity=st,
                    vol_20d=pb.vol_20d(tick, d0) if tick else pb.vol_20d(info["index"], d0),
                    log_dollar_volume_20d=pb.log_dollar_volume_20d(tick, d0) if tick else float("nan"),
                    country=e.country,
                    sector=e.sector,
                    kind=e.kind,
                    source_type="news",
                    gdelt_event=tr.doc.meta.get("gdelt_event"),
                    published_at=ts,
                    close_utc=info["close_utc"],
                    earnings_window=cal.window(tick, d0),
                )
                rows.append(
                    {
                        "published_at": ts,
                        "d0": d0,
                        "entity": e.id,
                        "doc_id": tr.doc.doc_id,
                        "title": tr.doc.title,
                        "z": z,
                        "heuristic_impact": sig.impact.score,
                        **feats,
                    }
                )
        if len(rows) and len(rows) % 2000 < 40:
            log.info("window %s: %d rows", w, len(rows))
    out = pd.DataFrame(rows)
    path = raw_dir() / "impact_dataset.parquet"
    out.to_parquet(path)
    log.info("impact dataset: %d rows (%d with labels) -> %s", len(out), out["z"].notna().sum(), path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prices", "gdelt", "burst", "features", "all"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.step in {"prices", "all"}:
        download_prices()
    if args.step in {"gdelt", "all"}:
        asyncio.run(fetch_windows())
    if args.step in {"burst", "all"}:
        asyncio.run(fetch_windows(offset_hours=-1, subdir="impact_burst"))
    if args.step in {"features", "all"}:
        build_features()


if __name__ == "__main__":
    main()
