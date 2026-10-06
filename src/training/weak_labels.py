"""Weak event labels for classes the public topic dataset does not cover.

    python -m training.weak_labels --windows 160

twitter-financial-news-topic has no Credit Event, Cyber, ESG/Climate or
Operational/Supply-Chain class, and it is English only. This script samples
historical GDELT 15-minute windows (English + translingual), labels headlines with
high-precision rules over GDELT's CAMEO codes, GKG themes and (for English titles)
keyword patterns, and writes ``data/raw/weak_events.parquet``.

Labels are *weak*: they train the classifier, but no reported metric is computed
on them (see docs/model_card.md).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from engine.config import get_settings
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import make_http_client
from engine.ingest.replay import load_docs, twin_paths
from engine.schema import EventClass, RawDoc
from training.data import raw_dir

log = logging.getLogger("weak")

# (class, required GKG theme prefixes, English title regex). A row gets a class when a
# required theme is present AND (for English titles) the headline matches the regex.
# Non-English titles rely on themes only, so their per-class cap is lower.
# (class, GKG theme prefixes, loose title regex, strict title regex).
# English titles only: a row is labelled when the strict regex matches, or when a
# class theme is present AND the loose regex matches. Theme-only labels on
# non-English titles proved too noisy (see docs/DECISIONS.md), so multilingual
# event data comes from translating these English rows instead.
RULES: list[tuple[EventClass, tuple[str, ...], re.Pattern[str], re.Pattern[str]]] = [
    (
        EventClass.CREDIT_EVENT,
        ("ECON_BANKRUPTCY", "ECON_DEBT", "WB_1235", "ECON_CREDITRATINGS"),
        re.compile(
            r"\b(default(s|ed)?|insolven\w*|downgrad\w*|credit rating|junk|liquidat\w*|receivership|"
            r"debt restructur\w*|creditors?)\b",
            re.I,
        ),
        re.compile(
            r"\b(files? for bankruptcy|filed for bankruptcy|chapter 11|bankruptcy (filing|protection)|"
            r"defaults? on|missed (a |an )?(coupon|interest|bond|debt) payment|(moody'?s|fitch|s&p) (cuts?|downgrades?)|"
            r"downgraded to junk|sovereign default|debt default)\b",
            re.I,
        ),
    ),
    (
        EventClass.CYBER_TECH,
        ("CYBER_ATTACK", "WB_2457_CYBERSECURITY"),
        re.compile(r"\b(cyber\w*|hack(ed|ers?|ing)?|breach|malware|outage|ddos)\b", re.I),
        re.compile(
            r"\b(ransomware|cyber[- ]?attacks?|data breach(es)?|hackers? (stole|breached|hit)|was hacked|"
            r"system outage|global outage|ddos attack)\b",
            re.I,
        ),
    ),
    (
        EventClass.ESG_CLIMATE,
        ("ENV_CLIMATECHANGE", "ENV_EMISSIONS", "ENV_CARBONCAPTURE", "ENV_GREEN", "WB_567_CLIMATE"),
        re.compile(
            r"\b(climate|emission\w*|carbon|net[- ]zero|esg|sustainab\w*|pollut\w*|renewable)\b", re.I
        ),
        re.compile(
            r"\b(esg|greenwash\w*|net[- ]zero (target|pledge)|carbon (emissions|tax|credits)|climate (risk|target|pledge)s?|"
            r"emissions (target|cut)s?|oil spill|toxic (spill|leak))\b",
            re.I,
        ),
    ),
    (
        EventClass.OPERATIONAL_SUPPLY_CHAIN,
        ("STRIKE", "MANMADE_DISASTER", "WB_2443_SUPPLY_CHAIN", "ECON_SUPPLYCHAIN", "SHORTAGE"),
        re.compile(r"\b(supply[- ]chain|shortage\w*|recall\w*|walkout|disruption\w*)\b", re.I),
        re.compile(
            r"\b(supply[- ]chain (disruption|crisis|woes|issues)|chip shortage|workers? (go )?on strike|"
            r"(labou?r|union|workers'?) strike|recalls? \d|recalls? (cars|vehicles|products)|plant (closure|shutdown|fire)|"
            r"factory (fire|closure|shutdown)|production (halt|halted|suspended)|port congestion)\b",
            re.I,
        ),
    ),
    (
        EventClass.GEOPOLITICAL,
        ("ARMEDCONFLICT", "MILITARY", "SANCTIONS", "TERROR", "WB_2433_CONFLICT"),
        re.compile(
            r"\b(war|invasion|invade\w*|missiles?|sanction\w*|troops|military|airstrikes?|ceasefire|tariffs?|coup|conflict)\b",
            re.I,
        ),
        re.compile(
            r"\b(invasion of|imposes? sanctions|new sanctions|missile strikes?|declares? war|military coup|trade war)\b",
            re.I,
        ),
    ),
    (
        EventClass.MACROECONOMIC,
        ("ECON_INFLATION", "ECON_INTEREST_RATES", "ECON_CENTRALBANK", "ECON_UNEMPLOYMENT"),
        re.compile(
            r"\b(inflation|interest rates?|rate (hike|cut)s?|central bank|gdp|recession|unemployment|jobs report|cpi|monetary policy)\b",
            re.I,
        ),
        re.compile(
            r"\b(raises? (interest )?rates|cuts? (interest )?rates|inflation (rises|falls|eases|accelerates|cools)|gdp (grew|shrank|contracted))\b",
            re.I,
        ),
    ),
    (
        EventClass.REGULATORY_LEGAL,
        ("LEGISLATION", "TRIAL", "ARREST", "CORRUPTION", "REGULATION"),
        re.compile(
            r"\b(lawsuit|sue[sd]?|fined|antitrust|regulators?|probe|investigation|settle(s|d|ment)|indict\w*|penalt\w*)\b",
            re.I,
        ),
        re.compile(
            r"\b(antitrust (lawsuit|probe|case)|class[- ]action|fined \$|sec (charges|sues|probe)|doj (sues|probe))\b",
            re.I,
        ),
    ),
    (
        EventClass.EARNINGS_GUIDANCE,
        ("ECON_EARNINGSREPORT", "ECON_STOCKMARKET"),
        re.compile(r"\b(earnings|quarterly (profit|loss|results)|profit warning|guidance|revenue)\b", re.I),
        re.compile(
            r"\b(beats? (estimates|expectations|forecasts)|miss(es|ed)? (estimates|expectations)|profit warning|"
            r"(raises|cuts|lowers|slashes) (its )?(full[- ]year )?(guidance|forecast|outlook)|"
            r"(quarterly|q[1-4]) (profit|loss|earnings) (rises|falls|jumps|slumps|plunges|surges)|"
            r"shares? (slump|plunge|tumble|jump|surge|soar)s? (after|on) (earnings|results|loss|profit))\b",
            re.I,
        ),
    ),
    (
        EventClass.MERGER_ACQUISITION,
        ("ECON_MERGER", "ECON_ACQUISITION", "MERGER", "ACQUISITION"),
        re.compile(r"\b(acquir\w*|acquisition|merger|takeover|buyout)\b", re.I),
        re.compile(
            r"\b(agrees? to (buy|acquire)|to acquire|takeover bid|merger (deal|agreement)|completes? (acquisition|merger))\b",
            re.I,
        ),
    ),
]

CAMEO_GEO_ROOTS = {"14", "15", "16", "17", "18", "19", "20"}  # protest ... mass violence


def label_doc(d: RawDoc) -> EventClass | None:
    if (d.language or "en") != "en":
        return None
    themes = d.meta.get("gdelt_themes", [])
    title = d.title
    matches: list[EventClass] = []
    for cls, theme_prefixes, loose, strict in RULES:
        has_theme = any(t.startswith(theme_prefixes) for t in themes)
        if strict.search(title) or (has_theme and loose.search(title)):
            matches.append(cls)
    ev = d.meta.get("gdelt_event") or {}
    strong_conflict = ev.get("cameo_root") in CAMEO_GEO_ROOTS and abs(ev.get("goldstein", 0)) >= 7
    if not matches and strong_conflict and RULES[4][2].search(title):
        matches.append(EventClass.GEOPOLITICAL)
    return matches[0] if len(matches) == 1 else None  # ambiguous rows are dropped


async def sample_windows(n: int, seed: int = 7) -> list[RawDoc]:
    rng = random.Random(seed)
    start, end = datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
    span = int((end - start).total_seconds() // 900)
    stamps = sorted(
        {(start + timedelta(minutes=15 * rng.randrange(span))).strftime("%Y%m%d%H%M%S") for _ in range(n)}
    )
    src = GdeltGkgSource(translation=False, max_docs_per_file=2000)
    docs: list[RawDoc] = []
    sem = asyncio.Semaphore(4)
    async with make_http_client(get_settings()) as client:

        async def one(stamp: str) -> list[RawDoc]:
            async with sem:
                try:
                    return await src.fetch_window(client, stamp)
                except Exception as exc:
                    log.warning("window %s: %s", stamp, exc)
                    return []

        for i, res in enumerate(await asyncio.gather(*(one(s) for s in stamps))):
            docs += res
            if i % 20 == 0:
                log.info("windows %d/%d, docs %d", i, len(stamps), len(docs))
    return docs


def edgar_rows() -> list[dict[str, Any]]:
    """8-K item codes give clean weak labels for filings (English)."""
    rows = []
    for d in load_docs(twin_paths(get_settings().replay_dir, "edgar_8k")):
        ev = d.meta.get("weak_events") or []
        ev = [e for e in ev if e != EventClass.OTHER.value]
        if len(ev) == 1:
            rows.append({"text": d.title, "event": ev[0], "lang": "en", "source": "edgar_items"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--cap", type=int, default=2500, help="max rows per class")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    docs = asyncio.run(sample_windows(args.windows))
    rows = []
    seen: set[str] = set()
    for d in docs:
        key = d.title.lower()
        if key in seen:
            continue
        seen.add(key)
        cls = label_doc(d)
        if cls is not None:
            rows.append(
                {"text": d.title, "event": cls.value, "lang": d.language or "und", "source": "gdelt_weak"}
            )
    rows += edgar_rows()
    df = pd.DataFrame(rows)
    capped = [g.sample(min(len(g), args.cap), random_state=42) for _, g in df.groupby("event")]
    out = pd.concat(capped, ignore_index=True)
    rng = random.Random(42)
    out["split"] = [("val" if rng.random() < 0.1 else "train") for _ in range(len(out))]
    path = raw_dir() / "weak_events.parquet"
    out.to_parquet(path)
    log.info("weak labels: %d rows from %d docs -> %s", len(out), len(docs), path)
    log.info("per class:\n%s", out.groupby("event").size().to_string())


if __name__ == "__main__":
    main()
