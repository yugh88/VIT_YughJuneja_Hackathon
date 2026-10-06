"""Distant supervision for the Credit Event class (Step B of the credit work).

    python -m training.credit_labels edgar      # SEC 8-K filings: the item code is the label
    python -m training.credit_labels gdelt      # GDELT headlines kept only when 2+ independent rules agree
    python -m training.credit_labels all

Outputs
* ``data/raw/credit_events.parquet``: training/validation rows (positives and hard negatives);
  filed/published before 2025-01-01 only.
* ``data/eval/edgar_credit_test.csv.gz``: every 8-K credit filing (items 1.03 / 2.04 / 3.01) and a
  sample of non-credit 8-Ks filed **on or after 2025-01-01**. Never used for training or tuning.

Rules for GDELT (a headline needs 2+ rules, at least one of them specific (R1/R3/R4), to count as
Credit; R2 and R5 alone fire on any negative debt story, so they are not independent evidence):
  R1 unambiguous credit phrase in the headline's own language (engine.nlp.credit)
  R2 GDELT tagged the article with a credit/insolvency theme
  R3 rating-agency action wording (Moody's/S&P/Fitch + downgrade/outlook negative/watch negative)
  R4 India-specific insolvency process (NCLT / IBC / insolvency resolution)
  R5 strongly negative GDELT tone (< -4) together with a debt word
Hard negatives (credit lookalikes) become non-Credit rows: credit cards, tax/carbon credits,
"Credit Suisse"/"Credit Agricole" as names without distress words, credit/loan growth, upgrades.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import random
import re
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from engine.config import get_settings
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import make_http_client
from engine.ingest.rss import EDGAR_ITEM_EVENTS
from engine.nlp.credit import credit_phrase
from engine.schema import EventClass, RawDoc
from training.data import raw_dir

log = logging.getLogger("credit_labels")
CREDIT = EventClass.CREDIT_EVENT.value
CUTOFF = date(2025, 1, 1)  # filings/headlines on or after this date are test-only
CREDIT_ITEMS = {"1.03", "2.04", "3.01"}
ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement", "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership", "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets", "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities", "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities", "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant", "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant", "5.02": "Departure or Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws", "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure", "8.01": "Other Events", "9.01": "Financial Statements and Exhibits",
}  # fmt: skip
EFTS = "https://efts.sec.gov/LATEST/search-index"


def edgar_text(company: str, items: list[str]) -> str:
    """Same shape the live EDGAR connector produces (engine.ingest.rss.EdgarSource)."""
    parts = [f"Item {i} {ITEM_TITLES[i]}" for i in items if i in ITEM_TITLES and i != "9.01"]
    return f"{company} files 8-K: " + "; ".join(parts) if parts else f"{company} files 8-K"


def edgar_label(items: list[str]) -> str:
    if CREDIT_ITEMS & set(items):
        return CREDIT
    weak = [
        EDGAR_ITEM_EVENTS[i]
        for i in items
        if i in EDGAR_ITEM_EVENTS and EDGAR_ITEM_EVENTS[i] != "Other/Non-event"
    ]
    return weak[0] if len(set(weak)) == 1 else EventClass.OTHER.value


def _clean_company(display: str) -> str:
    return re.sub(r"\s*\((?:[A-Z.\-]+|CIK \d+)\)", "", display).strip()


def fetch_edgar() -> pd.DataFrame:
    queries = {"credit": ['"Bankruptcy or Receivership"', '"Triggering Events That Accelerate"', '"Notice of Delisting"'],
               "other": ['"Results of Operations and Financial Condition"', '"Departure of Directors"',
                         '"Entry into a Material Definitive Agreement"']}  # fmt: skip
    rows: dict[str, dict[str, Any]] = {}
    headers = {"User-Agent": get_settings().user_agent}
    with httpx.Client(headers=headers, timeout=60) as client:
        month = date(2019, 1, 1)
        while month <= date(2026, 9, 1):
            nxt = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
            for kind, qs in queries.items():
                for q in qs if kind == "credit" else [random.Random(month.toordinal()).choice(qs)]:
                    params = {
                        "q": q,
                        "forms": "8-K",
                        "dateRange": "custom",
                        "startdt": month.isoformat(),
                        "enddt": (nxt - timedelta(days=1)).isoformat(),
                    }
                    for attempt in range(3):
                        try:
                            hits = client.get(EFTS, params=params).raise_for_status().json()["hits"]["hits"]
                            break
                        except (httpx.HTTPError, KeyError, ValueError):
                            time.sleep(2 * (attempt + 1))
                            hits = []
                    for h in hits:
                        src = h["_source"]
                        items = sorted(set(src.get("items") or []))
                        if not items or not src.get("display_names"):
                            continue
                        company = _clean_company(src["display_names"][0])
                        rows[src["adsh"]] = {
                            "text": edgar_text(company, items),
                            "event": edgar_label(items),
                            "items": ";".join(items),
                            "date": src["file_date"],
                            "source": "edgar_8k",
                            "lang": "en",
                            "cluster": src["ciks"][0],
                        }
                    time.sleep(0.25)  # SEC fair-access: well under 10 requests/s
            log.info("edgar %s: %d filings so far", month.isoformat()[:7], len(rows))
            month = nxt
    return pd.DataFrame(rows.values())


# ----------------------------------------------------------------------------- GDELT, multi-rule
CREDIT_THEMES = (
    "ECON_BANKRUPTCY",
    "ECON_DEBT",
    "ECON_CREDITRATINGS",
    "WB_1235",
    "ECON_DEFAULT",
    "WB_1920_FINANCIAL_SECTOR_DEVELOPMENT",
)
R3 = re.compile(
    r"\b(moody'?s|s&p|standard (&|and) poor'?s|fitch|icra|crisil|care ratings|india ratings)\b.{0,60}"
    r"\b(downgrad\w*|cuts?|lowers?|outlook (to )?negative|negative outlook|(credit ?)?watch negative|"
    r"review for (possible )?downgrade|default)\b|"
    r"\b(downgrad\w*|outlook (to )?negative|watch negative)\b.{0,40}\b(moody'?s|s&p|fitch|icra|crisil)\b",
    re.I,
)
R4 = re.compile(
    r"\b(nclt|ibc|insolvency (and bankruptcy|resolution)|corporate insolvency|resolution plan|cirp)\b", re.I
)
DEBT_WORD = re.compile(
    r"\b(debt|bonds?|loans?|lenders?|creditors?|coupon|notes|borrow\w*)\b|债|債|ऋण|कर्ज|dívida|deuda|dette|schulden|долг",
    re.I,
)
HARD_NEG = re.compile(
    r"\bcredit cards?\b|\btax credits?\b|\bcarbon credits?\b|\bcredit (growth|demand|offtake)\b|\bloan growth\b|"
    r"\bupgrad\w*\b|\bcredit (score|limit|line increase)\b|\bcredit (suisse|agricole|mutuel|karma|acceptance)\b",
    re.I,
)
DISTRESS = re.compile(
    r"default|bankrupt|insolv|downgrad|restructur|liquidat|collapse|crisis|rescue|bailout|run on", re.I
)


def gdelt_rules(d: RawDoc) -> tuple[int, list[str]]:
    title, lang = d.title, d.language or "en"
    themes = d.meta.get("gdelt_themes", [])
    tone = (d.meta.get("gdelt_tone") or {}).get("tone", 0.0)
    fired = []
    if credit_phrase(title, lang):
        fired.append("R1")
    if any(t.startswith(CREDIT_THEMES) for t in themes):
        fired.append("R2")
    if R3.search(title):
        fired.append("R3")
    if R4.search(title):
        fired.append("R4")
    if tone < -4 and DEBT_WORD.search(title):
        fired.append("R5")
    return len(fired), fired


async def _gdelt_windows(n: int, seed: int = 11) -> list[RawDoc]:
    rng = random.Random(seed)
    # GKG files carry <PAGE_TITLE> only from mid-2019 on; earlier windows parse to nothing.
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
    span = int((end - start).total_seconds() // 900)
    stamps = sorted(
        {(start + timedelta(minutes=15 * rng.randrange(span))).strftime("%Y%m%d%H%M%S") for _ in range(n)}
    )
    src = GdeltGkgSource(translation=True, max_docs_per_file=None)  # type: ignore[arg-type]
    out: list[RawDoc] = []
    sem = asyncio.Semaphore(6)
    async with make_http_client(get_settings()) as client:

        async def one(stamp: str) -> list[RawDoc]:
            async with sem:
                try:
                    return await src.fetch_window(client, stamp)
                except Exception as exc:
                    log.warning("window %s: %s: %s", stamp, type(exc).__name__, exc)
                    return []

        for i in range(0, len(stamps), 60):
            for docs in await asyncio.gather(*(one(s) for s in stamps[i : i + 60])):
                # keep only candidates: any credit evidence or a lookalike (hard negative)
                out += [d for d in docs if gdelt_rules(d)[0] >= 1 or HARD_NEG.search(d.title)]
            log.info("gdelt windows %d/%d, candidates %d", min(i + 60, len(stamps)), len(stamps), len(out))
    return out


def label_gdelt(docs: list[RawDoc]) -> pd.DataFrame:
    rows, seen = [], set()
    for d in docs:
        key = d.title.lower()
        if key in seen or len(d.title) < 15:
            continue
        seen.add(key)
        n, fired = gdelt_rules(d)
        lookalike = bool(HARD_NEG.search(d.title)) and not DISTRESS.search(d.title)
        specific = bool({"R1", "R3", "R4"} & set(fired))  # R2/R5 are correlated generic-debt signals
        if n >= 2 and specific and not lookalike:
            event = CREDIT
        elif (lookalike and n == 0) or (lookalike and fired == ["R2"]):
            event = EventClass.OTHER.value  # hard negative: credit vocabulary, no credit event
        else:
            continue  # one weak rule only: not trusted either way
        rows.append(
            {
                "text": d.title,
                "event": event,
                "rules": ",".join(fired) or "hard_negative",
                "lang": d.language or "und",
                "date": d.published_at.date().isoformat(),
                "source": "gdelt_credit",
                "cluster": "",
            }
        )
    return pd.DataFrame(rows)


def finalize(edgar: pd.DataFrame | None, gdelt: pd.DataFrame | None) -> None:
    if gdelt is not None and len(gdelt):  # apply the "one specific rule" requirement to cached labels too
        rules = gdelt["rules"].fillna("")
        weak_only = (gdelt["event"] == CREDIT) & ~rules.str.contains("R1|R3|R4")
        gdelt = gdelt[~weak_only]
    parts = [p for p in (edgar, gdelt) if p is not None and len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    test = df[(df["source"] == "edgar_8k") & (df["date"] >= CUTOFF)]
    train = df[df["date"] < CUTOFF].copy()
    # Validation: 15% held out by *cluster* (EDGAR: filer CIK) or by headline hash, never by row,
    # so a filer's repeated filings or a syndicated headline never straddle train and validation.
    key = train["cluster"].where(train["cluster"] != "", train["text"].str.lower())
    train["split"] = [
        "val" if int(hashlib.sha1(k.encode()).hexdigest(), 16) % 100 < 15 else "train" for k in key
    ]
    out = raw_dir() / "credit_events.parquet"
    train.drop(columns=["cluster"]).to_parquet(out)
    tpath = get_settings().data_dir / "eval" / "edgar_credit_test.csv.gz"
    test.drop(columns=["cluster"]).to_csv(tpath, index=False)
    log.info(
        "credit data: %s; by source/event:\n%s",
        out,
        train.groupby(["source", "event", "split"]).size().to_string(),
    )
    log.info(
        "EDGAR test (>= %s): %d rows, %d credit -> %s",
        CUTOFF,
        len(test),
        (test["event"] == CREDIT).sum(),
        tpath,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["edgar", "gdelt", "all"])
    ap.add_argument("--windows", type=int, default=900)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cache = raw_dir() / "credit_parts"
    cache.mkdir(parents=True, exist_ok=True)
    if args.what in {"edgar", "all"}:
        fetch_edgar().to_parquet(cache / "edgar.parquet")
    if args.what in {"gdelt", "all"}:
        label_gdelt(asyncio.run(_gdelt_windows(args.windows))).to_parquet(cache / "gdelt.parquet")
    e = pd.read_parquet(cache / "edgar.parquet") if (cache / "edgar.parquet").exists() else None
    g = pd.read_parquet(cache / "gdelt.parquet") if (cache / "gdelt.parquet").exists() else None
    finalize(e, g)


if __name__ == "__main__":
    main()
