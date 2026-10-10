"""Untouched v2 evaluation sets for the score-improvement round (DECISIONS, score improvement round).

    python -m training.v2_sets native     # -> data/eval/v2/native_v2_{dev,test}_to_label.csv
    python -m training.v2_sets credit     # -> data/eval/v2/credit_v2_{dev,test}_to_label.csv
    python -m training.v2_sets freeze     # sha256 of the labelled test files -> data/eval/v2/FREEZE.json

Every earlier set is "seen", so the v2 sets avoid it:
* native_v2: GDELT translingual headlines, 30 per language per split, from 15-minute windows whose
  calendar *dates* appear in no earlier set (weak-label, credit-mining and native-300 windows are
  regenerated from their seeds; impact windows, credit pools and replays are read from disk).
* credit_v2: English headlines from 2025-26 from a new source, Google News RSS search (many
  outlets), instead of the GDELT pools used for credit-150.
Both drop near-duplicates (MinHash Jaccard >= 0.8) of any earlier evaluation or training text and
of each other, so dev and test never share a story.
"""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import hashlib
import json
import logging
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd

from engine.config import get_settings
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import make_http_client
from engine.nlp.entities import EntityLinker
from engine.schema import RawDoc
from training.credit_label_set import LIKELY, drop_self_near_duplicates, training_texts
from training.data import drop_near_duplicates_of, raw_dir
from training.multilingual_set import LANGS

log = logging.getLogger("v2_sets")
PER_LANG = 30  # per language per split
OUT = get_settings().data_dir / "eval" / "v2"


def _stamps(n: int, seed: int, start: datetime, end: datetime) -> set[str]:
    rng = random.Random(seed)
    span = int((end - start).total_seconds() // 900)
    return {(start + timedelta(minutes=15 * rng.randrange(span))).strftime("%Y%m%d") for _ in range(n)}


def seen_dates() -> set[str]:
    """Calendar dates (YYYYMMDD) touched by any earlier GDELT sample, pool or replay."""
    d: set[str] = set()
    d |= _stamps(700, 7, datetime(2023, 1, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))  # weak labels
    d |= _stamps(900, 11, datetime(2020, 1, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))  # credit mining
    d |= _stamps(120, 2026, datetime(2025, 6, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))  # native-300
    for sub in ("impact_windows", "impact_burst"):
        d |= {f.stem[:8] for f in (raw_dir() / sub).glob("*.parquet")}
    g = pd.read_parquet(raw_dir() / "credit_parts" / "gdelt.parquet")
    d |= set(pd.to_datetime(g["date"]).dt.strftime("%Y%m%d"))
    ml = pd.read_csv(get_settings().data_dir / "eval" / "multilingual_to_label.csv")
    d |= set(pd.to_datetime(ml["published"]).dt.strftime("%Y%m%d"))
    for f in (get_settings().data_dir / "replay").rglob("*.jsonl.gz"):
        m = re.search(r"(\d{8})T", f.name) or re.search(r"(\d{4})-(\d{2})-(\d{2})", f.name)
        if m:
            d.add("".join(m.groups()) if len(m.groups()) == 3 else m.group(1))
    d |= {"20220224", "20200713"}  # scenario replay and tweet replay days
    return d


def earlier_eval_texts() -> list[str]:
    ev = get_settings().data_dir / "eval"
    t = pd.read_csv(ev / "multilingual_to_label.csv")["headline"].astype(str).tolist()
    t += pd.read_csv(ev / "credit_to_label.csv")["text"].astype(str).tolist()
    return t


# ------------------------------------------------------------------ native_v2
async def _native_docs(n_windows: int = 260, seed: int = 4242) -> list[RawDoc]:
    seen = seen_dates()
    rng = random.Random(seed)
    start, end = datetime(2024, 7, 1, tzinfo=UTC), datetime(2026, 10, 8, tzinfo=UTC)
    span = int((end - start).total_seconds() // 900)
    stamps: set[str] = set()
    while len(stamps) < n_windows:
        s = (start + timedelta(minutes=15 * rng.randrange(span))).strftime("%Y%m%d%H%M%S")
        if s[:8] not in seen:
            stamps.add(s)
    log.info("native_v2: %d windows on unseen dates (%d dates excluded)", len(stamps), len(seen))
    src = GdeltGkgSource(translation=True, max_docs_per_file=3000)
    docs: list[RawDoc] = []
    ordered = sorted(stamps)
    async with make_http_client(get_settings()) as client:
        for i in range(0, len(ordered), 6):
            batch = await asyncio.gather(
                *(src.fetch_window(client, s) for s in ordered[i : i + 6]), return_exceptions=True
            )
            for b in batch:
                if isinstance(b, list):
                    docs += [d for d in b if d.language in LANGS]
            if i % 60 == 0:
                log.info("windows %d/%d, docs %d", i, len(ordered), len(docs))
    return docs


def build_native() -> None:
    docs = asyncio.run(_native_docs())
    linker = EntityLinker()
    rows = []
    for d in docs:
        if 12 <= len(d.title) <= 220:
            rows.append({"lang": d.language, "headline": d.title, "published": d.published_at.strftime("%Y-%m-%d"),
                         "source_url": d.url, "linked": bool(linker.link(d.title).mentions)})  # fmt: skip
    c = pd.DataFrame(rows).drop_duplicates("headline")
    c = c.rename(columns={"headline": "text"})
    c = drop_near_duplicates_of(c, training_texts() + earlier_eval_texts())
    c = drop_self_near_duplicates(c, threshold=0.8).rename(columns={"text": "headline"})
    rng = random.Random(7)
    dev, test = [], []
    for lang in LANGS:
        pool = c[c["lang"] == lang]
        linked = pool[pool["linked"]].sample(frac=1.0, random_state=11)
        other = pool[~pool["linked"]].sample(frac=1.0, random_state=11)
        need = 2 * PER_LANG
        # same recipe as native-300: prefer headlines naming a tracked entity, topped up with others
        pick = pd.concat([linked.head(need * 2 // 3), other]).head(need)
        if len(pick) < need:
            log.warning("%s: only %d headlines", lang, len(pick))
        idx = list(pick.index)
        rng.shuffle(idx)
        half = len(idx) // 2
        test.append(pick.loc[idx[:half]])
        dev.append(pick.loc[idx[half:]])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, parts in (("dev", dev), ("test", test)):
        df = pd.concat(parts).reset_index(drop=True)
        df.insert(0, "id", [f"N{name[0].upper()}{i + 1:03d}" for i in range(len(df))])
        df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        df[["id", "lang", "headline", "published", "source_url"]].to_csv(
            OUT / f"native_v2_{name}_to_label.csv", index=False
        )
        log.info("native_v2_%s: %d rows %s", name, len(df), df["lang"].value_counts().to_dict())


# ------------------------------------------------------------------ credit_v2
QUERIES = [
    # likely credit events
    "defaults on bonds",
    "files for bankruptcy",
    "chapter 11 filing",
    "missed coupon payment",
    "debt restructuring creditors",
    "downgraded by Moody's",
    "S&P cuts rating",
    "Fitch downgrades",
    "ICRA downgrades",
    "CRISIL downgrade rating",
    "CARE Ratings downgrade",
    "insolvency proceedings NCLT",
    "liquidation creditors",
    "distressed debt exchange",
    "sovereign default",
    "covenant breach lenders",
    # hard negatives
    "rating affirmed stable outlook",
    "Moody's affirms",
    "Fitch affirms",
    "upgraded by S&P",
    "outlook revised to positive",
    "analyst downgrades stock to sell",
    "downgraded to neutral price target",
    "price target cut",
    "cuts price target",
    "layoffs restructuring plan",
    "job cuts restructuring",
    "credit card spending",
    "tax credit",
    "carbon credits",
    "loan growth bank",
    "credit growth RBI",
    "private credit fund raises",
]
NEG = {
    "analyst_call": re.compile(
        r"price target|\bpt\b|to (sell|neutral|hold|underperform|underweight|equal.weight|market perform)|analyst",
        re.I,
    ),
    "layoffs_restructuring": re.compile(
        r"layoff|job cuts|jobs|workforce|headcount|restructuring plan|reorgani[sz]", re.I
    ),
    "affirmation": re.compile(r"\baffirm", re.I),
    "upgrade": re.compile(r"\bupgrad", re.I),
    "outlook_stable_positive": re.compile(r"outlook.{0,30}(stable|positive)|(stable|positive) outlook", re.I),
    "credit_lookalike": re.compile(
        r"credit card|tax credit|carbon credit|loan growth|credit growth|private credit", re.I
    ),
}
N_PER_SPLIT, N_LIKELY = 150, 55


def _bucket(t: str) -> str | None:
    for k, rx in NEG.items():
        if rx.search(t):
            return k
    return "likely_credit" if LIKELY.search(t) else None


async def _gnews(query: str, hl: str, gl: str) -> list[dict[str, str]]:
    url = (f"https://news.google.com/rss/search?q={quote_plus(query + ' after:2025-01-01')}"
           f"&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}")  # fmt: skip
    async with make_http_client(get_settings()) as client:
        r = await client.get(url)
    out = []
    for it in ElementTree.fromstring(r.text).iter("item"):
        title, src = it.findtext("title") or "", it.findtext("source") or ""
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3]
        pub = email.utils.parsedate_to_datetime(it.findtext("pubDate") or "Thu, 01 Jan 1970 00:00:00 GMT")
        out.append({"text": title.strip(), "outlet": src, "date": pub.strftime("%Y-%m-%d"), "query": query})
    return out


def build_credit() -> None:
    async def all_queries() -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for q in QUERIES:
            for hl, gl in (("en-US", "US"), ("en-IN", "IN"), ("en-GB", "GB")):
                try:
                    rows += await _gnews(q, hl, gl)
                except Exception as e:
                    log.warning("query %r %s failed: %s", q, gl, e)
                await asyncio.sleep(1.0)
        return rows

    c = pd.DataFrame(asyncio.run(all_queries()))
    c = c[(c["date"] >= "2025-01-01") & c["text"].str.len().between(25, 220)]
    c = c[~c["text"].str.contains(r"[^\x00-\x7F‘’“”–—… -ÿ]")]
    c["key"] = c["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    c = c.drop_duplicates("key")
    c["bucket"] = c["text"].map(_bucket)
    c = c[c["bucket"].notna()]
    c = drop_near_duplicates_of(c, training_texts() + earlier_eval_texts())
    c = (
        drop_self_near_duplicates(c, threshold=0.7)
        .sample(frac=1.0, random_state=20261010)
        .reset_index(drop=True)
    )
    log.info("credit candidates %d, buckets %s", len(c), c["bucket"].value_counts().to_dict())
    likely = c[c["bucket"] == "likely_credit"].head(2 * N_LIKELY)
    negs = c[c["bucket"] != "likely_credit"]
    per = -(-2 * (N_PER_SPLIT - N_LIKELY) // len(NEG))
    pick_neg = negs.groupby("bucket", group_keys=False).head(per)
    if len(pick_neg) < 2 * (N_PER_SPLIT - N_LIKELY):
        pick_neg = pd.concat(
            [pick_neg, negs.drop(pick_neg.index).head(2 * (N_PER_SPLIT - N_LIKELY) - len(pick_neg))]
        )
    sel = pd.concat([likely, pick_neg.head(2 * (N_PER_SPLIT - N_LIKELY))])
    # stratified split by bucket so dev and test have the same mix
    sel["split"] = ""
    for _, g in sel.groupby("bucket"):
        ids = list(g.index)
        random.Random(5).shuffle(ids)
        sel.loc[ids[: len(ids) // 2], "split"] = "test"
        sel.loc[ids[len(ids) // 2 :], "split"] = "dev"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("dev", "test"):
        df = sel[sel["split"] == name].sample(frac=1.0, random_state=9).reset_index(drop=True)
        df.insert(0, "id", [f"K{name[0].upper()}{i + 1:03d}" for i in range(len(df))])
        df[["id", "text", "date", "outlet"]].to_csv(OUT / f"credit_v2_{name}_to_label.csv", index=False)
        df[["id", "bucket", "query"]].to_csv(OUT / f"credit_v2_{name}_selection.csv", index=False)
        log.info("credit_v2_%s: %d rows %s", name, len(df), df["bucket"].value_counts().to_dict())


# ------------------------------------------------------------------ freeze
def freeze() -> None:
    files = sorted(OUT.glob("*_test_labelled.csv")) + sorted(OUT.glob("*_test_to_label.csv"))
    fz = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    fz["frozen_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (OUT / "FREEZE.json").write_text(json.dumps(fz, indent=2) + "\n")
    log.info("froze %s", list(fz))


def check_frozen(name: str) -> Path:
    """Path of a frozen v2 test file; raises if its content changed after freezing."""
    f = OUT / name
    fz = json.loads((OUT / "FREEZE.json").read_text())
    if hashlib.sha256(f.read_bytes()).hexdigest() != fz[name]:
        raise SystemExit(f"{name} changed after it was frozen")
    return f


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["native", "credit", "freeze"])
    a = ap.parse_args()
    {"native": build_native, "credit": build_credit, "freeze": freeze}[a.what]()


if __name__ == "__main__":
    main()
