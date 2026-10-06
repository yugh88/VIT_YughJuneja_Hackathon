"""Build the Credit Event labelling set (never used in training or tuning).

    python -m training.credit_label_set     # -> data/eval/credit_to_label.csv (+ selection file)

~150 English headlines dated 2024+:
* ~60 likely credit events (distress wording), and
* ~90 hard negatives: affirmations, upgrades, stable/positive outlooks, credit cards, tax/carbon
  credits, AML/fines, loan/credit growth.

Pools: GDELT credit candidates dated 2025+ (``credit_parts/gdelt.parquet``; ``finalize`` kept only
pre-2025 rows for training) and GDELT headlines from the impact windows dated 2024+ (used only as
impact-model *test* data). Any candidate that near-duplicates (MinHash Jaccard >= 0.8) a text from
any event/credit training source or the frozen event test is dropped.

The selection bucket is written to a separate file (``credit_to_label_selection.csv``) so the
labelling file carries no hint of it. Rows are shuffled with a fixed seed.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from engine.config import get_settings
from training.data import (
    drop_near_duplicates_of,
    frozen_test_texts,
    load_credit_events,
    load_credit_translations,
    load_translated_events,
    load_twitter_topic,
    load_weak_events,
    raw_dir,
)

log = logging.getLogger("credit_label_set")
SEED = 20261005
N_LIKELY, N_NEG = 60, 90
LIKELY = re.compile(
    r"\b(defaults?|defaulted|defaulting|bankrupt\w*|insolven\w*|chapter 11|downgrad\w*|restructur\w*|"
    r"missed (a |an )?(payment|coupon|interest)|liquidat\w*|receivership|administration|nclt|ibc|"
    r"covenant breach|creditors|selective default|debt crisis|junk)\b",
    re.I,
)
NEG_BUCKETS = {
    "affirmation": re.compile(r"\baffirm\w*\b", re.I),
    "upgrade": re.compile(r"\bupgrad\w*\b", re.I),
    "outlook_stable_positive": re.compile(
        r"\boutlook\b.{0,30}\b(stable|positive)\b|\b(stable|positive) outlook\b", re.I
    ),
    "credit_card": re.compile(r"\bcredit cards?\b", re.I),
    "tax_carbon_credit": re.compile(r"\b(tax|carbon|energy|solar|ev) credits?\b", re.I),
    "aml_fine": re.compile(r"money laundering|\banti-money\b|\baml\b|\bfined\b|\bpenalt(y|ies)\b", re.I),
    "loan_growth": re.compile(r"\b(loan|credit|lending) growth\b|\bcredit (demand|offtake)\b", re.I),
}


def candidates() -> pd.DataFrame:
    parts = []
    g = pd.read_parquet(raw_dir() / "credit_parts" / "gdelt.parquet")
    g["date"] = pd.to_datetime(g["date"])
    g = g[(g["date"] >= "2025-01-01") & (g["lang"] == "en")]
    parts.append(
        pd.DataFrame(
            {
                "text": g["text"],
                "date": g["date"].dt.date.astype(str),
                "outlet": "",
                "pool": "gdelt_credit_2025+",
            }
        )
    )
    for sub in ("impact_windows", "impact_burst"):
        for f in sorted((raw_dir() / sub).glob("*.parquet")):
            if f.stem < "20240101" or not f.stat().st_size:
                continue
            d = pd.read_parquet(f)
            if "title" not in d:  # empty window
                continue
            d = d[["title", "outlet"]].dropna()
            parts.append(pd.DataFrame({"text": d["title"], "date": f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}",
                                       "outlet": d["outlet"], "pool": f"{sub}_2024+"}))  # fmt: skip
    c = pd.concat(parts, ignore_index=True)
    c["text"] = c["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    c = c[c["text"].str.len().between(25, 220)]
    c = c[~c["text"].str.contains(r"[^\x00-\x7F‘’“”–—… -ÿ]")]  # English/Latin-1 only
    c["key"] = c["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    return c.drop_duplicates("key").reset_index(drop=True)


def training_texts() -> list[str]:
    texts: set[str] = set()
    for df in (load_twitter_topic(), load_weak_events(), load_credit_events(), load_credit_translations()):
        texts |= set(df["text"].astype(str))
    tr = load_translated_events()
    texts |= set(tr["text"].astype(str))
    if "text_en" in tr:
        texts |= set(tr["text_en"].dropna().astype(str))
    texts |= set(frozen_test_texts())
    return sorted(texts)


def bucket(text: str) -> str | None:
    for name, rx in NEG_BUCKETS.items():
        if rx.search(text):
            return name
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    c = candidates()
    c["likely"] = c["text"].map(lambda t: bool(LIKELY.search(t)))
    c["neg_bucket"] = [None if lk else bucket(t) for t, lk in zip(c["text"], c["likely"], strict=True)]
    c = c[c["likely"] | c["neg_bucket"].notna()]
    log.info(
        "candidates: %d likely, %d hard-negative", int(c["likely"].sum()), int(c["neg_bucket"].notna().sum())
    )
    c = drop_near_duplicates_of(c, training_texts())
    # near-duplicates among the candidates themselves (syndicated copies): keep the first
    c = drop_self_near_duplicates(c)
    likely = c[c["likely"]].sample(frac=1.0, random_state=SEED)
    likely = pd.concat([likely[likely["pool"].str.startswith("gdelt")].head(N_LIKELY // 2),
                        likely[~likely["pool"].str.startswith("gdelt")]]).head(N_LIKELY)  # fmt: skip
    negs = c[c["neg_bucket"].notna()].sample(frac=1.0, random_state=SEED)
    per = -(-N_NEG // len(NEG_BUCKETS))
    pool, negs = negs, negs.groupby("neg_bucket", group_keys=False).head(per)
    if len(negs) < N_NEG:  # a thin bucket (stable/positive outlooks are rare) is topped up from the rest
        negs = pd.concat([negs, pool.drop(negs.index).head(N_NEG - len(negs))])
    if len(negs) > N_NEG:
        negs = negs.sample(N_NEG, random_state=SEED)
    sel = pd.concat([likely.assign(bucket="likely_credit"), negs.assign(bucket=negs["neg_bucket"])])
    sel = sel.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    sel["id"] = [f"C{i + 1:03d}" for i in range(len(sel))]
    out = get_settings().data_dir / "eval"
    sel[["id", "text", "date", "outlet"]].to_csv(out / "credit_to_label.csv", index=False)
    sel[["id", "bucket", "pool"]].to_csv(out / "credit_to_label_selection.csv", index=False)
    log.info("credit_to_label.csv: %d rows; buckets %s", len(sel), sel["bucket"].value_counts().to_dict())


def drop_self_near_duplicates(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    from datasketch import MinHash, MinHashLSH

    from engine.nlp.dedupe import shingles

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep = []
    for i, t in enumerate(df["text"]):
        m = MinHash(num_perm=128)
        for sh in shingles(str(t)):
            m.update(sh.encode("utf-8"))
        if lsh.query(m):
            keep.append(False)
            continue
        lsh.insert(str(i), m)
        keep.append(True)
    return df[keep]


if __name__ == "__main__":
    main()
