"""Training/evaluation dataset loaders with fixed, seeded splits.

Every loader returns a DataFrame with at least ``text``, ``label`` and ``split``
(``train`` / ``val`` / ``test``). Sentiment labels are always 0 = negative,
1 = neutral, 2 = positive. Splits are deterministic (seed 42) so training,
evaluation and the model card all see the same held-out rows.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import get_settings
from engine.schema import EventClass

SEED = 42
SENT_LABELS = ["negative", "neutral", "positive"]
EVENT_LABELS = [e.value for e in EventClass]


def raw_dir() -> Path:
    return get_settings().data_dir / "raw"


def _stratified_split(
    df: pd.DataFrame, label_col: str = "label", val: float = 0.1, test: float = 0.1
) -> pd.Series:
    rng = np.random.default_rng(SEED)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby(label_col).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_test, n_val = round(len(idx) * test), round(len(idx) * val)
        split[idx[:n_test]] = "test"
        split[idx[n_test : n_test + n_val]] = "val"
    return split


# ----------------------------------------------------------------------------- sentiment
def load_phrasebank() -> pd.DataFrame:
    """Financial PhraseBank, 50%-agreement set (4,846 sentences) with an ``all_agree`` flag."""
    path = raw_dir() / "financial_phrasebank" / "FinancialPhraseBank-v1.0.zip"
    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def read(name: str) -> list[tuple[str, int]]:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read(f"FinancialPhraseBank-v1.0/Sentences_{name}.txt").decode("latin-1")
        rows = []
        for line in raw.splitlines():
            if "@" not in line:
                continue
            text, lab = line.rsplit("@", 1)
            if lab.strip() in label_map:
                rows.append((text.strip(), label_map[lab.strip()]))
        return rows

    df = (
        pd.DataFrame(read("50Agree"), columns=["text", "label"])
        .drop_duplicates("text")
        .reset_index(drop=True)
    )
    all_agree = {t for t, _ in read("AllAgree")}
    df["all_agree"] = df["text"].isin(all_agree)
    df["split"] = _stratified_split(df)
    df["source"] = "phrasebank"
    df["entity"] = ""
    return df


def load_twitter_sentiment() -> pd.DataFrame:
    """zeroshot/twitter-financial-news-sentiment: HF 'validation' is our test set."""
    d = raw_dir() / "twitter_fin_sentiment"
    remap = {0: 0, 1: 2, 2: 1}  # Bearish, Bullish, Neutral -> neg, pos, neu
    tr = pd.read_parquet(d / "train.parquet")
    te = pd.read_parquet(d / "validation.parquet")
    tr["split"] = _stratified_split(tr, val=0.1, test=0.0)
    te["split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    df["label"] = df["label"].map(remap)
    df["text"] = df["text"].str.replace(r"https?://\S+", "", regex=True).str.strip()
    df["source"] = "twitter_sentiment"
    df["entity"] = ""
    return df[["text", "label", "split", "source", "entity"]]


def load_sentfin() -> pd.DataFrame:
    """SEntFiN 1.0 headlines (entity-level labels), from the Hugging Face mirror."""
    d = pd.read_parquet(raw_dir() / "sentfin" / "train.parquet")
    d = d[d["Source"] == "SentFIN"].rename(
        columns={"Sentences": "text", "Labels": "label", "Entities": "entity"}
    )
    d = d[["text", "label", "entity"]].dropna(subset=["text"]).reset_index(drop=True)
    d["entity"] = d["entity"].fillna("")
    d["split"] = _stratified_split(d)
    d["source"] = "sentfin"
    return d


def load_fiqa() -> pd.DataFrame:
    """FiQA-2018 task 1: continuous score in [-1, 1] per (sentence, target)."""
    d = raw_dir() / "fiqa_sentiment"
    parts = []
    for fname, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        p = pd.read_parquet(d / f"{fname}.parquet")
        p["split"] = split
        parts.append(p)
    df = pd.concat(parts, ignore_index=True).rename(columns={"sentence": "text", "target": "entity"})
    df["label"] = np.select([df["score"] < -0.1, df["score"] > 0.1], [0, 2], 1)
    df["source"] = "fiqa"
    return df[["text", "entity", "score", "label", "split", "source", "type"]]


def load_hindi_idioms() -> pd.DataFrame:
    """~30 native Hindi finance sentences written for training only (never in any eval set)."""
    p = get_settings().data_dir / "train_extra" / "hindi_finance_idioms.csv"
    df = pd.read_csv(p)
    return df.assign(split="train", source="hindi_idioms", entity="", lang="hi")[
        ["text", "label", "split", "source", "entity", "lang"]
    ]


def load_translated_sentiment() -> pd.DataFrame:
    """PhraseBank machine-translated with NLLB (see ``training.translate``)."""
    p = raw_dir() / "translated" / "phrasebank.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["text", "label", "split", "lang"])


# ----------------------------------------------------------------------------- events
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
    "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
    "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip

# Topic -> our 12-class taxonomy. Topics with no clean counterpart map to Other.
TOPIC_TO_EVENT = {
    "Analyst Update": EventClass.EARNINGS_GUIDANCE,
    "Fed | Central Banks": EventClass.MACROECONOMIC,
    "Company | Product News": EventClass.PRODUCT_LAUNCH,
    "Treasuries | Corporate Debt": EventClass.MACROECONOMIC,
    "Dividend": EventClass.EARNINGS_GUIDANCE,
    "Earnings": EventClass.EARNINGS_GUIDANCE,
    "Energy | Oil": EventClass.MACROECONOMIC,
    "Financials": EventClass.EARNINGS_GUIDANCE,
    "Currencies": EventClass.MACROECONOMIC,
    "General News | Opinion": EventClass.OTHER,
    "Gold | Metals | Materials": EventClass.MACROECONOMIC,
    "IPO": EventClass.OTHER,
    "Legal | Regulation": EventClass.REGULATORY_LEGAL,
    "M&A | Investments": EventClass.MERGER_ACQUISITION,
    "Macro": EventClass.MACROECONOMIC,
    "Markets": EventClass.OTHER,
    "Politics": EventClass.GEOPOLITICAL,
    "Personnel Change": EventClass.MANAGEMENT_CHANGE,
    "Stock Commentary": EventClass.OTHER,
    "Stock Movement": EventClass.OTHER,
}


_LAUNCH = re.compile(
    r"\b(launch\w*|unveil\w*|introduc\w*|debut\w*|rolls? out|rolled out|releases?|new (product|model|phone|chip|feature|service)|"
    r"announces? (new|the)|now available|approval for|approves?|fda)\b",
    re.I,
)
_CREDIT = re.compile(
    r"\b(default\w*|bankrupt\w*|downgrad\w*|insolven\w*|restructur\w*|creditors?|bondholders?|distress\w*|"
    r"junk|missed|chapter 11|liquidat\w*|rating cut)\b",
    re.I,
)


_EARNINGS = re.compile(
    r"\b(earnings|results|quarter(ly)?|q[1-4]|profit|loss(es)?|revenue|sales|guidance|forecast|outlook|estimates?|"
    r"eps|beats?|miss(es|ed)?|dividend)\b",
    re.I,
)


def _refine(topic: str, text: str) -> str:
    """Topic->event mapping, sharpened with keywords where a topic mixes several events."""
    if topic == "Company | Product News":
        return EventClass.PRODUCT_LAUNCH.value if _LAUNCH.search(text) else EventClass.OTHER.value
    if topic in {"Stock Movement", "Stock Commentary"} and _EARNINGS.search(text):
        return EventClass.EARNINGS_GUIDANCE.value  # "shares slump after loss" is about earnings
    if topic == "Treasuries | Corporate Debt" and _CREDIT.search(text):
        return EventClass.CREDIT_EVENT.value
    return TOPIC_TO_EVENT[topic].value


def load_twitter_topic() -> pd.DataFrame:
    d = raw_dir() / "twitter_fin_topic"
    tr = pd.read_parquet(d / "train.parquet")
    te = pd.read_parquet(d / "validation.parquet")
    tr["split"] = _stratified_split(tr, val=0.1, test=0.0)
    te["split"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    df["topic"] = df["label"].map(dict(enumerate(TOPIC_NAMES)))
    df["text"] = df["text"].str.replace(r"https?://\S+", "", regex=True).str.strip()
    df["event"] = [_refine(t, x) for t, x in zip(df["topic"], df["text"], strict=True)]
    df["source"] = "twitter_topic"
    df["lang"] = "en"
    return df[["text", "event", "topic", "split", "source", "lang"]]


def load_weak_events() -> pd.DataFrame:
    """Weakly labelled multilingual headlines built by ``training.weak_labels``."""
    p = raw_dir() / "weak_events.parquet"
    return (
        pd.read_parquet(p)
        if p.exists()
        else pd.DataFrame(columns=["text", "event", "split", "source", "lang"])
    )


def load_translated_events() -> pd.DataFrame:
    """NLLB translations; labels are re-derived from the English source so mapping changes propagate."""
    p = raw_dir() / "translated" / "events.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["text", "event", "split", "lang"])
    df = pd.read_parquet(p)
    topic = load_twitter_topic()
    current = dict(zip(topic["text"], topic["event"], strict=False))
    mt = df["source"] == "twitter_topic_mt"
    df.loc[mt, "event"] = df.loc[mt, "text_en"].map(current).fillna(df.loc[mt, "event"])
    return df


def load_translated_twitter() -> tuple[pd.DataFrame, pd.DataFrame]:
    """D73(d) translate-train extension: (sentiment rows, topic rows); empty frames until translated."""
    d = raw_dir() / "translated"
    ps, pt = d / "twitter_sentiment.parquet", d / "twitter_topic.parquet"
    sent = (
        pd.read_parquet(ps)
        if ps.exists()
        else pd.DataFrame(columns=["text", "label", "split", "lang", "source"])
    )
    top = (
        pd.read_parquet(pt)
        if pt.exists()
        else pd.DataFrame(columns=["text", "event", "split", "lang", "source"])
    )
    if len(top):
        current = dict(zip(load_twitter_topic()["text"], load_twitter_topic()["event"], strict=False))
        top["event"] = top["text_en"].map(current).fillna(top["event"])
    return sent, top


def load_credit_events() -> pd.DataFrame:
    """Credit distant supervision (training.credit_labels): EDGAR item codes, multi-rule GDELT, hard negatives."""
    p = raw_dir() / "credit_events.parquet"
    cols = ["text", "event", "split", "source", "lang"]
    return pd.read_parquet(p)[cols] if p.exists() else pd.DataFrame(columns=cols)


def load_credit_translations() -> pd.DataFrame:
    p = raw_dir() / "translated" / "credit.parquet"
    cols = ["text", "event", "split", "source", "lang"]
    return pd.read_parquet(p)[cols] if p.exists() else pd.DataFrame(columns=cols)


def frozen_test_texts() -> list[str]:
    """The event test set frozen before the credit work (data/eval/frozen_event_test.json)."""
    import hashlib
    import json

    frozen = set(
        json.loads((get_settings().data_dir / "eval" / "frozen_event_test.json").read_text())["sha1"]
    )
    tt = load_twitter_topic().query("split == 'test'")
    texts = sorted({t for t in tt["text"] if hashlib.sha1(t.encode()).hexdigest() in frozen})
    assert {hashlib.sha1(t.encode()).hexdigest() for t in texts} == frozen, "frozen event test set changed"
    return texts


def drop_near_duplicates_of(df: pd.DataFrame, protected: list[str], threshold: float = 0.8) -> pd.DataFrame:
    """Remove rows whose text near-duplicates any protected (test) text: syndicated copies
    of a test headline must never sit in training or validation. Only the protected texts
    are indexed, so ordinary duplicates *within* training are untouched."""
    from datasketch import MinHash, MinHashLSH

    from engine.nlp.dedupe import shingles

    def mh(text: str) -> MinHash:
        m = MinHash(num_perm=128)
        for sh in shingles(text):
            m.update(sh.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, t in enumerate(protected):
        lsh.insert(f"t{i}", mh(t))
    keep = [not lsh.query(mh(str(t))) for t in df["text"]]
    return df[keep]
