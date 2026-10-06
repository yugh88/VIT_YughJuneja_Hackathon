"""Download every public dataset the project uses (keyless) and build derived files.

    python -m training.download_data all          # everything below
    python -m training.download_data tweets       # tweet replay twin (data/replay/tweets/)
    python -m training.download_data training     # sentiment/event/impact training sets (data/raw/)

Raw downloads go to ``data/raw/`` (git-ignored, re-creatable). Exact row counts are
written to ``data/raw/manifest.json``, which ``data/SOURCES.md`` is generated from
(``python -m training.download_data sources``), so the cited sizes are measured.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from engine.config import get_settings
from engine.schema import RawDoc, SourceType

log = logging.getLogger("download")
HF = "https://huggingface.co"


@dataclass
class DatasetSpec:
    name: str
    title: str
    url: str  # human-facing citation URL
    license: str
    citation: str
    files: dict[str, str] = field(default_factory=dict)  # local file -> download URL
    used_for: str = ""


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="financial_phrasebank",
        title="Financial PhraseBank v1.0",
        url="https://huggingface.co/datasets/takala/financial_phrasebank",
        license="CC BY-NC-SA 3.0",
        citation="Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). Good debt or bad debt: "
        "Detecting semantic orientations in economic texts. JASIST 65(4).",
        files={
            "FinancialPhraseBank-v1.0.zip": f"{HF}/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
        },
        used_for="Sentiment fine-tuning and benchmark (all-agree and full splits)",
    ),
    DatasetSpec(
        name="twitter_fin_sentiment",
        title="Twitter Financial News Sentiment",
        url="https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment",
        license="MIT",
        citation="zeroshot (2022), twitter-financial-news-sentiment, Hugging Face Datasets.",
        files={
            "train.parquet": f"{HF}/api/datasets/zeroshot/twitter-financial-news-sentiment/parquet/default/train/0.parquet",
            "validation.parquet": f"{HF}/api/datasets/zeroshot/twitter-financial-news-sentiment/parquet/default/validation/0.parquet",
        },
        used_for="Social-domain sentiment fine-tuning and test",
    ),
    DatasetSpec(
        name="twitter_fin_topic",
        title="Twitter Financial News Topic",
        url="https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic",
        license="MIT",
        citation="zeroshot (2022), twitter-financial-news-topic, Hugging Face Datasets.",
        files={
            "train.parquet": f"{HF}/api/datasets/zeroshot/twitter-financial-news-topic/parquet/default/train/0.parquet",
            "validation.parquet": f"{HF}/api/datasets/zeroshot/twitter-financial-news-topic/parquet/default/validation/0.parquet",
        },
        used_for="Event classifier (20 topics mapped to our 12 classes)",
    ),
    DatasetSpec(
        name="fiqa_sentiment",
        title="FiQA-2018 Task 1 (aspect-based financial sentiment)",
        url="https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification",
        license="CC BY 4.0 (FiQA 2018 challenge data)",
        citation="Maia, M. et al. (2018). WWW'18 Open Challenge: Financial Opinion Mining and Question Answering.",
        files={
            f"{s}.parquet": f"{HF}/api/datasets/TheFinAI/fiqa-sentiment-classification/parquet/default/{s}/0.parquet"
            for s in ("train", "valid", "test")
        },
        used_for="Calibrating the continuous [-1, 1] sentiment scale (isotonic) and MSE benchmark",
    ),
    DatasetSpec(
        name="sentfin",
        title="SEntFiN 1.0 (entity-level sentiment, Indian financial headlines)",
        url="https://huggingface.co/datasets/temetnosce01/phrasebank_and_sentfin",
        license="Research use (per the SEntFiN paper); HF mirror of the Kaggle release",
        citation="Sinha, A., Kedas, S., Kumar, R., Malo, P. (2022). SEntFiN 1.0: Entity-aware sentiment analysis "
        "for financial news. JASIST 73(9).",
        files={
            "train.parquet": f"{HF}/api/datasets/temetnosce01/phrasebank_and_sentfin/parquet/default/train/0.parquet"
        },
        used_for="Entity-aware sentiment fine-tuning (India coverage); PhraseBank rows in the mirror are ignored",
    ),
    DatasetSpec(
        name="stock_tweets",
        title="Stock Market Tweets Data (Apr-Jul 2020)",
        url="https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data",
        license="CC BY 4.0 (per dataset card)",
        citation="Akkerman, S. (2023), stock-market-tweets-data, Hugging Face Datasets.",
        files={
            "train.parquet": f"{HF}/api/datasets/StephanAkkerman/stock-market-tweets-data/parquet/default/train/0.parquet"
        },
        used_for="Social stream replay (stands in for X/Twitter, which has no keyless API)",
    ),
    DatasetSpec(
        name="company_tweets",
        title="Tweets about the top companies 2015-2019 (ticker-tagged)",
        url="https://huggingface.co/datasets/mjw/stock_market_tweets",
        license="CC0 / public (mirror of the Kaggle 'Tweets about the Top Companies from 2015 to 2020')",
        citation="Dogan, M., Metin, O., Tek, E., Yumusak, S., Oztoprak, K. (2020). Speculator and influencer "
        "evaluation in stock market by using social media. IEEE BigData.",
        files={"train.parquet": f"{HF}/api/datasets/mjw/stock_market_tweets/parquet/default/train/0.parquet"},
        used_for="Impact-label construction and Module A backtest (daily ticker sentiment)",
    ),
]

TRAINING_SETS = [
    "financial_phrasebank",
    "twitter_fin_sentiment",
    "twitter_fin_topic",
    "fiqa_sentiment",
    "sentfin",
]


def raw_dir() -> Path:
    return get_settings().data_dir / "raw"


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached %s", dest.relative_to(raw_dir().parent))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _count_rows(spec: DatasetSpec, d: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fname in spec.files:
        p = d / fname
        if p.suffix == ".parquet":
            counts[p.stem] = int(pd.read_parquet(p, columns=None).shape[0])
        elif p.suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.endswith(".txt") and "Sentences" in n:
                        counts[Path(n).stem] = sum(1 for line in zf.read(n).splitlines() if line.strip())
    return counts


def _update_manifest(name: str, info: dict[str, Any]) -> None:
    path = raw_dir() / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[name] = info
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def fetch(names: list[str]) -> None:
    specs = {s.name: s for s in DATASETS}
    with httpx.Client(headers={"User-Agent": get_settings().user_agent}) as client:
        for n in names:
            spec = specs[n]
            d = raw_dir() / spec.name
            for fname, url in spec.files.items():
                _download(client, url, d / fname)
            counts = _count_rows(spec, d)
            _update_manifest(spec.name, {"rows": counts, "downloaded_at": datetime.now(UTC).isoformat()})
            log.info("%s rows: %s", spec.name, counts)


# ----------------------------------------------------------------------------- tweet replay twin
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
REPLAY_DAY = "2020-07-13"  # busiest day in the dataset (19k tweets), a Monday


def build_tweet_replay(day: str = REPLAY_DAY, max_tweets: int = 6000) -> Path:
    """Turn one day of real historical tweets into the ``tweets`` replay twin.

    Keeps original tweets (no retweets) that carry at least one cashtag, so the
    stream is about markets; timestamps are the real ones, so replay pacing is real.
    """
    fetch(["stock_tweets"])
    df = pd.read_parquet(raw_dir() / "stock_tweets" / "train.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df[df["created_at"].dt.strftime("%Y-%m-%d") == day]
    df = df[~df["text"].str.startswith("RT @")]
    df = df[df["text"].str.contains(r"\$[A-Z]{1,5}\b", regex=True)]
    df = df.sort_values("created_at").head(max_tweets)
    out = get_settings().replay_dir / "tweets" / f"stock_tweets_{day}.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            text = re.sub(r"https?://\S+", "", str(row.text)).strip()
            doc = RawDoc(
                doc_id=RawDoc.make_id("tweets", str(row.id)),
                source="tweets",
                source_type=SourceType.SOCIAL,
                outlet="twitter.com",
                url="",
                title="",
                text=text,
                language=None,
                published_at=row.created_at.to_pydatetime(),
                meta={
                    "cashtags": sorted(set(_CASHTAG.findall(text)))[:10],
                    "dataset": "StephanAkkerman/stock-market-tweets-data",
                },
            )
            fh.write(doc.model_dump_json() + "\n")
    _update_manifest(
        "tweets_replay", {"rows": {day: len(df)}, "file": str(out.relative_to(get_settings().data_dir))}
    )
    log.info("tweet replay twin: %d tweets -> %s", len(df), out)
    return out


# ----------------------------------------------------------------------------- SOURCES.md
SOURCES_HEADER = """# Data sources

Every file under `data/` is either public (cited below with licence and URL) or
synthetic (generated by a seeded script in this repo). **No real client data is used.**
Row counts are measured by `python -m training.download_data sources`, not typed by hand.

"""


def write_sources_md() -> Path:
    manifest_path = raw_dir() / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    lines = [
        SOURCES_HEADER,
        "## Datasets\n",
        "| Dataset | Licence | Rows (measured) | Used for |",
        "| --- | --- | --- | --- |",
    ]
    for s in DATASETS:
        rows = manifest.get(s.name, {}).get("rows")
        rows_txt = ", ".join(f"{k}: {v:,}" for k, v in rows.items()) if rows else "not downloaded yet"
        lines.append(f"| [{s.title}]({s.url}) | {s.license} | {rows_txt} | {s.used_for} |")
    lines += ["", "### Citations", ""]
    lines += [f"- **{s.title}**: {s.citation}" for s in DATASETS]
    static = Path(__file__).with_name("sources_static.md")
    if static.exists():
        lines += ["", static.read_text().strip()]
    out = get_settings().data_dir / "SOURCES.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "tweets", "training", "impact", "sources"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.what in {"all", "tweets"}:
        build_tweet_replay()
    if args.what in {"all", "training"}:
        fetch(TRAINING_SETS)
    if args.what in {"all", "impact"}:
        fetch(["company_tweets"])
    print(f"wrote {write_sources_md()}")


if __name__ == "__main__":
    main()
