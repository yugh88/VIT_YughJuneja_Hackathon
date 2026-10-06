"""Fine-tune the sentiment models.

    python -m training.train_sentiment en      # FinBERT (ProsusAI/finbert), English
    python -m training.train_sentiment multi   # XLM-RoBERTa-base, English + translate-train (8 languages)

Training data (train splits only; val for model selection, test never touched):
Financial PhraseBank, twitter-financial-news-sentiment, SEntFiN (entity-level),
FiQA-2018 (entity-level, score bucketed at +/-0.1), and for ``multi`` the NLLB
translations of PhraseBank's train split.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from training.common import TrainConfig, train_classifier
from training.data import (
    SENT_LABELS,
    load_fiqa,
    load_hindi_idioms,
    load_phrasebank,
    load_sentfin,
    load_translated_sentiment,
    load_twitter_sentiment,
)

BASES = {"en": "ProsusAI/finbert", "multi": "FacebookAI/xlm-roberta-base"}


def build_frames(which: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_phrasebank(), load_twitter_sentiment(), load_sentfin(), load_fiqa()]
    if which == "multi":
        tr = load_translated_sentiment()
        if tr.empty:
            raise SystemExit("run `python -m training.translate phrasebank` first")
        tr = tr.assign(entity="", source="phrasebank_mt")
        frames.append(tr)
        # Oversampled x4 so 31 sentences are not drowned out by 50k others.
        frames += [load_hindi_idioms()] * 4
    df = pd.concat([f[["text", "entity", "label", "split", "source"]] for f in frames], ignore_index=True)
    df["entity"] = df["entity"].fillna("").astype(str)
    df["y"] = df["label"].astype(int)
    df = df.dropna(subset=["text"])
    train = df[df["split"] == "train"].sample(frac=1.0, random_state=42).reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(BASES))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    train, val = build_frames(args.which)
    logging.info(
        "train %d rows by source %s; val %d", len(train), train["source"].value_counts().to_dict(), len(val)
    )
    cfg = TrainConfig(
        base_model=BASES[args.which],
        out_name=f"sentiment-{args.which}",
        labels=SENT_LABELS,
        epochs=args.epochs,
        lr=2e-5,
        batch_size=args.batch_size,
        max_len=128,
        freeze_embeddings=args.which == "multi",
    )
    train_classifier(cfg, train, val)


if __name__ == "__main__":
    main()
