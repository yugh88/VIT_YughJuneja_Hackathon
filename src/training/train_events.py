"""Fine-tune the 12-class event classifier (multilingual, XLM-RoBERTa-base).

The plan named mDeBERTa-v3-base; XLM-R is used because DeBERTa-v3 loses most of its
output contrast under INT8 quantisation (docs/DECISIONS.md D30), while XLM-R keeps
CPU inference small and fast.

    python -m training.train_events

Training data (train splits only):
* twitter-financial-news-topic, 20 topics mapped to our 12 classes (``training.data``);
* weak labels from GDELT themes/CAMEO + strict headline rules, and SEC 8-K item codes
  (``training.weak_labels``) - the only source for Credit, Cyber, ESG and Supply-Chain;
* NLLB translations of both into 8 languages (``training.translate events``).
Classes are loss-weighted by 1/sqrt(frequency).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from training.common import TrainConfig, train_classifier
from training.data import (
    EVENT_LABELS,
    drop_near_duplicates_of,
    frozen_test_texts,
    load_credit_events,
    load_credit_translations,
    load_translated_events,
    load_twitter_topic,
    load_weak_events,
)

BASE = "FacebookAI/xlm-roberta-base"


def build_frames(credit_data: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic = load_twitter_topic()
    weak = load_weak_events()
    mt = load_translated_events()
    cols = ["text", "event", "split", "source"]
    parts = [topic[cols], weak[cols], mt.assign(split="train")[cols] if not mt.empty else None]
    if credit_data:  # Credit distant supervision + its back-translations (training.credit_labels)
        ce = load_credit_events()
        # EDGAR rows are templated ("<co> files 8-K: Item 1.03 ..."): cap them so the model learns
        # credit language from news, not the 8-K template (the template is handled by the item override).
        edgar = ce[ce["source"] == "edgar_8k"]
        edgar = pd.concat(
            [
                g.sample(min(len(g), 600 if sp == "train" else 150), random_state=0)
                for (sp, _ev), g in edgar.groupby(["split", "event"])
            ]
        )
        parts += [edgar[cols], ce[ce["source"] != "edgar_8k"][cols], load_credit_translations()[cols]]
    df = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    df = df.dropna(subset=["text"])
    df = df[df["split"] != "test"]
    n0 = len(df)
    df = drop_near_duplicates_of(df, frozen_test_texts())  # frozen test stays unseen, incl. syndicated copies
    logging.info("removed %d train/val rows that near-duplicate the frozen test set", n0 - len(df))
    df["entity"] = ""
    df["y"] = df["event"].map(EVENT_LABELS.index)
    train = df[df["split"] == "train"].sample(frac=1.0, random_state=42).reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="event-multi")
    ap.add_argument("--no-credit-data", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    train, val = build_frames(credit_data=not args.no_credit_data)
    logging.info("train %d by source %s", len(train), train["source"].value_counts().to_dict())
    logging.info("train by class %s", train["event"].value_counts().to_dict())
    cfg = TrainConfig(
        base_model=args.base,
        out_name=args.out,
        labels=EVENT_LABELS,
        epochs=args.epochs,
        lr=3e-5,
        batch_size=args.batch_size,
        max_len=96,
        class_weighted=True,
        freeze_embeddings=True,
    )
    train_classifier(cfg, train, val)


if __name__ == "__main__":
    main()
