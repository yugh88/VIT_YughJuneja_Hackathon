"""Multilingual evaluation set: build it for human labelling, then score against it.

    python -m training.multilingual_set build     # -> data/eval/multilingual_to_label.csv
    python -m training.multilingual_set label     # optional terminal helper to fill in labels

300 real GDELT headlines (30 x 10 languages) from translingual 15-minute windows the
training pipeline never sampled (all weak labels are English; no GDELT non-English
title is used in training). Each row carries an NLLB English translation **only as a
reading aid**: label from the original headline. Labels are left blank on purpose:
our own models must never label their own test set.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from engine.config import get_settings
from engine.ingest.gdelt import GdeltGkgSource
from engine.ingest.manager import make_http_client
from engine.nlp.entities import EntityLinker
from engine.schema import EventClass, RawDoc

log = logging.getLogger("multilingual_set")
LANGS = ["hi", "zh", "ja", "de", "fr", "es", "ar", "pt", "ru", "it"]
NLLB_SRC = {"hi": "hin_Deva", "zh": "zho_Hans", "ja": "jpn_Jpan", "de": "deu_Latn", "fr": "fra_Latn",
            "es": "spa_Latn", "ar": "arb_Arab", "pt": "por_Latn", "ru": "rus_Cyrl", "it": "ita_Latn"}  # fmt: skip
PER_LANG = 30
SENTIMENTS = ["negative", "neutral", "positive"]
EVENTS = [e.value for e in EventClass]


def path() -> Path:
    return get_settings().data_dir / "eval" / "multilingual_to_label.csv"


async def _sample(n_windows: int = 120, seed: int = 2026) -> list[RawDoc]:
    rng = random.Random(seed)
    start, end = datetime(2025, 6, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
    span = int((end - start).total_seconds() // 900)
    stamps = sorted(
        {
            (start + timedelta(minutes=15 * rng.randrange(span))).strftime("%Y%m%d%H%M%S")
            for _ in range(n_windows)
        }
    )
    src = GdeltGkgSource(translation=True, max_docs_per_file=3000)
    docs: list[RawDoc] = []
    async with make_http_client(get_settings()) as client:
        for i in range(0, len(stamps), 6):
            batch = await asyncio.gather(
                *(src.fetch_window(client, s) for s in stamps[i : i + 6]), return_exceptions=True
            )
            for b in batch:
                if isinstance(b, list):
                    docs += [d for d in b if d.language in LANGS]
    return docs


def _translate(rows: pd.DataFrame) -> list[str]:
    from training.translate import Translator

    tr = Translator()
    out = [""] * len(rows)
    for lang, g in rows.groupby("lang"):
        tr.tok.src_lang = NLLB_SRC[str(lang)]
        eng = tr.tok.convert_tokens_to_ids("eng_Latn")
        enc_texts = g["headline"].tolist()
        import torch

        with torch.inference_mode():
            for b in range(0, len(enc_texts), 16):
                chunk = enc_texts[b : b + 16]
                enc = tr.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
                enc = {k: v.to(tr.dev) for k, v in enc.items()}
                gen = tr.model.generate(**enc, forced_bos_token_id=eng, max_new_tokens=96, num_beams=2)
                for idx, t in zip(
                    g.index[b : b + 16], tr.tok.batch_decode(gen, skip_special_tokens=True), strict=True
                ):
                    out[rows.index.get_loc(idx)] = t
    return out


def build() -> Path:
    docs = asyncio.run(_sample())
    linker = EntityLinker()
    rng = random.Random(7)
    rows = []
    for lang in LANGS:
        pool = {d.title: d for d in docs if d.language == lang and 12 <= len(d.title) <= 220}
        linked = [d for d in pool.values() if linker.link(d.title).mentions]
        other = [d for d in pool.values() if d not in linked]
        rng.shuffle(linked)
        rng.shuffle(other)
        # Prefer headlines that name a tracked entity (what the engine actually scores), topped up with others.
        pick = (linked[: PER_LANG * 2 // 3] + other)[:PER_LANG]
        if len(pick) < PER_LANG:
            log.warning("%s: only %d headlines available", lang, len(pick))
        for d in pick:
            rows.append(
                {
                    "id": f"{lang}-{len(rows):04d}",
                    "lang": lang,
                    "headline": d.title,
                    "linked_entities": ";".join(e.id for e in linker.link(d.title).entities()),
                    "source_url": d.url,
                    "published": d.published_at.strftime("%Y-%m-%d"),
                }
            )
    df = pd.DataFrame(rows)
    df.insert(3, "english_translation_mt", _translate(df))
    df["sentiment_label"] = ""
    df["event_label"] = ""
    df["labeller_notes"] = ""
    out = path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("wrote %s: %d rows, per language %s", out, len(df), df["lang"].value_counts().to_dict())
    return out


def label() -> None:
    """Minimal terminal helper: shows each unlabelled headline, saves after every answer."""
    df = pd.read_csv(path(), dtype=str).fillna("")
    todo = df.index[df["sentiment_label"] == ""].tolist()
    print(f"{len(todo)} rows to label. Sentiment: n/u/p (negative/neutral/positive). Event: number. q quits.")
    for i, e in enumerate(EVENTS):
        print(f"  {i:2d} {e}")
    for idx in todo:
        r = df.loc[idx]
        print(f"\n[{r['id']}] {r['headline']}\n      (MT: {r['english_translation_mt']})")
        s = input("sentiment n/u/p > ").strip().lower()
        if s == "q":
            break
        ev = input("event # > ").strip()
        if ev == "q":
            break
        df.loc[idx, "sentiment_label"] = {"n": "negative", "u": "neutral", "p": "positive"}.get(s, "")
        df.loc[idx, "event_label"] = EVENTS[int(ev)] if ev.isdigit() and int(ev) < len(EVENTS) else ""
        df.to_csv(path(), index=False)


def labelled() -> pd.DataFrame | None:
    """Rows with both labels filled in (data/eval/multilingual_labelled.csv when the AI-annotated,
    blind two-pass labels exist; else the to-label file), or None if nothing is labelled yet."""
    annotated = path().with_name("multilingual_labelled.csv")
    p = annotated if annotated.exists() else path()
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype=str).fillna("")
    df = df[df["sentiment_label"].isin(SENTIMENTS) & df["event_label"].isin(EVENTS)]
    return df if len(df) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "label"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build() if args.what == "build" else label()


if __name__ == "__main__":
    main()
