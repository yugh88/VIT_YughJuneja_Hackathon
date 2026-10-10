"""Translate-train data: machine-translate English training sets with NLLB-200.

    python -m training.translate phrasebank          # 4,846 sentences x 8 languages
    python -m training.translate events --per-lang 1200   # topic tweets + English weak labels

Runs once, offline, on GPU/MPS if present. Each split is translated separately, so
a test sentence never leaks into training in another language. Output goes to
``data/raw/translated/<name>.parquet`` with a ``lang`` column.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from training.data import load_phrasebank, load_twitter_topic, load_weak_events, raw_dir

log = logging.getLogger("translate")
MODEL = "facebook/nllb-200-distilled-600M"
# Plan section 5: hi, zh, ja, de, fr, es, ar, pt
NLLB_CODES = {
    "hi": "hin_Deva", "zh": "zho_Hans", "ja": "jpn_Jpan", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "ar": "arb_Arab", "pt": "por_Latn",
}  # fmt: skip


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Translator:
    def __init__(self, model: str = MODEL) -> None:
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(model, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.dev).eval()
        if self.dev != "cpu":
            self.model = self.model.half()

    @torch.inference_mode()
    def __call__(self, texts: list[str], lang: str, batch_size: int = 48) -> list[str]:
        out: list[str] = []
        tgt = self.tok.convert_tokens_to_ids(NLLB_CODES[lang])
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketing
        res: dict[int, str] = {}
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = self.tok(
                [texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            gen = self.model.generate(**enc, forced_bos_token_id=tgt, max_new_tokens=160, num_beams=1)
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                res[i] = t
        out = [res[i] for i in range(len(texts))]
        return out


def translate_frame(df: pd.DataFrame, tr: Translator, langs: list[str]) -> pd.DataFrame:
    parts = []
    for lang in langs:
        t0 = time.time()
        texts = tr(df["text"].tolist(), lang)
        part = df.copy()
        part["text_en"] = part["text"]
        part["text"] = texts
        part["lang"] = lang
        parts.append(part)
        log.info("%s: %d rows in %.0fs", lang, len(df), time.time() - t0)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["phrasebank", "events", "credit", "twitter"])
    ap.add_argument("--topic-per-lang", type=int, default=4000, help="twitter: topic train rows per language")
    ap.add_argument("--langs", default=",".join(NLLB_CODES))
    ap.add_argument("--per-lang", type=int, default=1200, help="topic rows per language (train split)")
    ap.add_argument("--weak-per-class", type=int, default=120, help="weak-label rows per class per language")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    langs = args.langs.split(",")
    out_dir = raw_dir() / "translated"
    out_dir.mkdir(parents=True, exist_ok=True)
    from engine.diskguard import require_space

    require_space("NLLB download", 2.5)
    tr = Translator()
    log.info("NLLB on %s", tr.dev)
    if args.what == "phrasebank":
        df = load_phrasebank()
        res = translate_frame(df, tr, langs)
    elif args.what == "credit":
        # Back-translate English credit positives + hard negatives (train split only) into 8 languages.
        from training.data import load_credit_events

        ce = load_credit_events()
        ce = ce[(ce["split"] == "train") & (ce["lang"] == "en") & (ce["source"] == "gdelt_credit")]
        topic = load_twitter_topic()
        tc = topic[(topic["split"] == "train") & (topic["event"] == "Credit Event")]
        pos = pd.concat([ce[ce["event"] == "Credit Event"][["text", "event"]], tc[["text", "event"]]])
        neg = ce[ce["event"] != "Credit Event"][["text", "event"]]
        parts = []
        for i, lang in enumerate(langs):  # different rows per language for diversity, same cost
            p = pos.sample(min(len(pos), 400), random_state=i)
            n = neg.sample(min(len(neg), 150), random_state=i)
            both = pd.concat(
                [p.assign(source="credit_mt"), n.assign(source="credit_hardneg_mt")], ignore_index=True
            )
            parts.append(translate_frame(both, tr, [lang]))
        res = pd.concat(parts, ignore_index=True)
        res["split"] = "train"
    elif args.what == "twitter":
        # DECISIONS D73(d): all twitter-financial-news sentiment train rows + topic train rows, into 8 languages.
        from training.data import load_twitter_sentiment

        sent = load_twitter_sentiment()
        sent = sent[sent["split"] == "train"][["text", "label"]].assign(source="twitter_sentiment_mt")
        topic = load_twitter_topic()
        topic = topic[topic["split"] == "train"]
        res_s = translate_frame(sent, tr, langs)
        res_s["split"] = "train"
        res_s.to_parquet(out_dir / "twitter_sentiment.parquet")
        log.info("wrote twitter_sentiment.parquet (%d rows)", len(res_s))
        parts = []
        for i, lang in enumerate(langs):  # seeds differ from the earlier `events` sample (0..7)
            t = topic.sample(min(len(topic), args.topic_per_lang), random_state=100 + i)
            parts.append(translate_frame(t[["text", "event"]].assign(source="twitter_topic_mt2"), tr, [lang]))
        res = pd.concat(parts, ignore_index=True)
        res["split"] = "train"
        args.what = "twitter_topic"
    else:
        topic = load_twitter_topic()
        topic = topic[topic["split"] != "test"]
        weak = load_weak_events()
        weak = weak[(weak["split"] == "train") & (weak["lang"] == "en")]
        parts = []
        for i, lang in enumerate(langs):  # different random rows per language
            t = topic.sample(min(len(topic), args.per_lang), random_state=i)
            w = pd.concat(
                [
                    g.sample(min(len(g), args.weak_per_class), random_state=i)
                    for _, g in weak.groupby("event")
                ],
                ignore_index=True,
            )
            both = pd.concat(
                [
                    t[["text", "event"]].assign(source="twitter_topic_mt"),
                    w[["text", "event"]].assign(source="weak_mt"),
                ],
                ignore_index=True,
            )
            parts.append(translate_frame(both, tr, [lang]))
        res = pd.concat(parts, ignore_index=True)
        res["split"] = "train"
    path = out_dir / f"{args.what}.parquet"
    res.to_parquet(path)
    log.info("wrote %s (%d rows)", path, len(res))


if __name__ == "__main__":
    main()
