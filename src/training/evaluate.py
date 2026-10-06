"""`make eval`: measure every shipped model on held-out data and write the model card.

    python -m training.evaluate

Scores the exported ONNX artefacts (exactly what the engine runs), never the
PyTorch checkpoints. Results go to ``docs/eval/eval.json``; ``docs/model_card.md``
is rendered from it plus ``docs/eval/impact.json`` (``training.train_impact``) and
``docs/eval/bench.json`` (``make bench``), each stamped with time and source version.

Ordering is deliberate: uncontaminated results first. FinBERT was itself trained on
Financial PhraseBank, so PhraseBank numbers are reported last, with that caveat.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from engine.config import REPO_ROOT
from engine.nlp.onnx_model import OnnxClassifier, softmax
from engine.nlp.sentiment import LexiconSentiment, intensity, raw_of
from training.data import (
    EVENT_LABELS,
    SENT_LABELS,
    load_fiqa,
    load_hindi_idioms,
    load_phrasebank,
    load_sentfin,
    load_translated_sentiment,
    load_twitter_sentiment,
    load_twitter_topic,
    load_weak_events,
)
from training.model_card import provenance, render_model_card

log = logging.getLogger("evaluate")
ONNX = REPO_ROOT / "models" / "onnx"
EVAL_JSON = REPO_ROOT / "docs" / "eval" / "eval.json"


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of max-probability confidence (10 equal-width bins)."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return round(float(total), 4)


def cls_metrics(y: np.ndarray, pred: np.ndarray, conf: np.ndarray | None = None) -> dict[str, Any]:
    out = {
        "n": len(y),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
    }
    if conf is not None:
        out["ece"] = ece(conf, (y == pred).astype(float))
    return out


class SentimentEval:
    def __init__(self, name: str) -> None:
        self.m = OnnxClassifier(ONNX / name)

    def run(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.m.logits(df["entity"].fillna("").astype(str).tolist(), df["text"].tolist())
        p = softmax(logits / self.m.temperature)
        return p.argmax(1), p.max(1), raw_of(logits)

    def fiqa(self, df: pd.DataFrame) -> dict[str, Any]:
        _, _, raw = self.run(df)
        inten = np.array([intensity(t) for t in df["text"]])
        y = df["score"].to_numpy()
        return {
            "n": len(df),
            **{
                f"mse_{w}": round(float(np.mean((self.m.calibrate(raw, inten, w) - y) ** 2)), 4)
                for w in ("range", "human")
            },
        }


def lexicon_baselines(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    import pysentiment2 as ps
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    lm, vader, ours = ps.LM(), SentimentIntensityAnalyzer(), LexiconSentiment()

    def lab(s: float) -> int:
        return 2 if s > 0.05 else 0 if s < -0.05 else 1

    lm_s = np.array([lm.get_score(lm.tokenize(t))["Polarity"] for t in df["text"]])
    va_s = np.array([vader.polarity_scores(t)["compound"] for t in df["text"]])
    y = df["label"].to_numpy()
    return {
        "loughran_mcdonald": cls_metrics(y, np.array([lab(s) for s in lm_s])),
        "vader": cls_metrics(y, np.array([lab(s) for s in va_s])),
        "lm_plus_vader": cls_metrics(y, np.array([lab(s) for s in (lm_s + va_s) / 2])),
        "engine_lexicon_fallback": cls_metrics(
            y, np.array([LABEL_IDX[ours.score(t).label] for t in df["text"]])
        ),
    }


LABEL_IDX = {lab: i for i, lab in enumerate(SENT_LABELS)}


def assert_no_leakage(eval_texts: set[str]) -> dict[str, int]:
    """Training-only material must never appear in an evaluation set."""
    idioms = set(load_hindi_idioms()["text"])
    weak = set(load_weak_events()["text"])
    hits = {"hindi_idioms_in_eval": len(idioms & eval_texts), "weak_labels_in_eval": len(weak & eval_texts)}
    if any(hits.values()):
        raise AssertionError(f"training-only text found in eval sets: {hits}")
    return hits


def evaluate() -> dict[str, Any]:
    en, multi = SentimentEval("sentiment-en"), SentimentEval("sentiment-multi")
    tw = load_twitter_sentiment().query("split == 'test'")
    sf = load_sentfin().query("split == 'test'")
    fq = load_fiqa()
    fq_test = fq[fq["split"] == "test"]
    pb = load_phrasebank().query("split == 'test'")
    mt = load_translated_sentiment().query("split == 'test'")
    res: dict[str, Any] = {"sentiment": {}, "event": {}}
    s = res["sentiment"]

    for name, ev in (("en", en), ("multi", multi)):
        out: dict[str, Any] = {}
        for label, df in (("twitter_test", tw), ("sentfin_test", sf)):
            pred, conf, _ = ev.run(df)
            out[label] = cls_metrics(df["label"].to_numpy(), pred, conf)
        out["fiqa_test"] = ev.fiqa(fq_test)
        out["calibration"] = {k: v for k, v in ev.m.cfg.get("calibration", {}).items() if k != "check_scores"}
        out["quantization"] = {
            k: ev.m.cfg.get(k) for k in ("quantization", "fidelity_vs_fp32", "fp16_storage_agreement")
        }
        s[name] = out

    # Multilingual: machine-translated PhraseBank test is a *proxy*; native labels replace it once provided.
    per_lang = {}
    for lang, g in mt.groupby("lang"):
        pred, conf, _ = multi.run(g.assign(entity=""))
        per_lang[str(lang)] = cls_metrics(g["label"].to_numpy(), pred, conf)
    pred, conf, _ = multi.run(pb)
    en_ref = cls_metrics(pb["label"].to_numpy(), pred, conf)
    s["multilingual_proxy"] = {
        "what": "PhraseBank test split machine-translated (NLLB-200) into 8 languages; a proxy, "
        "not native text",
        "english_reference_same_model": en_ref,
        "per_language": per_lang,
        "mean_macro_f1": round(float(np.mean([v["macro_f1"] for v in per_lang.values()])), 4),
    }
    from training.multilingual_set import labelled

    native = labelled()
    if native is None:
        s["multilingual_native"] = {"status": "pending human labels (data/eval/multilingual_to_label.csv)"}
    else:
        from engine.nlp.events import CLASSES
        from engine.nlp.pipeline import Pipeline

        pipe = Pipeline(zero_shot=False, dedupe=False)
        sent_pred, ev_pred = [], []
        for t in native["headline"]:
            tr = pipe.analyze(t)
            sres = next(iter(tr.sentiments.values()), None)
            if sres is None:
                sres = pipe.sentiment.score_batch([("", t, tr.lang.lang, tr.lang.confidence)])[0]
            sent_pred.append(sres.label)
            ev_pred.append(tr.event.primary if tr.event else pipe.events.classify_batch([t])[0].primary)
        native = native.assign(sent_pred=sent_pred, ev_pred=ev_pred)
        langs = {}
        for lang, g in native.groupby("lang"):
            langs[str(lang)] = {
                "n": len(g),
                "sentiment_macro_f1": round(
                    float(f1_score(g["sentiment_label"], g["sent_pred"], average="macro")), 4
                ),
                "event_macro_f1": round(
                    float(f1_score(g["event_label"], g["ev_pred"], average="macro", labels=CLASSES)), 4
                ),
            }
        mean_f1 = float(np.mean([v["sentiment_macro_f1"] for v in langs.values()]))
        import json as _json

        from engine.config import REPO_ROOT as _ROOT

        ann_p = _ROOT / "data" / "eval" / "annotation_stats.json"
        ann = _json.loads(ann_p.read_text()) if ann_p.exists() else {}

        def _boot(y: pd.Series, yhat: pd.Series, labels: list[str] | None = None) -> list[float]:
            rng = np.random.default_rng(0)
            ya, pa = y.to_numpy(), yhat.to_numpy()
            vals = []
            for _ in range(1000):
                i = rng.integers(0, len(ya), len(ya))
                vals.append(f1_score(ya[i], pa[i], average="macro", labels=labels))
            return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

        present = sorted(set(native["event_label"]))
        s["multilingual_native"] = {
            "status": "labelled",
            "annotation": "AI-annotated, blind, two-pass" if "annotation" in native.columns else "human",
            "kappa": (ann.get("kappa") or {}).get("multi"),
            "human_spot_check": (ann.get("human_spot_check") or {}).get("multi"),
            "sentiment_macro_f1": round(
                float(f1_score(native["sentiment_label"], native["sent_pred"], average="macro")), 4
            ),
            "sentiment_macro_f1_ci95": _boot(native["sentiment_label"], native["sent_pred"]),
            "event_macro_f1": round(
                float(f1_score(native["event_label"], native["ev_pred"], average="macro", labels=present)), 4
            ),
            "event_macro_f1_ci95": _boot(native["event_label"], native["ev_pred"], present),
            "event_accuracy": round(float((native["event_label"] == native["ev_pred"]).mean()), 4),
            "n": len(native),
            "per_language": langs,
            "mean_sentiment_macro_f1": round(mean_f1, 4),
            "gap_vs_english_twitter": round(s["en"]["twitter_test"]["macro_f1"] - mean_f1, 4),
        }

    pred, conf, _ = en.run(pb)
    aa = pb["all_agree"].to_numpy()
    s["phrasebank_contaminated"] = {
        "caveat": "ProsusAI/finbert, the base of sentiment-en, was itself fine-tuned on Financial PhraseBank; "
        "these scores are likely optimistic.",
        "full_test": cls_metrics(pb["label"].to_numpy(), pred, conf),
        "all_agree_test": cls_metrics(pb["label"].to_numpy()[aa], pred[aa], conf[aa]),
    }
    s["baselines"] = {
        "twitter_test": lexicon_baselines(tw),
        "sentfin_test": lexicon_baselines(sf),
        "phrasebank_test": lexicon_baselines(pb),
    }
    s["delta_vs_lm_plus_vader_macro_f1"] = {
        "twitter_test": round(
            s["en"]["twitter_test"]["macro_f1"] - s["baselines"]["twitter_test"]["lm_plus_vader"]["macro_f1"],
            4,
        ),
        "sentfin_test": round(
            s["en"]["sentfin_test"]["macro_f1"] - s["baselines"]["sentfin_test"]["lm_plus_vader"]["macro_f1"],
            4,
        ),
    }

    # Events: mapped topic test (9 of 12 classes present); weak-label classes have no clean test set.
    ev = OnnxClassifier(ONNX / "event-multi")
    tt = load_twitter_topic().query("split == 'test'")
    p = softmax(ev.logits([""] * len(tt), tt["text"].tolist()) / ev.temperature)
    y = tt["event"].map(EVENT_LABELS.index).to_numpy()
    pred = p.argmax(1)
    present = sorted(set(y))
    res["event"] = {
        "test": "twitter-financial-news-topic validation split, topics mapped to our classes",
        "classes_present": [EVENT_LABELS[i] for i in present],
        "classes_without_clean_test_data": [c for i, c in enumerate(EVENT_LABELS) if i not in present],
        "macro_f1_present_classes": round(float(f1_score(y, pred, average="macro", labels=present)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4), "n": len(tt),
        "ece": ece(p.max(1), (pred == y).astype(float)),
        "per_class_f1": {EVENT_LABELS[i]: round(float(v), 4) for i, v in
                         zip(present, f1_score(y, pred, average=None, labels=present), strict=True)},
        "confusion_matrix": {"labels": EVENT_LABELS, "matrix": confusion_matrix(y, pred, labels=range(12)).tolist()},
    }  # fmt: skip

    eval_texts = (
        set(tw["text"])
        | set(sf["text"])
        | set(fq_test["text"])
        | set(pb["text"])
        | set(mt["text"])
        | set(tt["text"])
    )
    if native is not None:
        eval_texts |= set(native["headline"])
    res["leakage_check"] = assert_no_leakage(eval_texts)
    res["provenance"] = provenance()
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    t0 = time.time()
    res = evaluate()
    EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVAL_JSON.write_text(json.dumps(res, indent=2, default=float))
    path = render_model_card()
    log.info("eval done in %.0fs -> %s, %s", time.time() - t0, EVAL_JSON, path)


if __name__ == "__main__":
    main()
