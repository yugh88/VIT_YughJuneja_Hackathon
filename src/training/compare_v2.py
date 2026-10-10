"""Translate-train retrains (DECISIONS D73(d)): current vs v2 model on the dev sets only.

    python -m training.compare_v2 sentiment   # models/onnx/sentiment-multi-v2 vs sentiment-multi
    python -m training.compare_v2 event       # models/onnx/event-multi-v2 vs event-multi

Keep rule (pre-registered): the v2 model is kept only if its target dev metric improves and no other
dev metric drops by more than 0.01. Sentiment: target = native_v2_dev sentiment macro-F1 (shipped
engine path, NLI blend on); also checked: the Hinglish check set. Event: target = native_v2_dev event
macro-F1; also checked: credit_v2_dev precision, recall and F1. The event model reuses the shipped
credit rule and specialist head (both work on text, not on the classifier's internals).
Writes docs/eval/v2/compare_<which>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from engine.config import REPO_ROOT
from engine.nlp.events import CLASSES
from engine.nlp.onnx_model import OnnxClassifier
from training.eval_v2 import SENT, Native, credit_metrics, load, native_metrics

log = logging.getLogger("compare_v2")
ONNX = REPO_ROOT / "models" / "onnx"
OUT = REPO_ROOT / "docs" / "eval" / "v2"
TOL = 0.01


def _native(nat: Native) -> dict[str, Any]:
    d = load("native", "dev")
    ps, pe, _, _ = nat.probs(d["headline"].tolist())
    m = native_metrics(d, [SENT[i] for i in ps.argmax(1)], [CLASSES[i] for i in pe.argmax(1)])
    return {
        k: m[k]
        for k in ("sentiment_macro_f1", "sentiment_macro_f1_ci95", "event_macro_f1", "event_macro_f1_ci95")
    }


def _hinglish(nat: Native) -> float:
    from sklearn.metrics import f1_score

    h = pd.read_csv(REPO_ROOT / "data" / "eval" / "v2" / "hinglish_check.csv")
    ps, _, _, _ = nat.probs(h["text"].tolist())
    return round(
        float(f1_score(h["sentiment_label"], [SENT[i] for i in ps.argmax(1)], average="macro", labels=SENT)),
        4,
    )


def _credit(nat: Native) -> dict[str, Any]:
    c = load("credit", "dev")
    rs = nat.pipe.events.classify_batch(c["headline"].tolist(), allow_zero_shot=False)
    m = credit_metrics(c, [r.primary for r in rs], [r.confidence for r in rs])
    return {k: round(float(m[k]), 4) for k in ("precision", "recall", "f1")}


def score(which: str, model: str) -> dict[str, Any]:
    nat = Native()
    if which == "sentiment":
        nat.pipe.sentiment.multi = OnnxClassifier(ONNX / model)
        return _native(nat) | {"hinglish_f1": _hinglish(nat)}
    ev = nat.pipe.events
    ev.clf = OnnxClassifier(ONNX / model)
    return _native(nat) | {f"credit_{k}": v for k, v in _credit(nat).items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["sentiment", "event"])
    a = ap.parse_args()
    base = "sentiment-multi" if a.which == "sentiment" else "event-multi"
    target = "sentiment_macro_f1" if a.which == "sentiment" else "event_macro_f1"
    cur, new = score(a.which, base), score(a.which, f"{base}-v2")
    others = [k for k in cur if k != target and not k.endswith("_ci95")]
    drops = {k: round(cur[k] - new[k], 4) for k in others if cur[k] - new[k] > TOL}
    gain = round(new[target] - cur[target], 4)
    res = {"spec": "DECISIONS D73(d)", "split": "dev", "target": target, "current": cur, "v2": new,
           "gain": gain, "drops_over_tol": drops, "keep": bool(gain > 0 and not drops)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"compare_{a.which}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
