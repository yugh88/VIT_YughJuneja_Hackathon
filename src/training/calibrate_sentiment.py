"""Choose the map from raw sentiment to the published [-1, 1] score.

    python -m training.calibrate_sentiment            # both exported sentiment models

raw = P(positive) - P(negative) from the *untempered* softmax (temperature only
calibrates the confidence). Two maps are fitted and both are stored:

* **human**: one linear scale, least squares onto FiQA's human-rated scores. Best
  fidelity to FiQA, but clearly strong headlines rarely exceed |0.6|.
* **range**: g(r, I) = sign(r) * min(1, |r|^p * (M + beta * I)), where I in [0, 1] is
  the text's intensity cue (``engine.nlp.sentiment.intensity``: a fixed intensifier
  lexicon plus >= 10% moves). It must pass the range check in
  ``data/eval/sentiment_range_check.csv`` (all 40 clearly strong headlines |score| >= 0.7,
  all 20 neutral <= 0.2); among passing maps the one with the lowest FiQA val MSE wins.
  If one also keeps FiQA val MSE within +0.01 of the human map, the most expressive
  such map wins instead.

Fitting uses FiQA **val** only: both sentiment models saw FiQA train during
fine-tuning. FiQA test is only reported. Everything lands in ``sentinel.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.config import get_settings
from engine.nlp.entities import get_linker
from engine.nlp.onnx_model import OnnxClassifier, softmax
from engine.nlp.sentiment import intensity
from training.common import MODELS_DIR
from training.data import load_fiqa

log = logging.getLogger("calibrate")
CHECK = "sentiment_range_check.csv"


def cal_map(raw: np.ndarray, inten: np.ndarray, scale: float, power: float, beta: float) -> np.ndarray:
    return np.sign(raw) * np.minimum(1.0, np.abs(raw) ** power * (scale + beta * inten))  # type: ignore[no-any-return]


def raw_scores(clf: OnnxClassifier, entities: list[str], texts: list[str]) -> np.ndarray:
    p = softmax(clf.logits(entities, texts))
    return p[:, 2] - p[:, 0]  # type: ignore[no-any-return]


def check_set() -> pd.DataFrame:
    df = pd.read_csv(get_settings().data_dir / "eval" / CHECK)
    linker = get_linker()
    df["entity"] = [next(iter(e.name for e in linker.link(t).entities()), "") for t in df["text"]]
    return df


def _mse(g: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((g - y) ** 2))


def _check_stats(g: np.ndarray, strong: np.ndarray, neutral: np.ndarray) -> dict[str, float]:
    return {
        "check_strong_min_abs": round(float(np.abs(g[strong]).min()), 3),
        "check_strong_median_abs": round(float(np.median(np.abs(g[strong]))), 3),
        "check_neutral_max_abs": round(float(np.abs(g[neutral]).max()), 3),
    }


def calibrate(model_dir: Path) -> dict[str, Any]:
    clf = OnnxClassifier(model_dir)
    fiqa = load_fiqa()
    val, test = fiqa[fiqa["split"] == "val"], fiqa[fiqa["split"] == "test"]
    vr = raw_scores(clf, val["entity"].fillna("").tolist(), val["text"].tolist())
    tr = raw_scores(clf, test["entity"].fillna("").tolist(), test["text"].tolist())
    vi = np.array([intensity(t) for t in val["text"]])
    ti = np.array([intensity(t) for t in test["text"]])
    vy, ty = val["score"].to_numpy(), test["score"].to_numpy()

    chk = check_set()
    rc = raw_scores(clf, chk["entity"].tolist(), chk["text"].tolist())
    ic = np.array([intensity(t) for t in chk["text"]])
    strong = chk["expect"].str.startswith("strong").to_numpy()
    neutral = (chk["expect"] == "neutral").to_numpy()
    wrong_sign = ((chk["expect"] == "strong_neg").to_numpy() & (rc >= 0)) | (
        (chk["expect"] == "strong_pos").to_numpy() & (rc <= 0)
    )

    lin = float(np.sum(vr * vy) / np.sum(vr * vr))
    human_val = _mse(np.clip(lin * vr, -1, 1), vy)
    grid = []
    for scale in np.round(np.arange(0.30, 1.001, 0.05), 2):
        for power in np.round(np.arange(1.0, 6.01, 0.5), 2):
            for beta in np.round(np.arange(0.0, 1.001, 0.05), 2):
                row: dict[str, Any] = {
                    "scale": float(scale),
                    "power": float(power),
                    "beta": float(beta),
                    "fiqa_val_mse": round(_mse(cal_map(vr, vi, scale, power, beta), vy), 4),
                    **_check_stats(cal_map(rc, ic, scale, power, beta), strong, neutral),
                }
                g = cal_map(rc, ic, scale, power, beta)
                ok = np.where(strong, np.abs(g) >= 0.7, np.abs(g) <= 0.2) & ~wrong_sign
                row["check_items_ok"] = int(ok.sum())
                row["check_passes"] = (
                    row["check_strong_min_abs"] >= 0.7 and row["check_neutral_max_abs"] <= 0.2
                )
                row["within_budget"] = row["fiqa_val_mse"] <= human_val + 0.01
                grid.append(row)
    both = [r for r in grid if r["check_passes"] and r["within_budget"]]
    check_ok = [r for r in grid if r["check_passes"]]
    if both:
        best = max(both, key=lambda r: (r["check_strong_median_abs"], -r["fiqa_val_mse"]))
    elif check_ok:  # the range requirement wins; the MSE cost is reported, not hidden
        best = min(check_ok, key=lambda r: r["fiqa_val_mse"])
    else:  # nothing passes every item: satisfy as many as possible, then fit FiQA best; failures are listed
        best = max(grid, key=lambda r: (r["check_items_ok"], -r["fiqa_val_mse"]))
    budget_best = max(
        (r for r in grid if r["within_budget"] and r["check_neutral_max_abs"] <= 0.2),
        key=lambda r: r["check_strong_min_abs"],
        default=None,
    )
    g_chk = cal_map(rc, ic, best["scale"], best["power"], best["beta"])
    g_test = cal_map(tr, ti, best["scale"], best["power"], best["beta"])
    keep = ("scale", "power", "beta", "check_passes", "within_budget", "fiqa_val_mse")
    result: dict[str, Any] = {
        "published": "range",
        "range": {
            "type": "intensity",
            **{k: best[k] for k in keep},
            "fiqa_test_mse": round(_mse(g_test, ty), 4),
            **_check_stats(g_chk, strong, neutral),
        },
        "human": {
            "type": "linear",
            "scale": round(lin, 4),
            "fiqa_val_mse": round(human_val, 4),
            "fiqa_test_mse": round(_mse(np.clip(lin * tr, -1, 1), ty), 4),
            **_check_stats(np.clip(lin * rc, -1, 1), strong, neutral),
        },
        "best_within_budget": budget_best,
        "check_sign_errors": int(wrong_sign.sum()),
        "check_items_ok": best["check_items_ok"],
        "check_failures": [
            {"expect": e, "text": t, "score": round(float(v), 3)}
            for e, t, v in zip(chk["expect"], chk["text"], g_chk, strict=True)
            if (abs(v) < 0.7 if e != "neutral" else abs(v) > 0.2)
        ],
        "check_scores": [
            {"expect": e, "text": t, "score": round(float(v), 3)}
            for e, t, v in zip(chk["expect"], chk["text"], g_chk, strict=True)
        ],
        "candidates": {"tried": len(grid), "pass_check": len(check_ok), "pass_check_and_budget": len(both)},
    }
    cfg_path = model_dir / "sentinel.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["calibration"] = result
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    r, h = result["range"], result["human"]
    log.info(
        "%s: range M=%.2f p=%.1f beta=%.2f check=%s (%d/60 ok) budget=%s FiQA test MSE %.4f | human FiQA test MSE %.4f | "
        "range strong min %.2f median %.2f neutral max %.2f | candidates %s",
        model_dir.name, r["scale"], r["power"], r["beta"], r["check_passes"], result["check_items_ok"], r["within_budget"],
        r["fiqa_test_mse"], h["fiqa_test_mse"], r["check_strong_min_abs"], r["check_strong_median_abs"],
        r["check_neutral_max_abs"], result["candidates"],
    )  # fmt: skip
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in ("sentiment-en", "sentiment-multi"):
        d = MODELS_DIR / "onnx" / name
        if (d / "model.int8.onnx").exists():
            calibrate(d)


if __name__ == "__main__":
    main()
