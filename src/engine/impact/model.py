"""Stage 7 - impact score 1-10, learned from realised abnormal returns.

Loads ``models/impact/impact_hgb.joblib`` (scikit-learn gradient boosting trained
by ``training.train_impact``) on log(1+z). Predictions map to 1-10 with the training
set's prediction deciles, so "8" means "in the top 30% of predicted reactions". The
range is empirical: the interquartile range of realised outcomes (same 1-10 scale)
among training signals that received that score.

Drivers are occlusion attributions: for each feature, how much the prediction
drops when that feature is reset to its training median. If the model or
scikit-learn is unavailable the documented heuristic is used and every signal
says ``impact.mode = "heuristic"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import REPO_ROOT
from engine.impact import heuristic
from engine.impact.features import CATEGORICAL, EVENTS, FEATURES, KINDS, SECTORS, SOURCE_TYPES

log = logging.getLogger(__name__)
MODEL_PATH = REPO_ROOT / "models" / "impact" / "impact_hgb.joblib"
VOCABS = {"event": EVENTS, "sector": SECTORS, "kind": KINDS, "source_type": SOURCE_TYPES}


@dataclass(frozen=True)
class ImpactOutput:
    score: int
    low: int
    high: int
    drivers: list[str]
    mode: str  # "model" | "heuristic"
    effects: tuple[dict[str, object], ...] = ()  # signed occlusion effects, strongest first
    cap_reason: str | None = None  # set by the pipeline's news-quality caps


def _fmt(name: str, value: float) -> str:
    if name in VOCABS:
        vocab = VOCABS[name]
        idx = int(value)
        return f"{name}={vocab[idx] if 0 <= idx < len(vocab) else 'other'}"
    if np.isnan(value):
        return f"{name}=n/a"
    return f"{name}={value:.2f}" if abs(value) < 100 else f"{name}={value:.0f}"


class ImpactScorer:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.bundle: dict[str, Any] | None = None
        if path.exists():
            try:
                import joblib

                self.bundle = joblib.load(path)
                log.info("impact model %s loaded", self.bundle.get("version"))
            except Exception as exc:  # sklearn missing or version mismatch
                log.warning("impact model unavailable (%s): heuristic mode", exc)
        else:
            log.warning("%s missing: heuristic impact", path)

    @property
    def mode(self) -> str:
        return "model" if self.bundle else "heuristic"

    @property
    def version(self) -> str:
        return str(self.bundle["version"]) if self.bundle else "heuristic-1.0"

    def _to_score(self, pred: np.ndarray) -> np.ndarray:
        assert self.bundle is not None
        return 1 + np.searchsorted(np.asarray(self.bundle["pred_deciles"]), pred, side="right")

    def score(
        self, rows: list[dict[str, float]], fallbacks: list[heuristic.ImpactFeatures]
    ) -> list[ImpactOutput]:
        if not rows:
            return []
        if self.bundle is None:
            out = []
            for f in fallbacks:
                h = heuristic.score(f)
                out.append(ImpactOutput(h.score, h.low, h.high, h.drivers, "heuristic"))
            return out
        b = self.bundle
        feats = list(b.get("features", FEATURES))  # the model's own feature order, not the current code's
        X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
        point = b["model"].predict(X)
        s = self._to_score(point)
        ranges = b.get("ranges", {})
        # Occlusion: reset one feature at a time to its training median, measure the drop.
        base = np.asarray(b["medians"], dtype=float)
        n, k = X.shape
        occluded = np.repeat(X, k, axis=0)
        for j in range(k):
            occluded[j::k, j] = base[j]
        deltas = point[:, None] - b["model"].predict(occluded).reshape(n, k)
        out = []
        for i in range(n):
            top = [j for j in np.argsort(-deltas[i]) if deltas[i, j] > 0.01 and not np.isnan(X[i, j])][:3]
            drivers = [_fmt(feats[j], X[i, j]) for j in top]
            score = int(s[i])
            lo, hi = ranges.get(score, (score, score))
            # Signed effects for plain-language reasons: up to 3 that raised the score, 2 that held it down.
            ok = [j for j in range(k) if not np.isnan(X[i, j])]
            ups = sorted((j for j in ok if deltas[i, j] > 0.002), key=lambda j: -deltas[i, j])[:3]
            downs = sorted((j for j in ok if deltas[i, j] < -0.002), key=lambda j: deltas[i, j])[:2]
            effects = tuple(
                {
                    "feature": feats[j],
                    "value": _fmt(feats[j], X[i, j]).split("=", 1)[1],
                    "effect": round(float(deltas[i, j]), 4),
                }
                for j in ups + downs
            )
            out.append(
                ImpactOutput(score, int(min(lo, score)), int(max(hi, score)), drivers, "model", effects)
            )
        return out


__all__ = ["CATEGORICAL", "ImpactOutput", "ImpactScorer"]
