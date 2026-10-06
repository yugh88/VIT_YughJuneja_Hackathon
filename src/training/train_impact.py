"""Train and evaluate the impact model on realised abnormal returns.

    python -m training.train_impact

Strict time split: fit on news from 2019-2023, report on 2024 onward. Hyper-parameters
are picked on 2023 (fit on 2019-2022) before the final refit, so the test years are
touched exactly once. Writes ``models/impact/impact_hgb.joblib`` and
``docs/eval/impact.json`` (the model card reads the latter).

Reported, whatever they turn out to be:
* Spearman rho between predicted impact and realised z (row level and per entity-day);
* precision at score >= 8: share of those signals whose realised z is in the training
  set's top decile, vs. the same share among all test signals (the base rate);
* mean / median realised z per score bucket 1-10 (the monotonicity chart);
* the same for the hand-set heuristic, as a baseline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.config import REPO_ROOT
from engine.impact import heuristic
from engine.impact.features import CATEGORICAL, FEATURES
from training.data import raw_dir

log = logging.getLogger("train_impact")
OUT = REPO_ROOT / "models" / "impact" / "impact_hgb.joblib"
EVAL = REPO_ROOT / "docs" / "eval" / "impact.json"
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL]
GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl, "max_iter": it}
    for lr in (0.03, 0.06) for leaves in (15, 31) for msl in (50, 200) for it in (200, 400)
]  # fmt: skip


def model(
    params: dict[str, Any], loss: str = "squared_error", quantile: float | None = None
) -> HistGradientBoostingRegressor:
    kw: dict[str, Any] = {
        "loss": loss,
        "categorical_features": CAT_IDX,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
        **params,
    }
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def to_score(pred: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return 1 + np.searchsorted(cuts, pred, side="right")


def report(
    score: np.ndarray, pred: np.ndarray, z: np.ndarray, z90: float, df: pd.DataFrame
) -> dict[str, Any]:
    rho, p = spearmanr(pred, z)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(z), len(z))
        boots.append(spearmanr(pred[i], z[i])[0])
    # Rows from the same entity on the same day share a label: also report per entity-day.
    g = pd.DataFrame({"k": df["entity"] + "|" + df["d0"].astype(str), "p": pred, "z": z}).groupby("k").mean()
    rho_ed, p_ed = spearmanr(g["p"], g["z"])
    top = z >= z90
    hi = score >= 8
    buckets = []
    for b in range(1, 11):
        m = score == b
        buckets.append(
            {
                "score": b,
                "n": int(m.sum()),
                "mean_z": round(float(z[m].mean()), 3) if m.any() else None,
                "median_z": round(float(np.median(z[m])), 3) if m.any() else None,
                "share_top_decile": round(float(top[m].mean()), 3) if m.any() else None,
            }
        )
    return {
        "n": len(z), "spearman_rho": round(float(rho), 4), "spearman_p": float(p),
        "spearman_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "entity_days": len(g), "spearman_rho_entity_day": round(float(rho_ed), 4), "spearman_p_entity_day": float(p_ed),
        "base_rate_top_decile": round(float(top.mean()), 4),
        "n_score_ge_8": int(hi.sum()),
        "precision_at_8": round(float(top[hi].mean()), 4) if hi.any() else None,
        "lift_at_8": round(float(top[hi].mean() / top.mean()), 3) if hi.any() and top.mean() else None,
        "buckets": buckets,
    }  # fmt: skip


def heuristic_scores(d: pd.DataFrame) -> np.ndarray:
    """The hand-set heuristic, recomputed from the stored features. (The dataset's
    ``heuristic_impact`` column holds whatever scorer the pipeline had loaded at build time,
    which is the previous learned model once one exists.)"""

    def opt(v: float) -> float | None:
        return None if pd.isna(v) else float(v)

    out = []
    for r in d.itertuples():
        f = heuristic.ImpactFeatures(
            event=str(r.event), abs_sentiment=float(r.abs_sentiment), confidence=float(r.confidence),
            credibility=float(r.credibility), severity_hits=int(r.severity_hits),
            syndication_count=round(float(np.expm1(r.log_syndication))) + 1, goldstein=opt(r.abs_goldstein),
            num_sources=None if pd.isna(r.log_num_sources) else round(float(np.expm1(r.log_num_sources))),
        )  # fmt: skip
        out.append(heuristic.score(f).score)
    return np.asarray(out, dtype=float)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    df = pd.read_parquet(raw_dir() / "impact_dataset.parquet")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    n_all = len(df)
    df = df.dropna(subset=["z"]).reset_index(drop=True)
    df = df[np.isfinite(df["z"])]
    # Kind-adjusted target: sovereign labels (max of index and VIX moves) run higher than
    # single-stock ones (median z 1.13 vs 0.84), so log(1+z) is centred within each entity kind
    # using *training-period* medians. Scores then mean "large for this kind of entity".
    pre = df[df["published_at"] < TEST_START]
    kind_median = pre.groupby("kind")["z"].apply(lambda z: float(np.median(np.log1p(z)))).to_dict()
    global_median = float(np.median(np.log1p(pre["z"])))
    df["y"] = np.log1p(df["z"]) - df["kind"].map(kind_median).fillna(global_median) + global_median
    df["z_adj"] = np.expm1(df["y"])
    train = df[df["published_at"] < TEST_START]
    test = df[df["published_at"] >= TEST_START]
    fit, val = train[train["published_at"] < VAL_START], train[train["published_at"] >= VAL_START]
    log.info(
        "rows: %d total, %d labelled; train %d (fit %d, val %d), test %d",
        n_all,
        len(df),
        len(train),
        len(fit),
        len(val),
        len(test),
    )

    X = lambda d: d[FEATURES].to_numpy(dtype=float)  # noqa: E731
    best, best_rho = GRID[0], -1.0
    for params in GRID:
        m = model(params).fit(X(fit), fit["y"])
        rho = spearmanr(m.predict(X(val)), val["z_adj"])[0]
        log.info("val rho %.4f  %s", rho, params)
        if rho > best_rho:
            best, best_rho = params, rho
    log.info("chosen %s (val rho %.4f)", best, best_rho)

    point = model(best).fit(X(train), train["y"])
    train_pred = point.predict(X(train))
    cuts = np.quantile(train_pred, np.linspace(0.1, 0.9, 9))
    z90 = float(np.quantile(train["z_adj"], 0.9))
    # Range = what actually happened historically: for each predicted score, the interquartile
    # range of realised outcomes (on the same 1-10 decile scale) among training signals with that score.
    z_cuts = np.quantile(train["z_adj"], np.linspace(0.1, 0.9, 9))
    realised = to_score(train["z_adj"].to_numpy(), z_cuts)
    tscore = to_score(train_pred, cuts)
    ranges = {
        int(b): [int(np.percentile(realised[tscore == b], 25)), int(np.percentile(realised[tscore == b], 75))]
        for b in range(1, 11)
        if (tscore == b).any()
    }

    pred = point.predict(X(test))
    score = to_score(pred, cuts)
    result_model = report(score, pred, test["z_adj"].to_numpy(), z90, test)
    result_model["spearman_rho_raw_z"] = round(float(spearmanr(pred, test["z"])[0]), 4)
    hz = heuristic_scores(test)
    result_heur = report(hz.astype(int), hz, test["z_adj"].to_numpy(), z90, test)
    result_heur["spearman_rho_raw_z"] = round(float(spearmanr(hz, test["z"])[0]), 4)

    # How much of the result is the earnings calendar rather than the news itself.
    no_earn = [j for j, f in enumerate(FEATURES) if f != "earnings_window"]
    abl = model(best).set_params(categorical_features=[no_earn.index(j) for j in CAT_IDX])
    abl.fit(X(train)[:, no_earn], train["y"])
    outside = (test["earnings_window"] != 1.0).to_numpy()
    ablation = {
        "rho_without_earnings_feature": round(
            float(spearmanr(abl.predict(X(test)[:, no_earn]), test["z_adj"])[0]), 4
        ),
        "rho_on_rows_outside_earnings_windows": round(
            float(spearmanr(pred[outside], test["z_adj"][outside])[0]), 4
        ),
        "share_of_test_rows_in_earnings_window": round(float((~outside).mean()), 4),
    }
    log.info("ablation: %s", ablation)

    # Permutation importance on the test years (drop in rho when a feature is shuffled).
    rng = np.random.default_rng(0)
    Xt = X(test)
    imp = {}
    for j, f in enumerate(FEATURES):
        Xp = Xt.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[f] = round(
            float(result_model["spearman_rho"] - spearmanr(point.predict(Xp), test["z_adj"])[0]), 4
        )
    medians = [
        float(np.nanmedian(X(train)[:, j])) if f not in CATEGORICAL else float(train[f].mode().iloc[0])
        for j, f in enumerate(FEATURES)
    ]

    version = f"hgb-impact-{time.strftime('%Y%m%d')}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": point,
            "ranges": ranges,
            "pred_deciles": cuts.tolist(),
            "medians": medians,
            "features": FEATURES,
            "z_top_decile": z90,
            "version": version,
            "params": best,
        },
        OUT,
        compress=3,
    )
    out = {
        "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": "z = |sum of market-model abnormal returns over d0, d0+1| / sigma60 (residual); sovereigns: "
                 "max(|index move|, |VIX change|) / sigma60; target and metrics use z adjusted within entity kind "
                 "(log1p z centred on each kind's 2019-2023 median)",
        "kind_median_log1p_z_train": {str(k): round(v, 4) for k, v in kind_median.items()},
        "split": {"train": "2019-01-01..2023-12-31", "test": "2024-01-01..", "hyperparameters_tuned_on": "2023"},
        "rows": {"all": n_all, "labelled": len(df), "train": len(train), "test": len(test),
                 "test_entities": int(test["entity"].nunique())},
        "z_top_decile_threshold_train": round(z90, 3),
        "params": best, "val_rho_tuning": round(float(best_rho), 4), "score_ranges_iqr_realised": ranges,
        "model": result_model, "heuristic_baseline": result_heur, "earnings_ablation": ablation,
        "permutation_importance_rho_drop": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
    }  # fmt: skip
    EVAL.parent.mkdir(parents=True, exist_ok=True)
    EVAL.write_text(json.dumps(out, indent=2, default=float))
    r = result_model
    log.info(
        "TEST rho %.4f (p=%.2g, CI %s), entity-day rho %.4f; precision@8 %s vs base %.3f (lift %s); heuristic rho %.4f",
        r["spearman_rho"],
        r["spearman_p"],
        r["spearman_ci95"],
        r["spearman_rho_entity_day"],
        r["precision_at_8"],
        r["base_rate_top_decile"],
        r["lift_at_8"],
        result_heur["spearman_rho"],
    )
    log.info("buckets mean z: %s", [b["mean_z"] for b in r["buckets"]])


if __name__ == "__main__":
    main()
