"""Impact experiment (DECISIONS D76): one pre-registered attempt, then stop.

    python -m training.impact_v2 embed    # mean-pooled FinBERT embeddings of every title (cached)
    python -m training.impact_v2 run      # dev (2023) rho with and without the new features; test once

New features: a 16-dim PCA of the mean-pooled embedding of the fine-tuned FinBERT (fitted on the fit
years only for the dev comparison, on all training years for the test run) and source-tier x
event-class interaction codes. Same time split and hyper-parameter grid as ``training.train_impact``.
Kept only if dev rho improves by >= 0.02.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from engine.config import REPO_ROOT
from engine.impact.features import FEATURES
from training.data import raw_dir
from training.train_impact import CAT_IDX, GRID, TEST_START, VAL_START, model

log = logging.getLogger("impact_v2")
EMB = raw_dir() / "impact_title_embeddings.npy"
OUT = REPO_ROOT / "docs" / "eval" / "v2" / "impact_v2_experiment.json"
N_PCA = 16
MIN_GAIN = 0.02


def embed() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from training.translate import device

    d = pd.read_parquet(raw_dir() / "impact_dataset.parquet")
    path = REPO_ROOT / "models" / "sentiment-en"
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModel.from_pretrained(path).to(device()).eval()
    titles = d["title"].fillna("").astype(str).tolist()
    out = np.zeros((len(titles), m.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in titles])
    with torch.inference_mode():
        for b in range(0, len(order), 256):
            idx = order[b : b + 256]
            enc = tok(
                [titles[i] for i in idx], padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            enc = {k: v.to(m.device) for k, v in enc.items()}
            h = m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
            if b % (256 * 40) == 0:
                log.info("embedded %d/%d", b, len(order))
    np.save(EMB, out)
    log.info("wrote %s %s", EMB, out.shape)


def _tier(cred: pd.Series) -> pd.Series:
    return pd.cut(cred.fillna(0.5), [-1, 0.6, 0.8, 2], labels=[0, 1, 2]).astype(int)


def run() -> dict[str, Any]:
    df = pd.read_parquet(raw_dir() / "impact_dataset.parquet")
    emb = np.load(EMB)
    df["row"] = np.arange(len(df))
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.dropna(subset=["z"])
    df = df[np.isfinite(df["z"])].reset_index(drop=True)
    pre = df[df["published_at"] < TEST_START]
    kind_median = pre.groupby("kind")["z"].apply(lambda z: float(np.median(np.log1p(z)))).to_dict()
    gm = float(np.median(np.log1p(pre["z"])))
    df["y"] = np.log1p(df["z"]) - df["kind"].map(kind_median).fillna(gm) + gm
    df["z_adj"] = np.expm1(df["y"])
    df["tier_event"] = _tier(df["credibility"]) * 100 + df["event"].astype(int)
    train, test = df[df["published_at"] < TEST_START], df[df["published_at"] >= TEST_START]
    fit, val = train[train["published_at"] < VAL_START], train[train["published_at"] >= VAL_START]

    def mats(pca: PCA | None, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        base = d[FEATURES].to_numpy(dtype=float)
        if pca is None:
            return base, base
        extra = np.column_stack(
            [pca.transform(emb[d["row"].to_numpy()]), d["tier_event"].to_numpy(dtype=float)]
        )
        return base, np.column_stack([base, extra])

    def mk(params: dict[str, Any], new: bool) -> Any:
        m = model(params)
        if new:  # the tier x event code is categorical; the PCA columns are numeric
            m.set_params(categorical_features=[*CAT_IDX, len(FEATURES) + N_PCA])
        return m

    def tuned(Xf: np.ndarray, Xv: np.ndarray, new: bool) -> tuple[dict[str, Any], float]:
        best, best_rho = GRID[0], -1.0
        for params in GRID:
            rho = spearmanr(mk(params, new).fit(Xf, fit["y"]).predict(Xv), val["z_adj"])[0]
            if rho > best_rho:
                best, best_rho = params, float(rho)
        return best, best_rho

    pca_fit = PCA(N_PCA, random_state=0).fit(emb[fit["row"].to_numpy()])
    bf, nf = mats(pca_fit, fit)
    bv, nv = mats(pca_fit, val)
    p_base, rho_base = tuned(bf, bv, False)
    p_new, rho_new = tuned(nf, nv, True)
    gain = rho_new - rho_base
    res: dict[str, Any] = {
        "spec": "DECISIONS D76", "dev": "2023 (fit 2019-2022)",
        "dev_rho_baseline": round(rho_base, 4), "dev_rho_new": round(rho_new, 4), "dev_gain": round(gain, 4),
        "keep": bool(gain >= MIN_GAIN), "params_baseline": p_base, "params_new": p_new,
        "pca_explained_variance": round(float(pca_fit.explained_variance_ratio_.sum()), 4),
    }  # fmt: skip
    # The 2024+ test is reported once either way (refit on all training years, PCA on training years).
    pca_tr = PCA(N_PCA, random_state=0).fit(emb[train["row"].to_numpy()])
    btr, ntr = mats(pca_tr, train)
    bte, nte = mats(pca_tr, test)
    res["test_rho_baseline"] = round(
        float(spearmanr(mk(p_base, False).fit(btr, train["y"]).predict(bte), test["z_adj"])[0]), 4
    )
    res["test_rho_new"] = round(
        float(spearmanr(mk(p_new, True).fit(ntr, train["y"]).predict(nte), test["z_adj"])[0]), 4
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    log.info("%s", res)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["embed", "run"])
    a = ap.parse_args()
    embed() if a.what == "embed" else run()


if __name__ == "__main__":
    main()
