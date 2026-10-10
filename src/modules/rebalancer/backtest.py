"""Module A backtest with costs, benchmarks and a placebo test.

    python -m modules.rebalancer.backtest

Data: the historical signals the shipped pipeline produced for the impact study (GDELT, one
15-minute window per weekday, 2020-2026; ``data/raw/impact_dataset.parquet``) and daily
prices (yfinance). Sentiment is the published ``sentiment.score``.

Timing (no look-ahead): weights are decided at each US close (20:00 UTC) from signals
published strictly before it, and earn the next day's close-to-close return. S_i uses the
live aggregator formula (12 h half-life, neutral prior). Costs: 10 bp per side on turnover.

Benchmarks: the same 20 names cap-weighted (float shares x price, 10% cap) and equal-weighted,
both rebalanced daily with the same costs. Placebo: the day labels of the sentiment matrix are
shuffled 100 times; a real signal should beat >= 95% of shuffles on information ratio.

Two versions (DECISIONS D67, specified before running): v1 tilts on S_i with hysteresis 0.10;
v2 tilts on the sentiment *surprise* S_i - (entity's mean S_i over the previous 30 days), built
from impact >= 6 signals only, hysteresis 0.15. The version is chosen on validation (2020-2023)
by net IR; the 2024+ test period is run once for both, each with its own placebo.

Known approximations: today's float share counts are applied to past prices; the universe
is fixed (survivorship); sampled GDELT coverage makes S sparse between windows.
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from engine.config import REPO_ROOT
from modules.rebalancer.universe import BASE_CAP, UNIVERSE, capped, load_universe
from modules.rebalancer.weights import (
    HYSTERESIS,
    KAPPA,
    NAME_CAP,
    NAME_FLOOR,
    SECTOR_CAP,
    TILT_MAX,
    TILT_MIN,
    TURNOVER_CAP,
    limits_np,
)
from training.data import raw_dir

log = logging.getLogger("backtest")
OUT = REPO_ROOT / "docs" / "eval" / "backtest.json"
COST = 0.0010  # 10 bp per side
HALF_LIFE_H = 12.0
PRIOR = 1.0
START, END = "2020-01-01", "2026-08-31"
VAL = ("2020-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-08-31")
VERSIONS: dict[str, dict[str, Any]] = {
    "v1_sentiment": {"min_impact": None, "surprise": False, "hysteresis": HYSTERESIS},
    "v2_surprise": {"min_impact": 6, "surprise": True, "hysteresis": 0.15},
}


def impact_scores(df: pd.DataFrame) -> np.ndarray:
    """Historical impact scores from the shipped impact model (trained on 2019-2023; see D67)."""
    from engine.impact.model import ImpactScorer

    sc = ImpactScorer()
    assert sc.bundle is not None, "impact model missing"
    X = df[list(sc.bundle["features"])].to_numpy(dtype=float)
    return sc._to_score(sc.bundle["model"].predict(X))


def daily_scores(days: pd.DatetimeIndex, min_impact: int | None = None) -> pd.DataFrame:
    """S_i at each decision time (US close), from signals published before it."""
    sig = pd.read_parquet(raw_dir() / "impact_dataset.parquet")
    sig = sig[sig["entity"].isin(UNIVERSE)].copy()
    if min_impact is not None:
        sig = sig[impact_scores(sig) >= min_impact].copy()
    sig["t"] = pd.to_datetime(sig["published_at"], utc=True).dt.tz_localize(None)
    sig["w"] = sig["confidence"] * sig["credibility"] * sig["novelty"]
    lam = math.log(2) / (HALF_LIFE_H * 3600)
    decision = days + pd.Timedelta(hours=20)
    num = pd.Series(0.0, index=UNIVERSE)
    den = pd.Series(0.0, index=UNIVERSE)
    out = []
    prev = None
    sig = sig.sort_values("t")
    j = 0
    rows = sig[["t", "entity", "sentiment", "w"]].to_numpy()
    for dt in decision:
        if prev is not None:
            f = math.exp(-lam * (dt - prev).total_seconds())
            num, den = num * f, den * f
        while j < len(rows) and rows[j][0] < dt:
            t, e, s, w = rows[j]
            d = w * math.exp(-lam * (dt - t).total_seconds())
            num[e] += d * s
            den[e] += d
            j += 1
        out.append((num / (PRIOR + den)).to_numpy())
        prev = dt
    return pd.DataFrame(out, index=days, columns=UNIVERSE)


def surprise(scores: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """S minus the entity's own mean S over the previous ``days`` calendar days (excluding today)."""
    prior = scores.rolling(f"{days}D", closed="left").mean()
    return (scores - prior).fillna(0.0)


def run(
    scores: np.ndarray,
    rets: np.ndarray,
    base: np.ndarray,
    codes: np.ndarray,
    kappa: float = KAPPA,
    tilt_on: bool = True,
    hysteresis: float = HYSTERESIS,
) -> dict[str, np.ndarray]:
    """Daily loop. scores[d] decides weights held over rets[d+1]."""
    n_days, n = rets.shape
    w = base[0].copy()
    used = np.zeros(n)
    port, gross, turns = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
    limits_ok = np.ones(n_days, dtype=bool)  # post-trade weights within every risk limit
    weights = np.zeros((n_days, n))
    for d in range(n_days - 1):
        if tilt_on:
            s = scores[d]
            used = np.where(np.abs(s - used) > hysteresis, s, used)
            target = limits_np(base[d] * np.clip(np.exp(kappa * used), TILT_MIN, TILT_MAX), codes)
        else:
            target = base[d]
        diff = target - w
        to = np.abs(diff).sum()
        if tilt_on and to > TURNOVER_CAP:
            w = w + diff * (TURNOVER_CAP / to)
            w = w / w.sum()
            to = TURNOVER_CAP
        else:
            w = target.copy()
        turns[d] = to
        weights[d] = w
        if tilt_on:
            sec = np.bincount(codes, weights=w)
            limits_ok[d] = bool(w.max() <= NAME_CAP + 1e-9 and w.min() >= NAME_FLOOR - 1e-9
                                and sec.max() <= SECTOR_CAP + 1e-9 and to <= TURNOVER_CAP + 1e-9)  # fmt: skip
        r = rets[d + 1]
        gross[d + 1] = float(w @ r)
        port[d + 1] = gross[d + 1] - COST * to
        w = w * (1 + r)
        w = w / w.sum()  # drift until the next decision
    return {"ret": port, "gross": gross, "turnover": turns, "weights": weights, "limits_ok": limits_ok}


def stats(
    r: np.ndarray, bench: np.ndarray | None = None, turnover: np.ndarray | None = None
) -> dict[str, float]:
    ann = 252
    eq = np.cumprod(1 + r)
    out = {
        "annual_return": float(eq[-1] ** (ann / len(r)) - 1),
        "annual_vol": float(r.std() * math.sqrt(ann)),
        "sharpe": float(r.mean() / r.std() * math.sqrt(ann)) if r.std() else 0.0,
        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
    }
    if bench is not None:
        a = r - bench
        out["tracking_error"] = float(a.std() * math.sqrt(ann))
        out["information_ratio"] = float(a.mean() / a.std() * math.sqrt(ann)) if a.std() else 0.0
        out["active_return"] = float(a.mean() * ann)
    if turnover is not None:
        out["avg_turnover_per_day"] = float(turnover.mean())
        out["max_turnover"] = float(turnover.max())
    return {k: round(v, 4) for k, v in out.items()}


def _placebo(args: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]) -> float:
    seed, scores, rets, base, codes, bench, hyst = args
    perm = np.random.default_rng(seed).permutation(len(scores))
    r = run(scores[perm], rets, base, codes, hysteresis=hyst)["ret"]
    a = r - bench
    return float(a.mean() / a.std() * math.sqrt(252))


def period(
    S: np.ndarray,
    R: np.ndarray,
    base: np.ndarray,
    codes: np.ndarray,
    mask: np.ndarray,
    hyst: float,
    n_placebo: int,
) -> dict[str, Any]:
    """One version on one period: net and gross IR vs cap-weighted, turnover, optional placebo."""
    s, r, b = S[mask], R[mask], base[mask]
    cw = run(s, r, b, codes, tilt_on=False)
    t = run(s, r, b, codes, hysteresis=hyst)
    net = stats(t["ret"], cw["ret"], t["turnover"])
    gross = stats(t["gross"], cw["gross"])
    out = {"days": int(mask.sum()), "net": net, "gross_information_ratio": gross["information_ratio"],
           "gross_active_return": gross["active_return"], "net_information_ratio": net["information_ratio"],
           "avg_turnover_per_day": net["avg_turnover_per_day"], "cap_weighted": stats(cw["ret"])}  # fmt: skip
    if n_placebo:
        with ProcessPoolExecutor(max_workers=8) as ex:
            irs = list(ex.map(_placebo, [(k, s, r, b, codes, cw["ret"], hyst) for k in range(n_placebo)]))
        out["placebo"] = {"n": n_placebo, "real_ir": net["information_ratio"],
                          "placebo_ir_mean": round(float(np.mean(irs)), 4),
                          "placebo_ir_p95": round(float(np.percentile(irs, 95)), 4),
                          "share_of_placebos_beaten": round(float(np.mean(np.array(irs) < net["information_ratio"])), 4)}  # fmt: skip
    return out


def main(n_placebo: int = 100) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    t0 = time.time()
    u = load_universe()
    px = pd.read_parquet(raw_dir() / "prices.parquet")
    close = px[px["ticker"].isin(UNIVERSE)].pivot(index="date", columns="ticker", values="close")[UNIVERSE]
    close = close.loc[START:END].dropna()
    rets = close.pct_change().fillna(0.0)
    days = close.index
    mcap = close * u["float_shares"]
    base = np.vstack([capped(mcap.loc[d].copy(), BASE_CAP).to_numpy() for d in days])
    codes, _ = pd.factorize(u["sector"])
    R = rets.to_numpy()
    val_mask = np.asarray((days >= VAL[0]) & (days <= VAL[1]))
    test_mask = np.asarray((days >= TEST[0]) & (days <= TEST[1]))
    scores: dict[str, np.ndarray] = {}
    res: dict[str, Any] = {"window": [str(days[0].date()), str(days[-1].date())], "days": len(days),
                           "split": {"validation": list(VAL), "test": list(TEST)},
                           "params": {"kappa": KAPPA, "half_life_h": HALF_LIFE_H, "cost_bp_per_side": COST * 1e4,
                                      "turnover_cap": TURNOVER_CAP}, "versions": {}}  # fmt: skip
    for name, cfg in VERSIONS.items():
        S = daily_scores(days, cfg["min_impact"])
        if cfg["surprise"]:
            S = surprise(S)
        scores[name] = S.to_numpy()
        res["versions"][name] = {"spec": cfg,
                                 "coverage": {"days_with_any_signal_change": int((np.abs(np.diff(scores[name], axis=0)).sum(1) > 1e-9).sum()),
                                              "mean_abs_score": round(float(np.abs(scores[name]).mean()), 4)},
                                 "validation": period(scores[name], R, base, codes, val_mask, cfg["hysteresis"], 0)}  # fmt: skip
    chosen = max(VERSIONS, key=lambda n: res["versions"][n]["validation"]["net_information_ratio"])
    res["chosen_on_validation"] = chosen
    log.info(
        "validation: %s -> chosen %s",
        {n: res["versions"][n]["validation"]["net_information_ratio"] for n in VERSIONS},
        chosen,
    )
    # Test period: run once for both versions, each with its own placebo.
    for name, cfg in VERSIONS.items():
        res["versions"][name]["test"] = period(
            scores[name], R, base, codes, test_mask, cfg["hysteresis"], n_placebo
        )
        tst = res["versions"][name]["test"]
        res["versions"][name]["passes"] = bool(
            tst["net_information_ratio"] > 0 and tst["placebo"]["share_of_placebos_beaten"] >= 0.95
        )
    res["any_version_passes"] = any(v["passes"] for v in res["versions"].values())
    res["verdict"] = ("A sentiment tilt passed the pre-registered test (net IR > 0 and >= 95% of placebos)."
                      if res["any_version_passes"] else
                      "Neither version adds value: on the 2024+ test period neither has a positive net information "
                      "ratio that beats 95% of date-shuffled placebos. The tilt is shown for transparency, not as an edge.")  # fmt: skip
    # Full-period run of the chosen version, plus benchmarks, for the charts.
    hyst = VERSIONS[chosen]["hysteresis"]
    S = scores[chosen]
    tilt = run(S, R, base, codes, hysteresis=hyst)
    cw = run(S, R, base, codes, tilt_on=False)
    eqw = run(S, R, np.full_like(base, 1 / len(UNIVERSE)), codes, tilt_on=False)
    res["full_period_chosen"] = {"tilted": stats(tilt["ret"], cw["ret"], tilt["turnover"]),
                                 "cap_weighted": stats(cw["ret"], None, cw["turnover"]),
                                 "equal_weighted": stats(eqw["ret"], None, eqw["turnover"])}  # fmt: skip
    ok = tilt["limits_ok"][:-1]
    res["risk_limits"] = {"rebalance_days": len(ok), "days_all_limits_held": int(ok.sum()),
                          "share_held": round(float(ok.mean()), 4),
                          "limits": {"name_cap": NAME_CAP, "name_floor": NAME_FLOOR, "sector_cap": SECTOR_CAP,
                                     "turnover_cap": TURNOVER_CAP}}  # fmt: skip
    last = slice(len(days) - 31, len(days) - 1)
    res["daily_last30"] = {"dates": [str(d.date()) for d in days[last]], "version": chosen,
                           "weights": {tk: np.round(tilt["weights"][last, i], 4).tolist() for i, tk in enumerate(UNIVERSE)},
                           "base": {tk: np.round(base[last, i], 4).tolist() for i, tk in enumerate(UNIVERSE)}}  # fmt: skip
    wk = slice(None, None, 5)
    res["chart"] = {
        "dates": [str(d.date()) for d in days[wk]],
        "test_start": TEST[0],
        "weights": {tk: np.round(tilt["weights"][wk, i], 4).tolist() for i, tk in enumerate(UNIVERSE)},
        "equity": {
            "tilted": np.round(np.cumprod(1 + tilt["ret"])[wk], 4).tolist(),
            "tilted_gross": np.round(np.cumprod(1 + tilt["gross"])[wk], 4).tolist(),
            "cap_weighted": np.round(np.cumprod(1 + cw["ret"])[wk], 4).tolist(),
            "equal_weighted": np.round(np.cumprod(1 + eqw["ret"])[wk], 4).tolist(),
        },
    }
    from training.model_card import provenance

    res["provenance"] = provenance()
    OUT.write_text(json.dumps(res, separators=(",", ":")))  # compact: chart arrays would be ~8k lines
    log.info("backtest in %.0fs: %s", time.time() - t0,
             {n: {"val_net_ir": v["validation"]["net_information_ratio"], "test_net_ir": v["test"]["net_information_ratio"],
                  "test_gross_ir": v["test"]["gross_information_ratio"], "turnover": v["test"]["avg_turnover_per_day"],
                  "placebo": v["test"]["placebo"]["share_of_placebos_beaten"], "passes": v["passes"]}
              for n, v in res["versions"].items()})  # fmt: skip
    return res


if __name__ == "__main__":
    main()
