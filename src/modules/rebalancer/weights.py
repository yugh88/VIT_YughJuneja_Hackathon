"""Module A weight rule and risk controls (pure functions, shared by live engine and backtest).

    w*_i = b_i * clip(exp(kappa * S_i), 0.5, 2.0), renormalised, then projected onto the limits:
    single name <= 15%, single name >= 1%, sector <= 35% (iterated until stable, <= 25 rounds).

Trading controls: hysteresis (a name's sentiment is only re-read when it moved > 0.10 since the
last rebalance) and a turnover cap (sum |dw| <= 10% per rebalance: a larger move is taken
partially, in the same direction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KAPPA = 0.5
TILT_MIN, TILT_MAX = 0.5, 2.0
NAME_CAP, NAME_FLOOR, SECTOR_CAP = 0.15, 0.01, 0.35
HYSTERESIS = 0.10
TURNOVER_CAP = 0.10
MAX_ROUNDS = 25  # the plan said <= 5; 5 can leave ~1e-5 violations, so iterate until stable (<= 25)


def tilt(base: pd.Series, scores: pd.Series, kappa: float = KAPPA) -> pd.Series:
    s = scores.reindex(base.index).fillna(0.0)
    raw = base * np.clip(np.exp(kappa * s), TILT_MIN, TILT_MAX)
    return raw / raw.sum()


def apply_limits(
    w: pd.Series,
    sectors: pd.Series,
    name_cap: float = NAME_CAP,
    floor: float = NAME_FLOOR,
    sector_cap: float = SECTOR_CAP,
    rounds: int = MAX_ROUNDS,
) -> pd.Series:
    """Project weights onto name cap / floor / sector cap. Each round fixes violators at their bound
    and redistributes the difference pro rata over names that still have room."""
    codes, _ = pd.factorize(sectors.reindex(w.index))
    return pd.Series(limits_np(w.to_numpy(float), codes, name_cap, floor, sector_cap, rounds), index=w.index)


def limits_np(
    w: np.ndarray,
    codes: np.ndarray,
    name_cap: float = NAME_CAP,
    floor: float = NAME_FLOOR,
    sector_cap: float = SECTOR_CAP,
    rounds: int = MAX_ROUNDS,
) -> np.ndarray:
    w = w / w.sum()
    n_sec = int(codes.max()) + 1
    for _ in range(rounds):
        changed = False
        fixed = np.zeros(len(w), bool)
        sec = np.bincount(codes, weights=w, minlength=n_sec)
        for k in np.flatnonzero(sec > sector_cap + 1e-12):
            members = codes == k
            excess = sec[k] - sector_cap
            w[members] *= sector_cap / sec[k]
            fixed |= members
            others = ~members
            w[others] += excess * w[others] / w[others].sum()
            changed = True
        over, under = w > name_cap + 1e-12, w < floor - 1e-12
        if over.any() or under.any():
            delta = (w[over] - name_cap).sum() - (floor - w[under]).sum()
            w[over], w[under] = name_cap, floor
            fixed |= over | under
            free = ~fixed
            if free.any() and w[free].sum() > 0:
                w[free] += delta * w[free] / w[free].sum()
            changed = True
        w = w / w.sum()
        if not changed:
            break
    return w


def target_weights(base: pd.Series, scores: pd.Series, sectors: pd.Series, kappa: float = KAPPA) -> pd.Series:
    return apply_limits(tilt(base, scores, kappa), sectors)


def hysteresis(new: pd.Series, last_used: pd.Series, band: float = HYSTERESIS) -> pd.Series:
    """Scores actually used: a name's new score replaces the old one only if it moved more than ``band``."""
    last = last_used.reindex(new.index).fillna(0.0)
    return new.where((new - last).abs() > band, last)


def cap_turnover(current: pd.Series, target: pd.Series, cap: float = TURNOVER_CAP) -> tuple[pd.Series, float]:
    """Move toward ``target`` but trade at most ``cap`` (sum of |dw|). Returns (weights, turnover)."""
    diff = target.reindex(current.index).fillna(0.0) - current
    turnover = float(diff.abs().sum())
    if turnover <= cap:
        return target.reindex(current.index).fillna(0.0), turnover
    step = current + diff * (cap / turnover)
    return step / step.sum(), cap
