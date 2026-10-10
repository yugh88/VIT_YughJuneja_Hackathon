"""Module A live engine: tilts the 20-stock index as signals arrive.

Rebalance when a signal on a universe name has impact >= 5, or every 15 minutes (both
clocks compressed by ``time_scale`` in replay). Each rebalance applies hysteresis to the
scores, computes target weights, caps turnover at 10% and logs the signal that caused it.
History is kept for the weights-over-time chart.

Version (DECISIONS D67, picked on validation by the backtest): ``v2_surprise`` tilts on
S_i minus the name's own mean S_i over the previous 30 days, with S_i built from impact >= 6
signals only and hysteresis 0.15. Neither version passed the test-period placebo, so the tilt
is a demonstration of the mechanism, not an edge. Live cold start: with no 30-day history the
prior mean is 0 (surprise = S); the backtest's first 30 days used surprise = 0 instead.
``v1_sentiment`` (the plain aggregate, hysteresis 0.10) remains available.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from engine.aggregate.aggregator import Aggregator
from engine.schema import Signal
from modules.rebalancer.universe import UNIVERSE, base_weights, load_universe
from modules.rebalancer.weights import HYSTERESIS, KAPPA, cap_turnover, hysteresis, target_weights

IMPACT_TRIGGER = 5
TIMER_S = 15 * 60
VERSIONS: dict[str, dict[str, Any]] = {
    "v1_sentiment": {"min_impact": None, "surprise": False, "hysteresis": HYSTERESIS},
    "v2_surprise": {"min_impact": 6, "surprise": True, "hysteresis": 0.15},
}
DEFAULT_VERSION = "v2_surprise"  # chosen on validation (docs/eval/backtest.json)
HALF_LIFE_S = 12 * 3600.0
PRIOR_DAYS = 30


@dataclass
class RebalanceEntry:
    t: float
    reason: str
    turnover: float
    changes: dict[str, float]
    signal: dict[str, Any] | None = None


@dataclass
class Rebalancer:
    aggregator: Aggregator
    time_scale: float = 1.0
    kappa: float = KAPPA
    paused: bool = False
    weights: pd.Series = field(default_factory=lambda: base_weights())
    used_scores: pd.Series = field(default_factory=lambda: pd.Series(0.0, index=UNIVERSE))
    history: deque[tuple[float, dict[str, float]]] = field(default_factory=lambda: deque(maxlen=5000))
    log: deque[RebalanceEntry] = field(default_factory=lambda: deque(maxlen=500))
    last_run: float = 0.0
    version: str = DEFAULT_VERSION
    _num: dict[str, float] = field(default_factory=dict)
    _den: dict[str, float] = field(default_factory=dict)
    _t: dict[str, float] = field(default_factory=dict)
    _snapshots: deque[tuple[float, dict[str, float]]] = field(default_factory=lambda: deque(maxlen=20_000))

    def __post_init__(self) -> None:
        self.sectors = load_universe()["sector"]
        self.base = base_weights()
        self.history.append((time.time(), self.weights.round(6).to_dict()))

    @property
    def cfg(self) -> dict[str, Any]:
        return VERSIONS[self.version]

    def _observe(self, sig: Signal) -> None:
        """Own decayed aggregate over impact-filtered signals (same formula as the aggregator)."""
        tk, t = sig.entity.id, sig.published_at.timestamp()
        lam = math.log(2) / HALF_LIFE_S
        f = math.exp(-lam * max(t - self._t.get(tk, t), 0.0))
        w = sig.sentiment.confidence * sig.source.credibility * sig.novelty
        self._num[tk] = self._num.get(tk, 0.0) * f + w * sig.sentiment.score
        self._den[tk] = self._den.get(tk, 0.0) * f + w
        self._t[tk] = t

    def _filtered(self, t: float) -> pd.Series:
        lam = math.log(2) / HALF_LIFE_S
        out = {}
        for tk in UNIVERSE:
            f = math.exp(-lam * max(t - self._t.get(tk, t), 0.0))
            out[tk] = self._num.get(tk, 0.0) * f / (1.0 + self._den.get(tk, 0.0) * f)
        return pd.Series(out)

    def raw_scores(self, t: float) -> pd.Series:
        if self.cfg["min_impact"] is None:
            return pd.Series({tk: self.aggregator.score(tk, t) for tk in UNIVERSE})
        return self._filtered(t)

    def scores(self, t: float) -> pd.Series:
        s = self.raw_scores(t)
        if not self.cfg["surprise"]:
            return s
        lo = t - PRIOR_DAYS * 86400 / self.time_scale
        prior = [snap for ts, snap in self._snapshots if lo <= ts < t - 1e-9]
        mean = pd.DataFrame(prior).mean() if prior else pd.Series(0.0, index=UNIVERSE)
        return s - mean.reindex(UNIVERSE).fillna(0.0)

    def rebalance(self, t: float, reason: str, sig: Signal | None = None) -> RebalanceEntry | None:
        if self.paused:
            return None
        if self.cfg["surprise"]:
            self._snapshots.append((t, self.raw_scores(t).to_dict()))
        used = hysteresis(self.scores(t), self.used_scores, self.cfg["hysteresis"])
        target = target_weights(self.base, used, self.sectors, self.kappa)
        new, turnover = cap_turnover(self.weights, target)
        self.last_run = t
        if turnover < 1e-6:
            return None
        changes = (new - self.weights).round(5)
        changes = changes[changes.abs() > 1e-4].to_dict()
        self.weights, self.used_scores = new, used
        self.history.append((t, new.round(6).to_dict()))
        entry = RebalanceEntry(
            t,
            reason,
            round(turnover, 5),
            changes,
            None
            if sig is None
            else {
                "signal_id": sig.signal_id,
                "entity": sig.entity.id,
                "title": sig.evidence.title or sig.evidence.sentence,
                "sentiment": sig.sentiment.score,
                "impact": sig.impact.score,
                "event": sig.event.primary.value,
            },
        )
        self.log.append(entry)
        return entry

    def on_signal(self, sig: Signal) -> RebalanceEntry | None:
        mi = self.cfg["min_impact"]
        if sig.entity.id in UNIVERSE and mi is not None and sig.impact.score >= mi:
            self._observe(sig)
        if sig.entity.id in UNIVERSE and sig.impact.score >= IMPACT_TRIGGER:
            return self.rebalance(sig.published_at.timestamp(), f"impact {sig.impact.score} signal", sig)
        return None

    def on_timer(self, t: float) -> RebalanceEntry | None:
        if t - self.last_run >= TIMER_S / self.time_scale:
            return self.rebalance(t, "scheduled (15 min)")
        return None

    def state(self, t: float) -> dict[str, Any]:
        u = load_universe()
        s = self.scores(t)
        return {
            "kappa": self.kappa, "paused": self.paused, "time_scale": self.time_scale,
            "version": self.version, "version_spec": self.cfg,
            "weights": [{"ticker": tk, "name": u.loc[tk, "name"], "sector": u.loc[tk, "sector"],
                         "base": round(float(self.base[tk]), 5), "weight": round(float(self.weights[tk]), 5),
                         "active": round(float(self.weights[tk] - self.base[tk]), 5), "score": round(float(s[tk]), 4)}
                        for tk in UNIVERSE],
            "sector_weights": self.weights.groupby(self.sectors).sum().round(5).to_dict(),
            "rebalances": len(self.log),
        }  # fmt: skip
