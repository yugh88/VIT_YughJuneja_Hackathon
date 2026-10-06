"""Per-entity aggregate sentiment S_e(t) - what Module A consumes.

    S_e(t) = sum_i w_i s_i exp(-lambda_k(i) (t - t_i)) / (w0 + sum_i w_i exp(-lambda_k(i) (t - t_i)))
    w_i    = confidence_i x credibility_i x novelty_i

Half-life 12 h for news/filings/central banks, 2 h for social (lambda = ln 2 / half-life).
The neutral prior ``w0`` (one confident, credible signal's worth of weight) does two
jobs: thin evidence is shrunk toward 0, and when the news stops S decays back to 0
at the half-life instead of freezing at its last value (plan: "stale-signal decay").

Bot guard: a social account posting the same cashtag more than 20 times in an hour
gets weight 0. ``time_scale`` compresses every clock for accelerated replay.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from engine.schema import Signal, SourceType

HALF_LIFE_S = {SourceType.SOCIAL: 2 * 3600.0}
DEFAULT_HALF_LIFE_S = 12 * 3600.0
PRIOR_WEIGHT = 1.0
BOT_LIMIT_PER_HOUR = 20
_CASHTAG = re.compile(r"\$[A-Za-z]{1,6}")


@dataclass
class Contribution:
    t: float
    score: float
    weight: float
    lam: float
    signal_id: str
    title: str
    impact: int
    event: str


@dataclass
class EntityState:
    entity_id: str
    name: str
    sector: str | None
    country: str | None
    contributions: deque[Contribution] = field(default_factory=lambda: deque(maxlen=2000))
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=5000))
    n_signals: int = 0
    last_t: float = 0.0


class Aggregator:
    def __init__(
        self,
        time_scale: float = 1.0,
        prior_weight: float = PRIOR_WEIGHT,
        bot_limit_per_hour: int = BOT_LIMIT_PER_HOUR,
    ) -> None:
        self.time_scale = time_scale
        self.prior_weight = prior_weight
        self.bot_limit = bot_limit_per_hour
        self.entities: dict[str, EntityState] = {}
        self._posts: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self.bot_suppressed = 0

    def _lambda(self, source_type: SourceType) -> float:
        hl = HALF_LIFE_S.get(source_type, DEFAULT_HALF_LIFE_S) / self.time_scale
        return math.log(2) / hl

    def _is_bot(self, author: str | None, text: str, t: float) -> bool:
        if not author:
            return False
        flagged = False
        window = 3600.0 / self.time_scale
        for tag in {c.upper() for c in _CASHTAG.findall(text)}:
            q = self._posts[(author, tag)]
            q.append(t)
            while q and q[0] < t - window:
                q.popleft()
            flagged |= len(q) > self.bot_limit
        return flagged

    def add(self, sig: Signal, author: str | None = None, text: str = "") -> float:
        """Fold one signal in; returns the entity's new S."""
        t = sig.published_at.timestamp()
        weight = sig.sentiment.confidence * sig.source.credibility * sig.novelty
        if sig.source.type is SourceType.SOCIAL and self._is_bot(author, text, t):
            weight = 0.0
            self.bot_suppressed += 1
        st = self.entities.get(sig.entity.id)
        if st is None:
            st = self.entities[sig.entity.id] = EntityState(
                sig.entity.id, sig.entity.name, sig.entity.sector, sig.entity.country
            )
        st.contributions.append(
            Contribution(
                t,
                sig.sentiment.score,
                weight,
                self._lambda(sig.source.type),
                sig.signal_id,
                sig.evidence.title or sig.evidence.sentence,
                sig.impact.score,
                sig.event.primary.value,
            )
        )
        st.n_signals += 1
        st.last_t = max(st.last_t, t)
        s = self.score(sig.entity.id, t)
        st.history.append((t, s))
        return s

    def score(self, entity_id: str, t: float) -> float:
        st = self.entities.get(entity_id)
        if st is None:
            return 0.0
        num, den = 0.0, self.prior_weight
        for c in st.contributions:
            if c.t > t:
                continue
            d = c.weight * math.exp(-c.lam * (t - c.t))
            num += d * c.score
            den += d
        return num / den

    def evidence_weight(self, entity_id: str, t: float) -> float:
        st = self.entities.get(entity_id)
        if st is None:
            return 0.0
        return sum(c.weight * math.exp(-c.lam * (t - c.t)) for c in st.contributions if c.t <= t)

    def scores(self, t: float) -> dict[str, float]:
        return {eid: self.score(eid, t) for eid in self.entities}

    def snapshot(self, entity_id: str, t: float, trend_points: int = 48) -> dict[str, Any] | None:
        st = self.entities.get(entity_id)
        if st is None:
            return None
        horizon = 24 * 3600.0 / self.time_scale
        hist = [(ht, hs) for ht, hs in st.history if ht >= t - horizon]
        step = max(1, len(hist) // trend_points)
        drivers = sorted(
            (c for c in st.contributions if c.t <= t),
            key=lambda c: -c.weight * math.exp(-c.lam * (t - c.t)) * abs(c.score),
        )[:5]
        return {
            "entity": {"id": st.entity_id, "name": st.name, "sector": st.sector, "country": st.country},
            "score": round(self.score(entity_id, t), 4),
            "evidence_weight": round(self.evidence_weight(entity_id, t), 3),
            "n_signals": st.n_signals,
            "trend_24h": [{"t": ht, "score": round(hs, 4)} for ht, hs in hist[::step]],
            "top_drivers": [
                {
                    "signal_id": c.signal_id,
                    "title": c.title,
                    "score": c.score,
                    "impact": c.impact,
                    "event": c.event,
                    "decayed_weight": round(c.weight * math.exp(-c.lam * (t - c.t)), 4),
                }
                for c in drivers
            ],
            "half_life_hours": {
                "news": DEFAULT_HALF_LIFE_S / 3600,
                "social": HALF_LIFE_S[SourceType.SOCIAL] / 3600,
            },
            "time_scale": self.time_scale,
        }
