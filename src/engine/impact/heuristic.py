"""Heuristic impact score (1-10): the documented fallback for the learned model.

A logistic function of the same feature families the LightGBM model uses
(plan section 6). Weights are hand-set, not fitted; signals produced this way
carry ``impact.mode = "heuristic"`` and the UI says so. Each feature's
contribution to the logit is kept so the top-3 drivers can be shown.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from engine.schema import EventClass

# Prior "how much does this kind of news move prices" on a 0-1 scale.
EVENT_PRIOR = {
    EventClass.CREDIT_EVENT.value: 0.95, EventClass.GEOPOLITICAL.value: 0.8,
    EventClass.MACROECONOMIC.value: 0.7, EventClass.MERGER_ACQUISITION.value: 0.65,
    EventClass.EARNINGS_GUIDANCE.value: 0.6, EventClass.CYBER_TECH.value: 0.6,
    EventClass.REGULATORY_LEGAL.value: 0.5, EventClass.OPERATIONAL_SUPPLY_CHAIN.value: 0.5,
    EventClass.MANAGEMENT_CHANGE.value: 0.4, EventClass.PRODUCT_LAUNCH.value: 0.3,
    EventClass.ESG_CLIMATE.value: 0.3, EventClass.OTHER.value: 0.1,
}  # fmt: skip

SEVERITY = re.compile(
    r"\b(default\w*|bankrupt\w*|insolven\w*|sanction\w*|war|invasion|invade\w*|fraud\w*|collapse\w*|crash\w*|"
    r"plunge\w*|plummet\w*|soar\w*|surge\w*|record|downgrad\w*|halt\w*|seiz\w*|nationali[sz]\w*|emergency|"
    r"crisis|nuclear|missile\w*|catastroph\w*|scandal|probe|recall\w*|ransomware|breach)\b"
    r"|破産|デフォルト|制裁|戦争|暴落|急落|急騰|违约|破产|制裁|战争|暴跌|दिवालिया|प्रतिबंध|युद्ध|"
    r"insolvenz|pleite|krieg|sanktion|faillite|guerre|quiebra|guerra|falência|дефолт|санкци|войн",
    re.I,
)

WEIGHTS = {
    "bias": -2.6,
    "event_prior": 2.2,
    "abs_sentiment": 1.6,
    "confidence": 0.4,
    "credibility": 0.6,
    "severity": 0.7,  # per lexicon hit, capped at 3
    "log_syndication": 0.45,
    "goldstein": 0.08,  # |Goldstein| on GDELT's 0-10 scale
    "log_sources": 0.25,
}


@dataclass
class ImpactFeatures:
    event: str
    abs_sentiment: float
    confidence: float
    credibility: float
    severity_hits: int = 0
    syndication_count: int = 1
    goldstein: float | None = None
    num_sources: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ImpactResult:
    score: int
    low: int
    high: int
    drivers: list[str]
    probability: float


def severity_hits(text: str) -> int:
    return len(SEVERITY.findall(text))


def score(f: ImpactFeatures) -> ImpactResult:
    contrib = {
        "event_prior": WEIGHTS["event_prior"] * EVENT_PRIOR.get(f.event, 0.1),
        "abs_sentiment": WEIGHTS["abs_sentiment"] * f.abs_sentiment,
        "confidence": WEIGHTS["confidence"] * f.confidence,
        "credibility": WEIGHTS["credibility"] * f.credibility,
        "severity": WEIGHTS["severity"] * min(f.severity_hits, 3),
        "log_syndication": WEIGHTS["log_syndication"] * math.log1p(max(f.syndication_count - 1, 0)),
        "goldstein": WEIGHTS["goldstein"] * abs(f.goldstein or 0.0),
        "log_sources": WEIGHTS["log_sources"] * math.log1p(f.num_sources or 0),
    }
    z = WEIGHTS["bias"] + sum(contrib.values())
    p = 1.0 / (1.0 + math.exp(-z))
    s = max(1, min(10, 1 + round(9 * p)))
    labels = {
        "event_prior": f"event={f.event}", "abs_sentiment": f"|sentiment|={f.abs_sentiment:.2f}",
        "confidence": f"confidence={f.confidence:.2f}", "credibility": f"credibility={f.credibility:.1f}",
        "severity": f"severity_terms={f.severity_hits}", "log_syndication": f"syndication={f.syndication_count}",
        "goldstein": f"goldstein={f.goldstein}", "log_sources": f"gdelt_sources={f.num_sources}",
    }  # fmt: skip
    top = sorted((k for k in contrib if contrib[k] > 0.05), key=lambda k: -contrib[k])[:3]
    return ImpactResult(s, max(1, s - 2), min(10, s + 2), [labels[k] for k in top], round(p, 4))
