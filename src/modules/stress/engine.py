"""Module B live engine: event-driven stress tests of the synthetic book.

Trigger v2 (DECISIONS D66), all must hold:
  1. event in {Geopolitical, Macroeconomic, Credit Event} and impact > 7 (configurable);
     a Credit Event also needs event confidence >= 0.7 and an explicit distress term;
  2. corroboration within 6 h for the story's cluster: >= 3 distinct outlets (syndicated copies
     count) or a mention burst z >= 3, AND at least one tier-1 outlet (or an official source).
     The learned impact score is a weak predictor (test rho 0.115, mostly the earnings
     calendar), so one high score alone never fires;
  3. merge: one test per cluster per 6 h; later qualifying signals join the open alert.
Cluster = (event class, country) for Geopolitical/Macroeconomic, (Credit Event, obligor) for
credit. Only high-impact signals count toward corroboration, and the burst z only counts once
the stream has >= 8 windows of history (cold starts look like bursts). Alerts are ranked by
severity = impact x corroboration (corroboration = max(outlets, burst z), capped at 20).
Clocks are compressed by ``time_scale`` in replay.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from engine.config import REPO_ROOT
from engine.nlp.credit import distress_term
from engine.schema import EventClass, Signal, SourceType
from modules.stress.pricing import scenario_for, stress

TRIGGER_EVENTS = {EventClass.GEOPOLITICAL, EventClass.MACROECONOMIC, EventClass.CREDIT_EVENT}
WINDOW_S = 6 * 3600.0
MIN_CREDIT_CONFIDENCE = 0.7
CORROBORATION_CAP = 20.0


def _load_tier1() -> frozenset[str]:
    import yaml

    data = yaml.safe_load((REPO_ROOT / "data" / "reference" / "tier1_outlets.yaml").read_text())
    return frozenset(d.lower() for group in data.values() for d in group)


TIER1 = _load_tier1()


def is_tier1(outlet: str | None, source_type: SourceType | None = None) -> bool:
    if source_type in (SourceType.CENTRAL_BANK, SourceType.FILING):
        return True
    o = (outlet or "").lower().removeprefix("www.")
    return any(o == d or o.endswith("." + d) for d in TIER1)


def _region(country: str | None) -> str:
    if not country:
        return "Global"
    from engine.nlp.entities import get_linker

    e = get_linker().by_country.get(country)
    return (e.region if e and e.region else country) or "Global"


def _country(sig: Signal) -> str | None:
    if sig.entity.country:
        return sig.entity.country
    if sig.geo and sig.geo.get("fips"):
        from engine.nlp.entities import FIPS_TO_ISO

        return FIPS_TO_ISO.get(str(sig.geo["fips"]))
    return None


@dataclass
class Cluster:
    outlets: set[str] = field(default_factory=set)
    extra_copies: int = 0
    max_burst: float = 0.0
    first_t: float = 0.0
    tier1: set[str] = field(default_factory=set)
    signals: deque[tuple[float, str, float]] = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class StressEngine:
    time_scale: float = 1.0
    min_impact: int = 7  # trigger requires impact > min_impact
    min_outlets: int = 3
    min_burst: float = 3.0
    clusters: dict[str, Cluster] = field(default_factory=dict)
    last_fired: dict[str, float] = field(default_factory=dict)
    results: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    alerts: dict[str, dict[str, Any]] = field(default_factory=dict)
    _open: dict[str, str] = field(default_factory=dict)
    decisions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    n_draws: int = 10_000

    @property
    def window(self) -> float:
        return WINDOW_S / self.time_scale

    def cluster_key(self, sig: Signal) -> str:
        """Credit: per named obligor (or sovereign). Geopolitical/macro: per country, so the same
        story from many outlets about one country is one cluster."""
        ev = sig.event.primary
        if ev is EventClass.CREDIT_EVENT:
            return f"{ev.value}|{sig.entity.id}"
        return f"{ev.value}|{_country(sig) or 'Global'}"

    @staticmethod
    def _corr(c: Cluster, min_outlets: int, min_burst: float) -> dict[str, Any]:
        n_out = len(c.outlets) + c.extra_copies
        score = min(max(float(n_out), c.max_burst), CORROBORATION_CAP)
        return {"outlets": n_out, "max_burst_z": round(c.max_burst, 2), "tier1_outlets": sorted(c.tier1)[:5],
                "score": round(score, 2),
                "corroborated": (n_out >= min_outlets or c.max_burst >= min_burst) and bool(c.tier1)}  # fmt: skip

    def corroboration(self, key: str, sig: Signal, t: float) -> dict[str, Any]:
        """Only high-impact signals (impact > min_impact) count: three outlets carrying routine news
        about the same country are not three confirmations of a shock."""
        c = self.clusters.get(key)
        if c is None or t - c.first_t > self.window:
            c = self.clusters[key] = Cluster(first_t=t)
        if sig.impact.score > self.min_impact:
            c.outlets.add(sig.source.outlet)
            c.extra_copies += max(sig.syndication_count - 1, 0)
            c.max_burst = max(c.max_burst, sig.attention.burst_z if sig.attention else 0.0)
            c.signals.append((t, sig.signal_id, sig.impact.score))
            if is_tier1(sig.source.outlet, sig.source.type):
                c.tier1.add(sig.source.outlet)
        return self._corr(c, self.min_outlets, self.min_burst)

    def _alert(
        self, key: str, sig: Signal, t: float, corr: dict[str, Any], fired: dict[str, Any] | None
    ) -> None:
        """Ranked alert queue: each stress test opens an alert; signals merged into that cluster
        within the next 6 h update it (count, best headline, corroboration)."""
        if fired is not None:
            a = {"id": fired["id"], "cluster": key, "event": sig.event.primary.value, "first_t": t,
                 "first_utc": sig.published_at.isoformat(), "merged_signals": 0, "severity": 0.0,
                 "result_id": fired["id"], "loss": fired["loss"], "delta_cet1_bps": fired["delta_cet1_bps"]}  # fmt: skip
            self.alerts[a["id"]] = a
            self._open[key] = a["id"]
        else:
            a = self.alerts.get(self._open.get(key, ""))
            if a is None:
                return
        a["merged_signals"] += 1
        a["last_utc"] = sig.published_at.isoformat()
        a["corroboration"] = corr
        sev = round(sig.impact.score * corr["score"], 1)
        if sev >= a["severity"]:
            a.update(severity=sev, impact=sig.impact.score, headline=sig.evidence.title or sig.evidence.sentence,
                     url=sig.evidence.url, entity=sig.entity.id)  # fmt: skip

    def alert_queue(self, limit: int = 50) -> list[dict[str, Any]]:
        return sorted(self.alerts.values(), key=lambda a: -a["severity"])[:limit]

    def on_signal(self, sig: Signal, now: float | None = None) -> dict[str, Any] | None:
        """Returns the stress result if this signal fired a test, else None."""
        if sig.event.primary not in TRIGGER_EVENTS:
            return None
        t = sig.published_at.timestamp() if now is None else now
        key = self.cluster_key(sig)
        credit = sig.event.primary is EventClass.CREDIT_EVENT
        credit_ok = not credit or (
            sig.event.confidence >= MIN_CREDIT_CONFIDENCE
            and bool(distress_term(f"{sig.evidence.title or ''} {sig.evidence.sentence or ''}"))
        )
        corr = (
            self.corroboration(key, sig, t)
            if credit_ok
            else {"outlets": 0, "max_burst_z": 0.0, "tier1_outlets": [], "score": 0.0, "corroborated": False}
        )
        decision = {"t": t, "cluster": key, "signal_id": sig.signal_id, "impact": sig.impact.score, **corr}
        if sig.impact.score <= self.min_impact:
            decision["fired"], decision["reason"] = False, f"impact {sig.impact.score} <= {self.min_impact}"
        elif not credit_ok:
            decision["fired"], decision["reason"] = (
                False,
                "credit gate (confidence < 0.7 or no distress term)",
            )
        elif not corr["corroborated"]:
            decision["fired"], decision["reason"] = (
                False,
                "not corroborated (needs >= 3 outlets or burst z >= 3, and a tier-1 outlet)",
            )
        elif t - self.last_fired.get(key, -1e18) < self.window:
            decision["fired"], decision["reason"] = (
                False,
                "merged into open alert (cluster tested within 6 h)",
            )
        else:
            decision["fired"], decision["reason"] = True, "trigger"
        self.decisions.append(decision)
        if not decision["fired"]:
            if "merged" in decision["reason"]:
                self._alert(key, sig, t, corr, None)
            return None
        self.last_fired[key] = t
        res = self.run(sig, key, corr)
        self._alert(key, sig, t, corr, res)
        return res

    def run(
        self,
        sig: Signal | None,
        key: str,
        corr: dict[str, Any] | None = None,
        *,
        event: str | None = None,
        impact: float | None = None,
        country: str | None = None,
        sector: str | None = None,
        entity_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        if sig is not None:
            event, impact = sig.event.primary.value, float(sig.impact.score)
            country = _country(sig)
            sector = sig.entity.sector if sig.entity.sector != "Sovereign" else None
            entity_id = sig.entity.id
        assert event is not None and impact is not None
        sc = scenario_for(event, impact, country, sector, entity_id, name)
        t0 = time.perf_counter()
        res = stress(sc, n_draws=self.n_draws)
        res.update({"id": uuid.uuid4().hex[:12], "cluster": key, "fired_at": time.time(), "corroboration": corr,
                    "runtime_s": round(time.perf_counter() - t0, 3),
                    "trigger": None if sig is None else {
                        "signal_id": sig.signal_id, "title": sig.evidence.title or sig.evidence.sentence,
                        "url": sig.evidence.url, "entity": sig.entity.id, "event": sig.event.primary.value,
                        "impact": sig.impact.score, "published_at": sig.published_at.isoformat()}})  # fmt: skip
        self.results.append(res)
        return res
