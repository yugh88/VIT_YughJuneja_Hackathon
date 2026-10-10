"""Module A (rebalancer) and Module B (stress testing)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.schema import (
    Attention,
    EntityRef,
    Event,
    EventClass,
    Evidence,
    Impact,
    Sentiment,
    Signal,
    SourceRef,
    SourceType,
)
from modules.rebalancer.universe import base_weights, load_universe
from modules.rebalancer.weights import apply_limits, cap_turnover, hysteresis, target_weights

U = load_universe()
BASE, SECTORS = base_weights(), U["sector"]


# ------------------------------------------------------------------ Module A
def test_weights_respect_every_limit_and_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    for _ in range(500):
        s = pd.Series(rng.uniform(-1, 1, len(BASE)), index=BASE.index)
        w = target_weights(BASE, s, SECTORS, kappa=float(rng.uniform(0, 2)))
        assert abs(w.sum() - 1) < 1e-9
        assert w.max() <= 0.15 + 1e-9 and w.min() >= 0.01 - 1e-9
        assert w.groupby(SECTORS).sum().max() <= 0.35 + 1e-9
    zero = target_weights(BASE, pd.Series(0.0, index=BASE.index), SECTORS)
    assert np.allclose(zero, apply_limits(BASE, SECTORS))  # no sentiment -> base (after the 1% floor)


def test_positive_sentiment_raises_weight_negative_lowers() -> None:
    s = pd.Series(0.0, index=BASE.index)
    up, down = s.copy(), s.copy()
    up["KO"], down["KO"] = 0.8, -0.8
    w0 = target_weights(BASE, s, SECTORS)["KO"]
    assert target_weights(BASE, up, SECTORS)["KO"] > w0 > target_weights(BASE, down, SECTORS)["KO"]


def test_turnover_cap_and_hysteresis() -> None:
    cur = BASE.copy()
    tgt = pd.Series(1 / len(BASE), index=BASE.index)
    new, to = cap_turnover(cur, tgt)
    assert to == pytest.approx(0.10) and (new - cur).abs().sum() == pytest.approx(0.10, abs=1e-9)
    assert abs(new.sum() - 1) < 1e-9
    last = pd.Series(0.30, index=BASE.index)
    used = hysteresis(pd.Series([0.35] * 10 + [0.55] * 10, index=BASE.index), last)
    assert (used.iloc[:10] == 0.30).all() and (used.iloc[10:] == 0.55).all()  # only moves > 0.10 count


def test_rebalancer_v2_tilts_on_high_impact_surprise_only() -> None:
    from engine.aggregate.aggregator import Aggregator
    from modules.rebalancer.engine import Rebalancer

    rb = Rebalancer(Aggregator(), version="v2_surprise")
    low = _sig(60, "KO slips", 5, EventClass.EARNINGS_GUIDANCE, "a.com", 0, entity="KO", country="US")
    low.sentiment.score = -0.9
    rb.on_signal(low)
    t = low.published_at.timestamp()
    assert rb.raw_scores(t)["KO"] == 0.0  # impact 5 < 6: ignored by v2
    high = _sig(61, "KO plunges", 8, EventClass.EARNINGS_GUIDANCE, "a.com", 1, entity="KO", country="US")
    high.sentiment.score = -0.9
    rb.on_signal(high)
    t = high.published_at.timestamp()
    assert rb.raw_scores(t)["KO"] < 0 and rb.scores(t)["KO"] < 0  # no 30-day history yet: surprise = S


# ------------------------------------------------------------------ Module B
def test_bond_swap_fx_cds_pnl_match_hand_calculation() -> None:
    from modules.stress.pricing import Revaluer, Scenario

    book = pd.DataFrame([
        {"asset_class": "bond_ig", "exposure": 1e8, "mod_duration": 5.0, "convexity": 30.0, "rating": "A",
         "spread_bp": 80, "pd": 0.0006, "lgd": 0.6, "country": "US", "region": "North America", "sector": "Energy",
         "emerging": False, "entity_id": None},
        {"asset_class": "irs", "exposure": 5e7, "dv01": 20_000.0, "mod_duration": 4.0, "convexity": 0, "rating": "AA",
         "spread_bp": 55, "pd": 0.0003, "lgd": 0.6, "country": "US", "region": "North America", "sector": "Financials",
         "emerging": False, "entity_id": None},
        {"asset_class": "fx_forward", "exposure": 2e7, "direction": 1, "mod_duration": 1, "convexity": 0, "rating": "A",
         "spread_bp": 80, "pd": 0.0006, "lgd": 0.6, "country": "DE", "region": "Europe", "sector": "Financials",
         "emerging": False, "entity_id": None},
        {"asset_class": "cds", "exposure": 1e7, "spread_dv01": 4_000.0, "mod_duration": 4, "convexity": 0, "rating": "BB",
         "spread_bp": 250, "pd": 0.006, "lgd": 0.6, "country": "US", "region": "North America", "sector": "Energy",
         "emerging": False, "entity_id": None},
    ])  # fmt: skip
    sc = Scenario("t", "Macroeconomic", 1.0, {"rates": 100.0, "ig": 50.0, "hy": 200.0, "usd": 0.05})
    for c in ("drawn_share", "direction", "dv01", "spread_dv01", "delta", "gamma", "vega"):
        book[c] = book.get(c, np.nan)
    X = np.zeros((1, 10))
    from modules.stress.pricing import FACTORS

    for f, v in sc.factors.items():
        X[0, FACTORS.index(f)] = v
    dv = Revaluer(book, sc).run(X)["dv"][0]
    dy = 0.015  # 100 bp rates + 50 bp IG spread
    assert dv[0] == pytest.approx(1e8 * (-5.0 * dy + 0.5 * 30.0 * dy**2), rel=1e-9, abs=1e-6)
    assert dv[1] == pytest.approx(-20_000.0 * 100, abs=1e-6)  # receive-fixed swap loses DV01 x 100 bp
    assert dv[2] == pytest.approx(2e7 * -0.05, abs=1e-6)  # long EUR, USD +5%
    assert dv[3] == pytest.approx(4_000.0 * 200, abs=1e-6)  # bought protection gains on +200 bp


def test_vasicek_stress_raises_pd_and_irb_weights_are_sane() -> None:
    from modules.stress.pricing import _irb_k, vasicek

    pd0 = np.array([0.0003, 0.0017, 0.006, 0.033, 0.27])
    for z in (0.0, 0.5, 1.5, 3.09):
        assert (vasicek(pd0, np.full(5, z)) >= pd0 - 1e-12).all()
    assert (np.diff([vasicek(np.array([0.0017]), np.array([z]))[0] for z in (0, 1, 2, 3)]) > 0).all()
    rw = 12.5 * _irb_k(np.array([0.0017]), np.array([0.45]), np.array([2.5]))[0]
    assert 0.35 < rw < 0.45  # BBB corporate, LGD 45%, M 2.5: ~40% risk weight


def _sig(i: int, title: str, impact: int, event: EventClass, outlet: str, minutes: float, burst: float = 0.0,
         country: str = "IN", entity: str = "SOV:IN", synd: int = 1) -> Signal:  # fmt: skip
    t0 = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)
    return Signal(
        signal_id=f"s{i}", doc_id=f"d{i}", published_at=t0 + timedelta(minutes=minutes), ingested_at=t0,
        source=SourceRef(name="gdelt_gkg", outlet=outlet, type=SourceType.NEWS, credibility=0.9),
        language="en", entity=EntityRef(id=entity, name=entity, country=country, sector="Sovereign"),
        sentiment=Sentiment(score=0.0, label="neutral", confidence=0.9),
        event=Event(primary=event, confidence=0.9), impact=Impact(score=impact, low=5, high=10),
        novelty=1.0, syndication_count=synd, evidence=Evidence(title=title), attention=Attention(burst_z=burst),
        model_versions={}, latency_ms=0,
    )  # fmt: skip


def test_stress_trigger_needs_impact_and_corroboration_and_debounces() -> None:
    from modules.stress.engine import StressEngine

    se = StressEngine(n_draws=200)
    tbill = _sig(
        1,
        "Auction of 91-Day, 182-Day and 364-Day Treasury Bills",
        9,
        EventClass.MACROECONOMIC,
        "rbi.org.in",
        0,
    )
    assert se.on_signal(tbill) is None  # high impact, but one outlet and no burst
    assert "not corroborated" in se.decisions[-1]["reason"]
    low = _sig(2, "Inflation eases slightly", 7, EventClass.MACROECONOMIC, "a.com", 1, burst=6)
    assert se.on_signal(low) is None  # impact must exceed 7
    outlets = ["o0.com", "o1.com", "o2.com", "reuters.com", "o4.com"]
    war = [_sig(10 + k, "Missile strikes hit Kyiv", 9, EventClass.GEOPOLITICAL, outlets[k], 2 + k, country="UA",
                entity="SOV:UA") for k in range(5)]  # fmt: skip
    fired = [se.on_signal(s) for s in war]
    assert fired[0] is None and fired[1] is None and fired[2] is None  # 3 outlets but none tier-1
    assert fired[3] is not None and fired[3]["loss"] > 0  # a tier-1 outlet joins: corroborated
    assert fired[4] is None and "merged" in se.decisions[-1]["reason"]  # same cluster within 6 h
    q = se.alert_queue()
    assert q[0]["cluster"] == "Geopolitical|UA" and q[0]["merged_signals"] == 2 and q[0]["severity"] == 9 * 5
    assert q[0]["result_id"] == fired[3]["id"]
    burst = _sig(30, "Rating agency cuts sovereign to junk", 9, EventClass.CREDIT_EVENT, "ft.com", 3, burst=4.2,
                 country="TR", entity="SOV:TR")  # fmt: skip
    assert se.on_signal(burst) is not None  # tier-1 outlet + mention burst corroborates


def test_credit_trigger_needs_confidence_and_distress_term() -> None:
    from modules.stress.engine import StressEngine

    se = StressEngine(n_draws=200)
    affirm = [_sig(40 + k, "Fitch affirms Turkey at BB-; outlook stable", 9, EventClass.CREDIT_EVENT,
                   ["reuters.com", "ft.com", "bbc.com"][k], k, country="TR", entity="SOV:TR") for k in range(3)]  # fmt: skip
    assert all(se.on_signal(s) is None for s in affirm)
    assert "credit gate" in se.decisions[-1]["reason"]
    default = [_sig(50 + k, "Turkey misses coupon payment, enters default", 9, EventClass.CREDIT_EVENT,
                    ["reuters.com", "ft.com", "bbc.com"][k], k, country="TR", entity="SOV:TR") for k in range(3)]  # fmt: skip
    assert [se.on_signal(s) is not None for s in default] == [False, False, True]


@pytest.mark.slow
def test_replay_of_24_feb_2022_fires_a_stress_test() -> None:
    """Phase 4 exit: the recorded 24 Feb 2022 GDELT windows, through the real pipeline and trigger."""
    from engine.config import get_settings

    if not (get_settings().replay_dir / "scenario_2022-02-24").exists():
        pytest.skip("scenario not recorded")
    from modules.stress.replay import run

    out = run(n_draws=200, start_hour=0.0, end_hour=8.0)
    assert out["stress_tests_fired"] >= 1, out
    assert any("Geopolitical" in t["cluster"] for t in out["timeline"])
    assert out["alert_queue"] and out["alert_queue"][0]["severity"] >= out["alert_queue"][-1]["severity"]
    Path("docs/eval").mkdir(exist_ok=True, parents=True)
    json.dumps(out)


def test_cet1_after_tax_loss_with_unchanged_rwa_matches_hand_calculation() -> None:
    from modules.stress.pricing import capital

    # Seeded book: RWA before = $5,923.8 M, CET1 = 12% of it = $710.86 M.
    # $100 M pre-tax loss -> $75 M after 25% tax; RWA unchanged.
    # CET1 after = (710.857 - 75) / 5923.810 = 10.7339% -> dCET1 = -75 / 5923.810 x 1e4 = -126.61 bp
    # (10 bp per $1 M of after-tax loss per $1 bn of RWA).
    r = capital(5_923_810_374.0, 5_923_810_374.0, 100e6)
    assert r["loss_after_tax"] == 75e6
    assert r["cet1_capital_before"] == pytest.approx(710_857_245, abs=1)
    assert r["delta_cet1_bps"] == pytest.approx(-126.6, abs=0.05)
    assert r["delta_cet1_from_losses_bps"] == pytest.approx(-126.6, abs=0.05)
    assert r["delta_cet1_from_rwa_bps"] == 0.0 and not r["rwa_cap_binds"]


def test_rwa_growth_from_migration_is_capped_at_15_percent() -> None:
    from modules.stress.pricing import capital

    r = capital(1e9, 1.41e9, 0.0)
    assert r["rwa_cap_binds"] and r["rwa_after"] == pytest.approx(1.15e9) and r["rwa_after_uncapped"] == 1.41e9
    # no loss: 12% x 1.00 / 1.15 = 10.4348% -> -156.5 bp
    assert r["delta_cet1_bps"] == pytest.approx(-156.5, abs=0.05)
    assert not capital(1e9, 1.10e9, 0.0)["rwa_cap_binds"]
