"""Module B pricing: scenario -> per-position P&L, expected loss, capital; plus Monte Carlo.

Factors (units as in data/reference/shocks.yaml): equity, financials, em_equity, oil, usd
(fractions), rates, ig, hy, em_spread (bp), vix (points).

Per position, a scenario becomes
* a credit-spread change: IG or HY factor by rating bucket, + half the EM spread move for
  emerging-market names, scaled by targeting (country 1.5x, region 1.2x, sector 1.3x), plus the
  named-obligor notch downgrade and the affected-country add-on;
* a rate change (one USD curve, a stated simplification), an equity move (financials / EM /
  broad index), an FX move for the position's currency, a VIX change.
Valuation: bonds dP/P = -D dy + 1/2 C dy^2; IRS -DV01 x d_rate; FX notional x d_spot; CDS +/- spread
DV01 x d_spread; options delta/gamma/vega. Loans and revolvers: Vasicek conditional PD with the
Basel corporate correlation; the systemic factor is the position's spread shock in standard
deviations of 1-year spread changes (PDs are 1-year; capped at the 99.9% level, 3.09), applied as
PD + V(z) - V(0) so z = 0 leaves PD unchanged; revolver CCF rises from
0.5 toward 0.75 with stress. dEL = PD_s*LGD*EAD_s - PD*LGD*EAD. Capital: Basel IRB risk weights,
CET1 ratio starting at 12% of the book's own RWA, hit by MTM losses and new provisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from engine.config import get_settings
from modules.stress.portfolio import downgrade, notch_pd, notch_spread

FACTORS = ["equity", "financials", "em_equity", "oil", "usd", "rates", "ig", "hy", "em_spread", "vix"]
HORIZON_DAYS = 20
PD_HORIZON_DAYS = 250
Z_CAP = 3.09  # 99.9%
START_CET1 = 0.12
COUNTRY_MULT, REGION_MULT, SECTOR_MULT = 1.5, 1.2, 1.3
BANKING_BOOK = {"term_loan", "revolver", "bond_ig", "bond_hy", "sovereign_bond"}


@dataclass
class Scenario:
    name: str
    event: str
    severity: float  # m = 0.5 + 0.5 * (impact - 7) / 3, in [0.5, 1]
    factors: dict[str, float]  # mean factor moves (already scaled by severity)
    country: str | None = None
    sector: str | None = None
    entity_id: str | None = None
    named_notches: int = 0
    affected_country_spread_bp: float = 0.0
    em_fx_pct: float = 0.0
    named_equity: float = 0.0
    source: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def shock_library() -> dict[str, Any]:
    return yaml.safe_load((get_settings().reference_dir / "shocks.yaml").read_text())  # type: ignore[no-any-return]


@lru_cache(maxsize=1)
def load_book() -> pd.DataFrame:
    return pd.read_csv(get_settings().data_dir / "portfolio" / "book.csv")


@lru_cache(maxsize=1)
def factor_cov() -> np.ndarray:
    """Covariance of 20-day factor moves, from 3 years of daily data (Cholesky-ready)."""
    hist = pd.read_parquet(get_settings().reference_dir / "factor_history.parquet")[FACTORS]
    return hist.cov().to_numpy() * HORIZON_DAYS  # type: ignore[no-any-return]


def severity(impact: float) -> float:
    return float(np.clip(0.5 + 0.5 * (impact - 7) / 3, 0.5, 1.0))


def scenario_for(
    event: str,
    impact: float,
    country: str | None = None,
    sector: str | None = None,
    entity_id: str | None = None,
    name: str | None = None,
) -> Scenario:
    lib = shock_library()["classes"].get(event)
    if lib is None:
        raise ValueError(f"no shock defined for event class {event!r}")
    m = severity(impact)
    f = {k: float(v) * m for k, v in lib.get("measured", {}).items()}
    x = lib.get("extras", {})
    return Scenario(
        name=name or f"{event} (impact {impact:g})", event=event, severity=round(m, 3),
        factors={k: f.get(k, 0.0) for k in FACTORS}, country=country, sector=sector, entity_id=entity_id,
        named_notches=int(x.get("named_notches", 0)) if entity_id else 0,
        affected_country_spread_bp=float(x.get("affected_country_spread_bp", 0.0)) * m if country else 0.0,
        em_fx_pct=float(x.get("em_fx_pct", 0.0)) * m, named_equity=float(x.get("named_equity", 0.0)) * m,
        source={"anchor": lib.get("anchor"), "window": lib.get("window")},
    )  # fmt: skip


def _irb_k(pd_: np.ndarray, lgd: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Basel II/III IRB corporate capital requirement K (per unit EAD)."""
    p = np.clip(pd_, 0.0003, 0.9999)
    e50 = (1 - np.exp(-50 * p)) / (1 - np.exp(-50))
    r = 0.12 * e50 + 0.24 * (1 - e50)
    b = (0.11852 - 0.05478 * np.log(p)) ** 2
    cond = norm.cdf((norm.ppf(p) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r))
    mat = (1 + (np.clip(m, 1, 5) - 2.5) * b) / (1 - 1.5 * b)
    return lgd * (cond - p) * mat  # type: ignore[no-any-return]


def _vasicek_conditional(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    e50 = (1 - np.exp(-50 * p)) / (1 - np.exp(-50))
    rho = 0.12 * e50 + 0.24 * (1 - e50)
    return norm.cdf((norm.ppf(p) + np.sqrt(rho) * z) / np.sqrt(1 - rho))  # type: ignore[no-any-return]


def vasicek(pd_: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Stressed PD for a systemic shock of ``z`` standard deviations above the median economy.

    Vasicek's conditional PD at z = 0 is *below* the unconditional PD (z = 0 is the median
    state, not the mean), which would release provisions on unstressed loans. So the scenario adds
    the *increment* of the conditional PD over the median state: PD + V(z) - V(0). It equals PD at
    z = 0 and rises with z (Basel corporate correlation inside V). A multiplicative form
    PD x V(z)/V(0) was tried first and rejected: ~6x PDs for BBB at z = 1, doubling RWA on one event."""
    p = np.clip(pd_, 1e-6, 0.9999)
    inc = _vasicek_conditional(p, z) - _vasicek_conditional(p, np.zeros_like(p))
    return np.clip(p + inc, p, 0.9999)  # type: ignore[no-any-return]


class Revaluer:
    """Vectorised revaluation of the whole book for one or many factor draws."""

    def __init__(self, book: pd.DataFrame, sc: Scenario) -> None:
        self.b = book
        self.sc = sc
        regions = book.set_index("country")["region"].to_dict()
        reg = regions.get(sc.country) if sc.country else None
        mult = np.ones(len(book))
        if sc.country:
            mult = np.where(
                book["country"] == sc.country, COUNTRY_MULT, np.where(book["region"] == reg, REGION_MULT, 1.0)
            )
        if sc.sector:
            mult = mult * np.where(book["sector"] == sc.sector, SECTOR_MULT, 1.0)
        self.mult = mult
        self.named = (
            (book["entity_id"].astype(str) == str(sc.entity_id)).to_numpy()
            if sc.entity_id
            else np.zeros(len(book), bool)
        )
        letter = book["rating"].str.rstrip("+-")
        self.ig = letter.isin(["AAA", "AA", "A", "BBB"]).to_numpy()
        self.em = book["emerging"].astype(bool).to_numpy()
        self.cls = book["asset_class"].to_numpy()
        self.sov = self.cls == "sovereign_bond"
        # named-obligor downgrade: new PD and the spread jump to the new rating
        new_rating = [
            downgrade(r, sc.named_notches) if n else r
            for r, n in zip(book["rating"], self.named, strict=True)
        ]
        self.pd0 = book["pd"].to_numpy()
        self.pd_named = np.array([notch_pd(r) for r in new_rating])
        self.spread_jump = np.array([notch_spread(r) for r in new_rating]) - book["spread_bp"].to_numpy()
        # PDs are 1-year quantities, so the systemic factor is the spread shock in units of the
        # *1-year* standard deviation of spread changes (20-day covariance scaled to 250 days).
        sd = np.sqrt(np.diag(factor_cov()) * PD_HORIZON_DAYS / HORIZON_DAYS)
        self.sd_ig, self.sd_hy = sd[FACTORS.index("ig")], sd[FACTORS.index("hy")]
        self.in_country = (
            (book["country"] == sc.country).to_numpy() if sc.country else np.zeros(len(book), bool)
        )

    def run(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """X: (draws, factors). Returns arrays shaped (draws, positions)."""
        b, sc = self.b, self.sc
        g = {f: X[:, [i]] for i, f in enumerate(FACTORS)}
        spread = np.where(self.ig, g["ig"], g["hy"]) + np.where(self.em, 0.5 * g["em_spread"], 0.0)
        spread = np.where(self.sov, np.where(self.em, g["em_spread"], 0.0), spread) * self.mult
        spread = spread + np.where(
            self.in_country, sc.affected_country_spread_bp * np.where(self.sov, 1.0, 0.5), 0.0
        )
        spread = spread + np.where(self.named, self.spread_jump, 0.0)
        rates = g["rates"]
        eq = np.where(
            b["sector"].to_numpy() == "Financials",
            g["financials"],
            np.where(self.em, g["em_equity"], g["equity"]),
        )
        eq = eq * self.mult + np.where(self.named, sc.named_equity, 0.0)
        fx = -g["usd"] + np.where(self.em, sc.em_fx_pct * np.where(self.in_country, COUNTRY_MULT, 1.0), 0.0)
        expo = b["exposure"].to_numpy()
        dur, conv = b["mod_duration"].to_numpy(), b["convexity"].to_numpy()
        dv = np.zeros_like(spread)
        bond = np.isin(self.cls, ["bond_ig", "bond_hy", "sovereign_bond"])
        dy = (rates + spread) / 1e4
        dv = np.where(bond, expo * (-dur * dy + 0.5 * conv * dy**2), dv)
        dv = np.where(self.cls == "irs", -b["dv01"].fillna(0).to_numpy() * rates, dv)
        dv = np.where(self.cls == "fx_forward", b["direction"].fillna(0).to_numpy() * expo * fx, dv)
        dv = np.where(self.cls == "cds", b["spread_dv01"].fillna(0).to_numpy() * spread, dv)
        opt = self.cls == "equity_option"
        delta, gamma, vega = (b[c].fillna(0).to_numpy() for c in ("delta", "gamma", "vega"))
        dv = np.where(opt, expo * (delta * eq + 0.5 * gamma * eq**2) + expo * vega * g["vix"], dv)
        # loans: Vasicek conditional PD with the systemic factor = spread shock in SDs (>= 0, <= 3.09)
        z = np.clip(spread / np.where(self.ig, self.sd_ig, self.sd_hy), 0, Z_CAP)
        base_pd = np.where(self.named, self.pd_named, self.pd0)
        pd_s = vasicek(base_pd, z)
        lgd = b["lgd"].to_numpy()
        drawn = b["drawn_share"].fillna(1.0).to_numpy()
        ccf_s = np.where(self.cls == "revolver", 0.5 + 0.25 * z / Z_CAP, 0.0)
        ead0 = np.where(self.cls == "revolver", expo * (drawn + 0.5 * (1 - drawn)), expo)
        ead_s = np.where(self.cls == "revolver", expo * (drawn + ccf_s * (1 - drawn)), expo)
        loan = np.isin(self.cls, ["term_loan", "revolver"])
        d_el = np.where(loan, pd_s * lgd * ead_s - self.pd0 * lgd * ead0, 0.0)
        shape = spread.shape  # (draws, positions); some inputs above are per-position only
        return {
            "dv": np.broadcast_to(dv, shape),
            "d_el": np.broadcast_to(d_el, shape),
            "pd_s": np.broadcast_to(pd_s, shape),
            "ead_s": np.broadcast_to(ead_s, shape),
            "ead0": np.broadcast_to(ead0, shape),
            "spread": spread,
        }


def stress(sc: Scenario, n_draws: int = 10_000, seed: int = 7) -> dict[str, Any]:
    book = load_book()
    rv = Revaluer(book, sc)
    mu = np.array([sc.factors[f] for f in FACTORS])
    det = rv.run(mu[None, :])
    dv, d_el = det["dv"][0], det["d_el"][0]
    # capital (banking book): IRB RWA before/after
    bb = book["asset_class"].isin(BANKING_BOOK).to_numpy()
    lgd, mat = book["lgd"].to_numpy(), book["maturity_y"].to_numpy()
    rwa0 = float((12.5 * _irb_k(book["pd"].to_numpy(), lgd, mat) * det["ead0"][0])[bb].sum())
    rwa1 = float((12.5 * _irb_k(det["pd_s"][0], lgd, mat) * det["ead_s"][0])[bb].sum())
    loss = float(-dv.sum() + d_el.sum())
    cet1_0 = START_CET1 * rwa0
    ratio1 = (cet1_0 - loss) / (rwa0 + (rwa1 - rwa0))
    # Monte Carlo around the scenario mean
    chol = np.linalg.cholesky(factor_cov() + 1e-9 * np.eye(len(FACTORS)))
    X = mu + np.random.default_rng(seed).standard_normal((n_draws, len(FACTORS))) @ chol.T
    mc = rv.run(X)
    losses = -mc["dv"].sum(1) + mc["d_el"].sum(1)
    var99 = float(np.quantile(losses, 0.99))
    es975 = float(losses[losses >= np.quantile(losses, 0.975)].mean())
    hist, edges = np.histogram(losses / 1e6, bins=40)
    pnl = pd.DataFrame(
        {
            "asset_class": book["asset_class"],
            "country": book["country"],
            "sector": book["sector"],
            "name": book["name"],
            "pnl": dv - d_el,
        }
    )
    by_cp = pnl.groupby("name")["pnl"].sum().sort_values().head(10)
    heat = pnl.pivot_table(index="country", columns="sector", values="pnl", aggfunc="sum", fill_value=0.0)
    value0 = float(book["exposure"].sum())
    return {
        "scenario": {**{k: v for k, v in sc.__dict__.items() if k != "factors"}, "factors": {k: round(v, 4) for k, v in sc.factors.items()}},
        "book_value_before": round(value0, 0), "book_value_after": round(value0 - loss, 0),
        "loss": round(loss, 0), "mtm_pnl": round(float(dv.sum()), 0), "delta_expected_loss": round(float(d_el.sum()), 0),
        "waterfall": {k: round(float(v), 0) for k, v in pnl.groupby("asset_class")["pnl"].sum().items()},
        "rwa_before": round(rwa0, 0), "rwa_after": round(rwa1, 0), "delta_rwa": round(rwa1 - rwa0, 0),
        "cet1_ratio_before": START_CET1, "cet1_ratio_after": round(float(ratio1), 5),
        "delta_cet1_bps": round(float((ratio1 - START_CET1) * 1e4), 1),
        # decomposition: losses alone (RWA unchanged) vs. RWA migration alone (no losses)
        "delta_cet1_from_losses_bps": round(float(((cet1_0 - loss) / rwa0 - START_CET1) * 1e4), 1),
        "delta_cet1_from_rwa_bps": round(float((cet1_0 / rwa1 - START_CET1) * 1e4), 1),
        "top_losers": [{"name": n, "pnl": round(float(v), 0)} for n, v in by_cp.items()],
        "heatmap": {"countries": heat.index.tolist(), "sectors": heat.columns.tolist(),
                    "pnl": np.round(heat.to_numpy() / 1e6, 3).tolist()},
        "monte_carlo": {"draws": n_draws, "horizon_days": HORIZON_DAYS, "var_99": round(var99, 0),
                        "es_97_5": round(es975, 0), "mean_loss": round(float(losses.mean()), 0),
                        "histogram_musd": {"counts": hist.tolist(), "edges": np.round(edges, 2).tolist()}},
    }  # fmt: skip
