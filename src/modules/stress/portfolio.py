"""Synthetic wholesale banking book (seed 42, regenerable; no real client data).

    python -m modules.stress.portfolio     # -> data/portfolio/book.csv, counterparties.csv

$10 bn across 300 positions and 120 counterparties in 11 sectors and 25 countries, ratings
AAA-CCC centred on BBB. Corporate sector weights come from the transactions-data sector mix
(data/reference/sector_mix.json) with a 2% floor so every sector is present. 40 counterparties
are real names from the gazetteer (synthetic exposures to public companies and banks), so a
Credit Event about a named company can downgrade "its" positions; the rest are synthetic.

Rating -> 1y PD and spread tables are rounded long-run averages in the spirit of the
published S&P/Moody's annual default studies (cited in data/SOURCES.md), with the Basel
3 bp PD floor.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from engine.config import get_settings
from engine.nlp.entities import EntityLinker

log = logging.getLogger("portfolio")
SEED = 42
TOTAL = 10_000_000_000
NOTCHES = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
           "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C"]  # fmt: skip
LETTER_PD = {
    "AAA": 0.0003,
    "AA": 0.0003,
    "A": 0.0006,
    "BBB": 0.0017,
    "BB": 0.0060,
    "B": 0.0330,
    "CCC": 0.2700,
}
LETTER_SPREAD_BP = {"AAA": 40, "AA": 55, "A": 80, "BBB": 130, "BB": 250, "B": 400, "CCC": 900}
RATING_MIX = {"AAA": 0.03, "AA": 0.08, "A": 0.20, "BBB": 0.32, "BB": 0.20, "B": 0.12, "CCC": 0.05}
COUNTRIES = {  # country: (weight, region, currency, emerging?)
    "US": (0.30, "North America", "USD", False), "GB": (0.07, "Europe", "GBP", False),
    "DE": (0.06, "Europe", "EUR", False), "FR": (0.05, "Europe", "EUR", False), "IT": (0.03, "Europe", "EUR", False),
    "ES": (0.03, "Europe", "EUR", False), "NL": (0.02, "Europe", "EUR", False), "CH": (0.02, "Europe", "CHF", False),
    "PL": (0.015, "Europe", "PLN", True), "RU": (0.015, "Europe", "RUB", True), "TR": (0.015, "Middle East", "TRY", True),
    "SA": (0.02, "Middle East", "SAR", True), "AE": (0.015, "Middle East", "AED", True),
    "JP": (0.06, "Asia Pacific", "JPY", False), "CN": (0.05, "Asia Pacific", "CNY", True),
    "KR": (0.02, "Asia Pacific", "KRW", False), "TW": (0.015, "Asia Pacific", "TWD", False),
    "IN": (0.05, "Asia Pacific", "INR", True), "SG": (0.02, "Asia Pacific", "SGD", False),
    "ID": (0.015, "Asia Pacific", "IDR", True), "AU": (0.02, "Asia Pacific", "AUD", False),
    "CA": (0.03, "North America", "CAD", False), "BR": (0.02, "Latin America", "BRL", True),
    "MX": (0.02, "Latin America", "MXN", True), "ZA": (0.015, "Africa", "ZAR", True),
}  # fmt: skip
ASSET_CLASSES = {  # class: (share of exposure, positions)
    "term_loan": (0.35, 105), "revolver": (0.10, 30), "bond_ig": (0.20, 60), "bond_hy": (0.05, 15),
    "sovereign_bond": (0.10, 20), "irs": (0.10, 30), "fx_forward": (0.05, 20), "cds": (0.03, 10),
    "equity_option": (0.02, 10),
}  # fmt: skip


def notch_pd(notch: str) -> float:
    """Log-linear interpolation of 1y PD across notches between letter-grade anchors."""
    anchors = {n: LETTER_PD[n] for n in LETTER_PD}
    idx = {n: i for i, n in enumerate(NOTCHES)}
    pts = sorted((idx[k], np.log(v)) for k, v in anchors.items())
    i = idx[notch]
    if notch in ("CC", "C"):
        return 0.45 if notch == "CC" else 0.60
    xs, ys = zip(*pts, strict=True)
    return float(np.exp(np.interp(i, xs, ys)))


def notch_spread(notch: str) -> float:
    letter = notch.rstrip("+-")
    letter = letter if letter in LETTER_SPREAD_BP else "CCC"
    return float(LETTER_SPREAD_BP[letter])


def downgrade(notch: str, n: int) -> str:
    return NOTCHES[min(NOTCHES.index(notch) + n, len(NOTCHES) - 1)]


def sector_weights() -> dict[str, float]:
    mix = json.loads((get_settings().reference_dir / "sector_mix.json").read_text())["sector_share"]
    sectors = [
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    ]
    w = np.array([max(mix.get(s, 0.0), 0.02) for s in sectors])
    return dict(zip(sectors, w / w.sum(), strict=True))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    linker = EntityLinker()
    sw = sector_weights()
    cc = list(COUNTRIES)
    cw = np.array([COUNTRIES[c][0] for c in cc])
    cw = cw / cw.sum()
    letters = list(RATING_MIX)
    lw = np.array(list(RATING_MIX.values()))
    # 40 real names (public companies/banks in the book's countries), 80 synthetic counterparties
    real = [e for e in linker.entities.values() if e.kind != "sovereign" and e.country in COUNTRIES]
    real = list(rng.choice(np.array(real, dtype=object), 40, replace=False))
    cps = []
    for i in range(120):
        if i < 40:
            e = real[i]
            name, country, sector, eid = e.name, e.country, e.sector, e.id
        else:  # first one synthetic counterparty per country, so all 25 are present
            country = cc[i - 40] if i - 40 < len(cc) else str(rng.choice(cc, p=cw))
            sector = str(rng.choice(list(sw), p=list(sw.values())))
            name, eid = f"CP-{i:03d} {sector.split()[0]} ({country})", None
        letter = str(rng.choice(letters, p=lw))
        notch = rng.choice([n for n in NOTCHES if n.rstrip("+-") == letter]) if letter != "CCC" else "CCC"
        if i in (43, 77, 101):  # guarantee a few distressed names (CCC) for the stress test to bite on
            notch = "CCC"
        cps.append(
            {
                "cp_id": f"CP{i:03d}",
                "name": name,
                "entity_id": eid,
                "country": country,
                "region": COUNTRIES[country][1],
                "currency": COUNTRIES[country][2],
                "emerging": COUNTRIES[country][3],
                "sector": sector,
                "rating": str(notch),
            }
        )
    cpd = pd.DataFrame(cps)
    rows = []
    pid = 0
    order = rng.permutation(len(cpd))  # loans/revolvers cycle through every counterparty
    lend = 0
    for cls, (share, n) in ASSET_CLASSES.items():
        sizes = rng.lognormal(0, 0.6, n)
        sizes = sizes / sizes.sum() * share * TOTAL
        for k in range(n):
            if cls == "sovereign_bond":
                country = cc[k % len(cc)] if k < len(cc) else str(rng.choice(cc, p=cw))
                cp = {
                    "cp_id": f"SOV-{country}",
                    "name": f"Government of {country}",
                    "entity_id": f"SOV:{country}",
                    "country": country,
                    "region": COUNTRIES[country][1],
                    "currency": COUNTRIES[country][2],
                    "emerging": COUNTRIES[country][3],
                    "sector": "Sovereign",
                    "rating": "BBB" if COUNTRIES[country][3] else str(rng.choice(["AAA", "AA", "AA-", "A+"])),
                }
            elif cls in ("term_loan", "revolver"):
                cp = cpd.iloc[int(order[lend % len(cpd)])].to_dict()
                lend += 1
            else:
                pool = (
                    cpd
                    if cls not in ("bond_hy",)
                    else cpd[cpd["rating"].str.rstrip("+-").isin(["BB", "B", "CCC"])]
                )
                pool = (
                    pool
                    if cls != "bond_ig"
                    else cpd[cpd["rating"].str.rstrip("+-").isin(["AAA", "AA", "A", "BBB"])]
                )
                cp = pool.iloc[int(rng.integers(len(pool)))].to_dict()
            mat = float(np.round(rng.uniform(1, 10 if cls != "revolver" else 5), 1))
            y = 0.04 + notch_spread(cp["rating"]) / 1e4
            dur = (1 - (1 + y) ** -mat) / y  # modified-duration approximation for a par bond
            row = {"position_id": f"P{pid:04d}", "asset_class": cls, **cp, "exposure": float(sizes[k]),
                   "maturity_y": mat, "pd": notch_pd(cp["rating"]), "lgd": 0.45 if cls in ("term_loan", "revolver") else 0.6,
                   "spread_bp": notch_spread(cp["rating"]), "mod_duration": round(dur, 3),
                   "convexity": round(dur**2 * 1.1, 3)}  # fmt: skip
            if cls == "revolver":
                row["drawn_share"], row["ccf"] = 0.5, 0.5
            if cls == "irs":  # notional-equivalent exposure; + = receive fixed (gains when rates fall)
                row["direction"] = int(rng.choice([1, 1, -1]))
                row["dv01"] = row["direction"] * row["exposure"] * dur / 1e4
            if cls == "fx_forward":  # long foreign currency vs USD (+) or short (-)
                row["direction"] = int(rng.choice([1, -1]))
                row["currency"] = (
                    cp["currency"]
                    if cp["currency"] != "USD"
                    else str(rng.choice(["EUR", "JPY", "GBP", "INR"]))
                )
            if cls == "cds":  # + = bought protection (gains when spreads widen)
                row["direction"] = 1 if k < 6 else -1
                row["spread_dv01"] = row["direction"] * row["exposure"] * min(dur, 5) / 1e4
            if cls == "equity_option":  # delta/gamma/vega per unit of exposure; TRS = pure delta
                kind = rng.choice(["call", "put", "trs"])
                row["option_type"] = str(kind)
                row["delta"] = {"call": 0.55, "put": -0.45, "trs": 1.0}[kind]
                row["gamma"] = 0.0 if kind == "trs" else 2.0
                row["vega"] = 0.0 if kind == "trs" else 0.004  # P&L per vol point per unit exposure
            rows.append(row)
            pid += 1
    book = pd.DataFrame(rows)
    return book, cpd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    book, cps = build()
    out = get_settings().data_dir / "portfolio"
    out.mkdir(parents=True, exist_ok=True)
    book.to_csv(out / "book.csv", index=False)
    cps.to_csv(out / "counterparties.csv", index=False)
    corp = book[book["asset_class"] != "sovereign_bond"]
    log.info(
        "book: %d positions, $%.2f bn, %d corporate counterparties (+%d sovereigns), %d countries, %d sectors",
        len(book),
        book["exposure"].sum() / 1e9,
        corp["cp_id"].nunique(),
        book.loc[book["asset_class"] == "sovereign_bond", "cp_id"].nunique(),
        book["country"].nunique(),
        corp["sector"].nunique(),
    )
    log.info("%s", book.groupby("asset_class")["exposure"].sum().div(1e9).round(2).to_dict())
    log.info("ratings: %s", book["rating"].str.rstrip("+-").value_counts().to_dict())


if __name__ == "__main__":
    main()
