"""Sector weights for the synthetic book, from the 'provided sample transaction data'.

    python -m modules.stress.sector_mix scratch/transactions.parquet

The case study asks for a synthetic portfolio built from the sample transaction data. The
only such public set is the Kaggle "Financial Transactions Dataset" (card transactions with
merchant category codes); a keyless Hugging Face mirror is `thiru1711/Financial_Transactions`.
We map each MCC to a GICS sector and use the share of absolute spend per sector as the
corporate-lending sector mix. Caveat: retail card spend is a proxy for which sectors a lender
is exposed to, not a measurement of a wholesale book.
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

from engine.config import get_settings

STAPLES = {5411, 5422, 5441, 5451, 5462, 5499, 5300, 5310, 5311, 5331, 5921, 5993, 5193}
HEALTH = {5912, 5975, 5976} | set(range(8000, 8100))
ENERGY = {5541, 5542, 5983}
MATERIALS = {5211, 5231, 5251, 5261, 5039, 5051}
TECH = {5045, 5732, 5734, 7372, 7379, 4816}
COMM = set(range(4800, 4900)) | {5815, 5816, 5817, 5818, 7832, 7841, 5994}
REAL_ESTATE = {6513}


def mcc_sector(mcc: int) -> str | None:
    if mcc in STAPLES:
        return "Consumer Staples"
    if mcc in HEALTH:
        return "Health Care"
    if mcc in ENERGY:
        return "Energy"
    if mcc in MATERIALS:
        return "Materials"
    if mcc in TECH:
        return "Information Technology"
    if mcc in COMM:
        return "Communication Services"
    if mcc in REAL_ESTATE:
        return "Real Estate"
    if mcc == 4900:
        return "Utilities"
    if 6000 <= mcc <= 6999:
        return "Financials"
    if 3000 <= mcc <= 3499 or 4000 <= mcc <= 4799 or 7300 <= mcc <= 7399 or 8100 <= mcc <= 8999:
        return "Industrials"
    if 3500 <= mcc <= 3999 or 5000 <= mcc <= 5999 or 7000 <= mcc <= 7299 or 7400 <= mcc <= 7999:
        return "Consumer Discretionary"
    return None  # government and unclassified codes are left out


def main(path: str) -> None:
    df = pd.read_parquet(path, columns=["mcc", "amount"])
    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    df["sector"] = df["mcc"].map(lambda m: mcc_sector(int(m)) if pd.notna(m) else None)
    spend = df.assign(a=df["amount"].abs()).groupby("sector")["a"].sum()
    share = (spend / spend.sum()).round(4).sort_values(ascending=False)
    out = {"source": "Kaggle Financial Transactions Dataset (HF mirror thiru1711/Financial_Transactions), first "
                     "parquet shard", "rows_used": len(df), "unmapped_share": round(float(df["sector"].isna().mean()), 4),
           "method": "share of absolute card spend by MCC mapped to GICS sector (modules/stress/sector_mix.py)",
           "computed_at": time.strftime("%Y-%m-%d"), "sector_share": share.to_dict()}  # fmt: skip
    p = get_settings().reference_dir / "sector_mix.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
