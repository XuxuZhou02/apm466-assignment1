"""Loading the bond universe and daily price quotes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bond import Bond


def load_bonds(path: str | Path) -> list[Bond]:
    """Read bond static data (ISIN, name, coupon, maturity)."""
    df = pd.read_csv(path, parse_dates=["maturity_date"])
    return [
        Bond(isin=row.isin, name=row.name, coupon=float(row.coupon), maturity=row.maturity_date.date())
        for row in df.itertuples(index=False)
    ]


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read clean prices: one row per trading date, one column per ISIN."""
    prices = pd.read_csv(path, index_col="date", parse_dates=["date"])
    prices.index = prices.index.date
    prices.index.name = "date"
    return prices
