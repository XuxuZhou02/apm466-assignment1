"""Yield-to-maturity solver."""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy.optimize import brentq

from .bond import Bond


def price_from_yield(bond: Bond, ytm: float, settle: date) -> float:
    """Dirty price implied by a yield compounded ``bond.freq`` times per year."""
    times, amounts = bond.cash_flows(settle)
    discount = (1.0 + ytm / bond.freq) ** (-bond.freq * times)
    return float(np.sum(amounts * discount))


def yield_to_maturity(bond: Bond, clean_price: float, settle: date) -> float:
    """Solve for the yield that reprices the bond to its dirty price.

    The yield follows the street convention: compounded at the coupon
    frequency (semi-annual for Government of Canada bonds), with fractional
    exponents for the first, partial coupon period.
    """
    target = bond.dirty_price(clean_price, settle)
    return brentq(lambda y: price_from_yield(bond, y, settle) - target, -0.5, 1.0, xtol=1e-12)
