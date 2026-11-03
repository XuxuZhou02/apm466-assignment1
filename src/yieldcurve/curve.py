"""Zero-coupon (spot) curve bootstrapping and forward rates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from .bond import Bond


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve.

    Rates are linearly interpolated between nodes and held flat beyond the
    first and last node.
    """

    times: np.ndarray
    rates: np.ndarray

    def __post_init__(self) -> None:
        if len(self.times) != len(self.rates):
            raise ValueError("times and rates must have the same length")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("curve nodes must be strictly increasing")

    def zero_rate(self, t: float | np.ndarray) -> np.ndarray:
        return np.interp(t, self.times, self.rates)

    def discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between ``t1`` and ``t2``."""
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        r1, r2 = self.zero_rate(t1), self.zero_rate(t2)
        return float((r2 * t2 - r1 * t1) / (t2 - t1))

    def price(self, times: np.ndarray, amounts: np.ndarray) -> float:
        return float(np.sum(amounts * self.discount_factor(times)))


def bootstrap(bonds: Sequence[Bond], clean_prices: Sequence[float], settle: date) -> ZeroCurve:
    """Bootstrap a zero curve that exactly reprices every bond.

    Bonds are processed in order of maturity. For each bond, coupons that
    fall before the previously solved node are discounted with the curve
    built so far; coupons between that node and the bond's maturity depend
    on the unknown maturity rate through interpolation, so the maturity
    rate is found with a 1-D root search rather than in closed form.
    """
    order = np.argsort([b.maturity for b in bonds])
    node_t: list[float] = []
    node_r: list[float] = []

    for i in order:
        bond = bonds[i]
        times, amounts = bond.cash_flows(settle)
        target = bond.dirty_price(clean_prices[i], settle)
        maturity = times[-1]

        def pricing_error(r: float) -> float:
            trial = ZeroCurve(np.array(node_t + [maturity]), np.array(node_r + [r]))
            return trial.price(times, amounts) - target

        rate = brentq(pricing_error, -0.5, 1.0, xtol=1e-12)
        node_t.append(maturity)
        node_r.append(rate)

    return ZeroCurve(np.array(node_t), np.array(node_r))
