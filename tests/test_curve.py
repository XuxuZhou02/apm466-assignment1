from datetime import date

import numpy as np
import pytest

from yieldcurve import Bond, ZeroCurve, bootstrap

SETTLE = date(2026, 1, 15)

BONDS = [
    Bond("A", "A", 0.0400, date(2026, 8, 1)),
    Bond("B", "B", 0.0300, date(2027, 2, 1)),
    Bond("C", "C", 0.0250, date(2027, 8, 1)),
    Bond("D", "D", 0.0200, date(2028, 6, 1)),  # off-cycle coupon dates
    Bond("E", "E", 0.0225, date(2029, 12, 1)),
    Bond("F", "F", 0.0275, date(2031, 3, 1)),
]


def clean_prices_from(curve: ZeroCurve) -> list[float]:
    out = []
    for b in BONDS:
        t, cf = b.cash_flows(SETTLE)
        out.append(curve.price(t, cf) - b.accrued_interest(SETTLE))
    return out


def test_flat_curve_is_recovered():
    flat = ZeroCurve(np.array([0.1, 10.0]), np.array([0.03, 0.03]))
    curve = bootstrap(BONDS, clean_prices_from(flat), SETTLE)
    assert curve.rates == pytest.approx(0.03, abs=1e-10)


def test_bootstrap_reprices_every_bond():
    true = ZeroCurve(np.array([0.5, 1.0, 2.0, 3.0, 5.5]), np.array([0.045, 0.040, 0.032, 0.030, 0.035]))
    prices = clean_prices_from(true)
    curve = bootstrap(BONDS, prices, SETTLE)
    for b, clean in zip(BONDS, prices):
        t, cf = b.cash_flows(SETTLE)
        assert curve.price(t, cf) == pytest.approx(b.dirty_price(clean, SETTLE), abs=1e-8)


def test_bootstrap_is_order_independent():
    true = ZeroCurve(np.array([0.5, 5.5]), np.array([0.04, 0.03]))
    prices = clean_prices_from(true)
    shuffled = [3, 0, 5, 1, 4, 2]
    a = bootstrap(BONDS, prices, SETTLE)
    b = bootstrap([BONDS[i] for i in shuffled], [prices[i] for i in shuffled], SETTLE)
    assert a.rates == pytest.approx(b.rates)


def test_forward_rate_consistency():
    curve = ZeroCurve(np.array([1.0, 2.0, 5.0]), np.array([0.02, 0.03, 0.035]))
    # Growth to T2 = growth to T1 x forward growth from T1 to T2.
    f = curve.forward_rate(1.0, 2.0)
    assert np.exp(0.02 * 1.0 + f * 1.0) == pytest.approx(np.exp(0.03 * 2.0))
    assert f == pytest.approx(0.04)


def test_forward_of_flat_curve_is_flat():
    curve = ZeroCurve(np.array([1.0, 5.0]), np.array([0.03, 0.03]))
    assert curve.forward_rate(1.0, 4.0) == pytest.approx(0.03)


def test_curve_rejects_unsorted_nodes():
    with pytest.raises(ValueError):
        ZeroCurve(np.array([2.0, 1.0]), np.array([0.03, 0.03]))
