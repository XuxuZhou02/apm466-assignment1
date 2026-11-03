from datetime import date

import numpy as np
import pytest

from yieldcurve import Bond, price_from_yield, yield_to_maturity


@pytest.fixture
def bond():
    return Bond("TEST", "CAN 3.0 Mar 30", coupon=0.03, maturity=date(2030, 3, 1))


def test_coupon_schedule_steps_back_from_maturity(bond):
    dates = bond.coupon_dates(date(2026, 1, 15))
    assert dates[0] == date(2026, 3, 1)
    assert dates[-1] == date(2030, 3, 1)
    assert len(dates) == 9
    assert bond.previous_coupon_date(date(2026, 1, 15)) == date(2025, 9, 1)


def test_accrued_interest_act365(bond):
    assert bond.accrued_interest(date(2025, 9, 1)) == 0.0
    settle = date(2026, 1, 15)
    days = (settle - date(2025, 9, 1)).days
    assert bond.accrued_interest(settle) == pytest.approx(3.0 * days / 365)


def test_cash_flows_include_principal(bond):
    times, amounts = bond.cash_flows(date(2026, 1, 15))
    assert np.all(np.diff(times) > 0)
    assert amounts[:-1] == pytest.approx(1.5)
    assert amounts[-1] == pytest.approx(101.5)


def test_par_bond_on_coupon_date_yields_its_coupon(bond):
    # ACT/365 year fractions are not exactly 0.5 per period, hence the tolerance.
    settle = date(2029, 3, 1)
    assert price_from_yield(bond, 0.03, settle) == pytest.approx(100.0, abs=0.02)
    assert yield_to_maturity(bond, 100.0, settle) == pytest.approx(0.03, abs=2e-4)


@pytest.mark.parametrize("y", [0.01, 0.035, 0.06])
def test_yield_round_trip(bond, y):
    settle = date(2026, 1, 15)
    dirty = price_from_yield(bond, y, settle)
    clean = dirty - bond.accrued_interest(settle)
    assert yield_to_maturity(bond, clean, settle) == pytest.approx(y, abs=1e-10)


def test_price_decreases_with_yield(bond):
    settle = date(2026, 1, 15)
    prices = [price_from_yield(bond, y, settle) for y in (0.01, 0.02, 0.03)]
    assert prices[0] > prices[1] > prices[2]
