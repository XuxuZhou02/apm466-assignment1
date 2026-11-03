from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldcurve import NelsonSiegel, log_returns, pca
from yieldcurve.data import load_bonds, load_prices
from yieldcurve.pipeline import run_analysis

DATA = Path(__file__).resolve().parents[1] / "data"


def test_log_returns():
    levels = pd.DataFrame({"a": [1.0, np.e, np.e**3]})
    assert log_returns(levels)["a"].tolist() == pytest.approx([1.0, 2.0])


def test_pca_recovers_known_structure():
    level = np.ones(4) / 2.0
    slope = np.array([-3, -1, 1, 3]) / np.sqrt(20)
    cov = 9.0 * np.outer(level, level) + 1.0 * np.outer(slope, slope) + 0.01 * np.eye(4)
    labels = ["1Y", "2Y", "3Y", "4Y"]
    res = pca(pd.DataFrame(cov, index=labels, columns=labels))

    assert res.eigenvalues.is_monotonic_decreasing
    assert res.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert res.eigenvectors["PC1"].to_numpy() == pytest.approx(level, abs=1e-8)
    assert abs(res.eigenvectors["PC2"].to_numpy() @ slope) == pytest.approx(1.0)
    # Eigenvectors are orthonormal.
    v = res.eigenvectors.to_numpy()
    assert v.T @ v == pytest.approx(np.eye(4), abs=1e-10)


def test_nelson_siegel_fits_its_own_curve():
    true = NelsonSiegel(0.035, 0.01, -0.01, tau=1.5)
    t = np.linspace(0.5, 5, 10)
    fit = NelsonSiegel.fit(t, true(t))
    assert fit(t) == pytest.approx(true(t), abs=1e-6)


def test_nelson_siegel_is_robust_to_one_bad_quote():
    true = NelsonSiegel(0.035, 0.01, -0.01, tau=1.5)
    t = np.linspace(0.5, 5, 10)
    y = true(t)
    y[5] += 0.02  # 200bp error on one bond
    residuals = (y - NelsonSiegel.fit(t, y)(t)) * 1e4
    assert abs(residuals[5]) > 150
    assert np.max(np.abs(np.delete(residuals, 5))) < 10


def test_pipeline_on_sample_data():
    res = run_analysis(load_bonds(DATA / "bonds.csv"), load_prices(DATA / "clean_prices.csv"))
    assert res.excluded == ["CA135087Q988"]  # CAN 4.0 Mar 29: quote ~190bp off the curve
    assert res.spot_rates.shape == (10, 5)
    assert res.forward_rates.shape == (10, 4)
    assert res.spot_rates.notna().all().all()
    assert res.yield_pca.explained_variance_ratio.iloc[0] > 0.5
