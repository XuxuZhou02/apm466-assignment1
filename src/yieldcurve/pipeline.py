"""End-to-end analysis: yields -> quote screen -> spot/forward curves -> PCA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .bond import Bond
from .curve import ZeroCurve, bootstrap
from .nelson_siegel import NelsonSiegel
from .pca import PCAResult, log_returns, pca
from .yields import yield_to_maturity

KEY_TENORS = (1.0, 2.0, 3.0, 4.0, 5.0)
FORWARD_START = 1.0
FORWARD_TENORS = (1.0, 2.0, 3.0, 4.0)


@dataclass
class AnalysisResult:
    bonds: list[Bond]
    ytm: pd.DataFrame  # date x ISIN
    maturities: pd.DataFrame  # date x ISIN, years to maturity
    ns_fits: dict[date, NelsonSiegel]
    residuals_bp: pd.DataFrame  # date x ISIN, YTM minus Nelson-Siegel fit
    excluded: list[str]  # ISINs dropped by the quote screen
    spot_curves: dict[date, ZeroCurve]
    key_yields: pd.DataFrame  # date x tenor
    spot_rates: pd.DataFrame  # date x tenor
    forward_rates: pd.DataFrame  # date x forward label
    yield_cov: pd.DataFrame
    forward_cov: pd.DataFrame
    yield_pca: PCAResult
    forward_pca: PCAResult

    @property
    def included(self) -> list[Bond]:
        return [b for b in self.bonds if b.isin not in self.excluded]


def _tenor_label(t: float) -> str:
    return f"{t:g}Y"


def compute_yields(bonds: list[Bond], prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ytm = pd.DataFrame(index=prices.index, columns=[b.isin for b in bonds], dtype=float)
    ttm = ytm.copy()
    for d in prices.index:
        for b in bonds:
            ytm.loc[d, b.isin] = yield_to_maturity(b, prices.loc[d, b.isin], d)
            ttm.loc[d, b.isin] = b.time_to_maturity(d)
    return ytm, ttm


def screen_quotes(
    ytm: pd.DataFrame, ttm: pd.DataFrame, threshold_bp: float
) -> tuple[dict[date, NelsonSiegel], pd.DataFrame, list[str]]:
    """Flag bonds whose yields sit persistently far from a robust smooth curve.

    A Nelson-Siegel curve is fitted each day with a soft-L1 loss. A bond is
    excluded when its median absolute residual over the sample exceeds
    ``threshold_bp``; using the median keeps the bond set stable across days,
    so every daily curve is built from the same instruments.
    """
    fits = {d: NelsonSiegel.fit(ttm.loc[d].to_numpy(), ytm.loc[d].to_numpy()) for d in ytm.index}
    residuals = pd.DataFrame(
        {d: (ytm.loc[d] - fits[d](ttm.loc[d].to_numpy())) * 1e4 for d in ytm.index}
    ).T
    residuals.index.name = "date"
    median_abs = residuals.abs().median()
    excluded = median_abs[median_abs > threshold_bp].index.tolist()
    return fits, residuals, excluded


def run_analysis(bonds: list[Bond], prices: pd.DataFrame, outlier_threshold_bp: float = 75.0) -> AnalysisResult:
    ytm, ttm = compute_yields(bonds, prices)
    ns_fits, residuals, excluded = screen_quotes(ytm, ttm, outlier_threshold_bp)
    used = [b for b in bonds if b.isin not in excluded]

    spot_curves: dict[date, ZeroCurve] = {}
    key_yields, spot_rates, forward_rates = {}, {}, {}
    for d in prices.index:
        curve = bootstrap(used, [prices.loc[d, b.isin] for b in used], d)
        spot_curves[d] = curve

        # Par-style yield curve: linear interpolation of YTM across maturities.
        t_used = ttm.loc[d, [b.isin for b in used]].to_numpy()
        y_used = ytm.loc[d, [b.isin for b in used]].to_numpy()
        order = np.argsort(t_used)
        key_yields[d] = {_tenor_label(t): np.interp(t, t_used[order], y_used[order]) for t in KEY_TENORS}

        spot_rates[d] = {_tenor_label(t): float(curve.zero_rate(t)) for t in KEY_TENORS}
        forward_rates[d] = {
            f"1Y{_tenor_label(n)}": curve.forward_rate(FORWARD_START, FORWARD_START + n) for n in FORWARD_TENORS
        }

    key_yields_df = pd.DataFrame(key_yields).T.rename_axis("date")
    spot_df = pd.DataFrame(spot_rates).T.rename_axis("date")
    fwd_df = pd.DataFrame(forward_rates).T.rename_axis("date")

    yield_cov = log_returns(key_yields_df).cov()
    forward_cov = log_returns(fwd_df).cov()

    return AnalysisResult(
        bonds=bonds,
        ytm=ytm,
        maturities=ttm,
        ns_fits=ns_fits,
        residuals_bp=residuals,
        excluded=excluded,
        spot_curves=spot_curves,
        key_yields=key_yields_df,
        spot_rates=spot_df,
        forward_rates=fwd_df,
        yield_cov=yield_cov,
        forward_cov=forward_cov,
        yield_pca=pca(yield_cov),
        forward_pca=pca(forward_cov),
    )
