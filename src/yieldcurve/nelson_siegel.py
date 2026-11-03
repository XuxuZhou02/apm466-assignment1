"""Robust Nelson-Siegel fit, used to smooth yields and screen bad quotes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

DEFAULT_TAUS = np.linspace(0.25, 5.0, 39)


def _loadings(t: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(t, dtype=float) / tau
    slope = (1.0 - np.exp(-x)) / x
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(x), slope, curvature])


@dataclass(frozen=True)
class NelsonSiegel:
    beta0: float  # long-run level
    beta1: float  # short-end (slope) component
    beta2: float  # medium-term hump (curvature)
    tau: float  # decay time in years

    def __call__(self, t: float | np.ndarray) -> np.ndarray:
        return _loadings(np.atleast_1d(t), self.tau) @ np.array([self.beta0, self.beta1, self.beta2])

    @classmethod
    def fit(
        cls,
        maturities: np.ndarray,
        yields: np.ndarray,
        taus: np.ndarray = DEFAULT_TAUS,
        f_scale: float = 0.002,
    ) -> "NelsonSiegel":
        """Fit by robust (soft-L1) least squares over a grid of decay times.

        For fixed ``tau`` the model is linear in the betas, so each grid point
        is a small, well-conditioned problem. The soft-L1 loss (``f_scale`` is
        roughly the residual size, in yield units, where it starts to
        down-weight points) stops a single bad quote from bending the curve.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)
        best = None
        for tau in taus:
            design = _loadings(maturities, tau)
            x0 = np.linalg.lstsq(design, yields, rcond=None)[0]
            res = least_squares(lambda b: design @ b - yields, x0, loss="soft_l1", f_scale=f_scale)
            if best is None or res.cost < best[0]:
                best = (res.cost, res.x, tau)
        _, betas, tau = best
        return cls(*betas, tau=tau)
