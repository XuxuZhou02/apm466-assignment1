"""Covariance and principal component analysis of rate changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def log_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily log changes ``ln(r_{t+1} / r_t)`` for each column."""
    return np.log(levels / levels.shift(1)).dropna(how="all")


@dataclass(frozen=True)
class PCAResult:
    eigenvalues: pd.Series
    eigenvectors: pd.DataFrame  # columns are components PC1, PC2, ...

    @property
    def explained_variance_ratio(self) -> pd.Series:
        return self.eigenvalues / self.eigenvalues.sum()

    def summary(self) -> pd.DataFrame:
        ratio = self.explained_variance_ratio
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues,
                "explained_variance": ratio,
                "cumulative_explained": ratio.cumsum(),
            }
        )


def pca(cov: pd.DataFrame) -> PCAResult:
    """Eigen-decompose a covariance matrix, largest component first.

    Each eigenvector's sign is fixed so its loadings sum to a non-negative
    number, making PC1 read as an upward level shift.
    """
    values, vectors = np.linalg.eigh(cov.to_numpy())
    idx = np.argsort(values)[::-1]
    values, vectors = values[idx], vectors[:, idx]
    signs = np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    vectors = vectors * signs

    labels = [f"PC{k + 1}" for k in range(len(values))]
    return PCAResult(
        eigenvalues=pd.Series(values, index=labels, name="eigenvalue"),
        eigenvectors=pd.DataFrame(vectors, index=cov.index, columns=labels),
    )
