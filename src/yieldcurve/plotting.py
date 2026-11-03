"""Figures for the README / report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .pca import PCAResult  # noqa: E402
from .pipeline import AnalysisResult  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
EXCLUDED = "#9a9993"
# Ordinal blue ramp (steps 250 -> 700): earlier dates lighter, later dates darker.
DATE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COMPONENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # PC1, PC2, PC3

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2,
    "font.size": 10,
})


def _date_colors(n: int) -> list[str]:
    idx = np.linspace(0, len(DATE_RAMP) - 1, n).round().astype(int)
    return [DATE_RAMP[i] for i in idx]


def _date_legend(ax) -> None:
    ax.legend(title="Trade date", title_fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))


def plot_yield_curves(res: AnalysisResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    used = [b.isin for b in res.included]
    for color, d in zip(_date_colors(len(res.ytm)), res.ytm.index):
        t = res.maturities.loc[d, used].to_numpy()
        y = res.ytm.loc[d, used].to_numpy() * 100
        order = np.argsort(t)
        ax.plot(t[order], y[order], "-o", color=color, markersize=4, label=d.strftime("%b %d"))
    for isin in res.excluded:
        ax.scatter(res.maturities[isin], res.ytm[isin] * 100, marker="x", color=EXCLUDED, s=30, zorder=3)
        name = next(b.name for b in res.bonds if b.isin == isin)
        ax.annotate(f"{name}\n(excluded quote)", (res.maturities[isin].iloc[-1], res.ytm[isin].iloc[-1] * 100),
                    xytext=(8, -4), textcoords="offset points", fontsize=8, color=TEXT_MUTED, va="top")
    ax.set(xlabel="Time to maturity (years)", ylabel="Yield to maturity (%)",
           title="Government of Canada yield curves, Jan 5–16 2026")
    _date_legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spot_curves(res: AnalysisResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for color, (d, curve) in zip(_date_colors(len(res.spot_curves)), res.spot_curves.items()):
        grid = np.linspace(curve.times[0], curve.times[-1], 200)
        ax.plot(grid, curve.zero_rate(grid) * 100, color=color, label=d.strftime("%b %d"))
        ax.plot(curve.times, curve.rates * 100, "o", color=color, markersize=3)
    ax.set(xlabel="Maturity (years)", ylabel="Zero rate, cont. compounded (%)",
           title="Bootstrapped spot (zero-coupon) curves")
    _date_legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forward_curves(res: AnalysisResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(res.forward_rates.columns)
    x = np.arange(1, len(labels) + 1)
    for color, d in zip(_date_colors(len(res.forward_rates)), res.forward_rates.index):
        ax.plot(x, res.forward_rates.loc[d] * 100, "-o", color=color, markersize=4, label=d.strftime("%b %d"))
    ax.set_xticks(x, labels)
    ax.set(xlabel="Forward (start 1Y, tenor nY)", ylabel="Forward rate, cont. compounded (%)",
           title="1-year forward curves")
    _date_legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca(result: PCAResult, title: str, path: Path, n_components: int = 3) -> None:
    fig, (ax_var, ax_load) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})

    ratio = result.explained_variance_ratio * 100
    ax_var.bar(ratio.index, ratio.values, color=DATE_RAMP[4], width=0.6)
    for i, v in enumerate(ratio.values):
        ax_var.annotate(f"{v:.1f}%", (i, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_MUTED)
    ax_var.set(ylabel="Variance explained (%)", ylim=(0, 105), title=f"{title}: variance explained")
    ax_var.grid(axis="x", visible=False)

    vecs = result.eigenvectors.iloc[:, :n_components]
    x = np.arange(len(vecs.index))
    for color, pc in zip(COMPONENT_COLORS, vecs.columns):
        ax_load.plot(x, vecs[pc], "-o", color=color, markersize=5,
                     label=f"{pc} ({result.explained_variance_ratio[pc] * 100:.0f}%)")
    ax_load.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax_load.set_xticks(x, vecs.index)
    ax_load.set(ylabel="Loading", title=f"{title}: eigenvector loadings")
    ax_load.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all_figures(res: AnalysisResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "yield_curves.png": lambda p: plot_yield_curves(res, p),
        "spot_curves.png": lambda p: plot_spot_curves(res, p),
        "forward_curves.png": lambda p: plot_forward_curves(res, p),
        "pca_yields.png": lambda p: plot_pca(res.yield_pca, "Yields (1Y–5Y)", p),
        "pca_forwards.png": lambda p: plot_pca(res.forward_pca, "1Y forwards", p),
    }
    written = []
    for name, fn in paths.items():
        fn(out_dir / name)
        written.append(out_dir / name)
    return written
