"""Command-line entry point: ``python -m yieldcurve``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_bonds, load_prices
from .pipeline import run_analysis
from .plotting import make_all_figures


def save_tables(res, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {b.isin: b.name for b in res.bonds}
    res.ytm.rename(columns=names).to_csv(out_dir / "ytm.csv", float_format="%.6f")
    res.residuals_bp.rename(columns=names).to_csv(out_dir / "nelson_siegel_residuals_bp.csv", float_format="%.2f")
    res.key_yields.to_csv(out_dir / "key_yields.csv", float_format="%.6f")
    res.spot_rates.to_csv(out_dir / "spot_rates.csv", float_format="%.6f")
    res.forward_rates.to_csv(out_dir / "forward_rates.csv", float_format="%.6f")
    res.yield_cov.to_csv(out_dir / "yield_logret_cov.csv", float_format="%.6e")
    res.forward_cov.to_csv(out_dir / "forward_logret_cov.csv", float_format="%.6e")
    for label, result in (("yield", res.yield_pca), ("forward", res.forward_pca)):
        result.summary().to_csv(out_dir / f"pca_{label}_summary.csv", float_format="%.6e")
        result.eigenvectors.to_csv(out_dir / f"pca_{label}_eigenvectors.csv", float_format="%.6f")


def print_report(res) -> None:
    pct = lambda s: (s * 100).round(3)  # noqa: E731
    pd.set_option("display.width", 120)
    print("Bonds:", len(res.bonds), "| trade dates:", len(res.ytm))
    if res.excluded:
        names = [b.name for b in res.bonds if b.isin in res.excluded]
        med = res.residuals_bp[res.excluded].abs().median().round(0).tolist()
        print(f"Excluded by quote screen: {', '.join(names)} (median |residual| bp: {med})")
    print("\nSpot rates (%):\n", pct(res.spot_rates))
    print("\n1Y forward rates (%):\n", pct(res.forward_rates))
    for label, result in (("Yield", res.yield_pca), ("Forward", res.forward_pca)):
        print(f"\n{label} PCA:\n", result.summary().round(4))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Government of Canada curves and run PCA.")
    parser.add_argument("--bonds", type=Path, default=Path("data/bonds.csv"))
    parser.add_argument("--prices", type=Path, default=Path("data/clean_prices.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--outlier-bp", type=float, default=75.0,
                        help="exclude bonds whose median |YTM - Nelson-Siegel fit| exceeds this (bp)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    res = run_analysis(load_bonds(args.bonds), load_prices(args.prices), args.outlier_bp)
    save_tables(res, args.out / "tables")
    if not args.no_plots:
        make_all_figures(res, args.out / "figures")
    print_report(res)


if __name__ == "__main__":
    main()
