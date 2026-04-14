#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)
warnings.filterwarnings(
    "ignore",
    message="Features .* are constant.",
    category=UserWarning,
    module="sklearn.feature_selection._univariate_selection",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)


# Script-level experiment configuration.
# Toggle methods by adding/removing keys in ENABLED_METHODS.
CLASSIFIER_NAME = "lgbm"
METRIC_NAME = "roc_auc"
BH_TEST_NAME = "mannwhitney"

ENABLED_METHODS = [
    "random_raw",
    "bh_raw",
    "random_clean",
    "random_filtered",
    "knockoff_topk",
    "bacteria_constant",
    "gene_clustered_random",
]

METHOD_LABELS = {
    "random_raw": "Random K (Raw MTX)",
    "bh_raw": "BH Top-K (Raw MTX)",
    "random_clean": "Random K (X_clean)",
    "random_filtered": "Random K (X_filtered)",
    "knockoff_topk": "Knockoff Top-K",
    "bacteria_constant": "Bacteria Features (Constant)",
    "gene_clustered_random": "Gene-Clustered Random K",
}


def _log(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare knockoff-selected features against random and bacteria baselines.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--run-folder", required=True, help="Run folder under study/runs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--k-grid-points", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--k-end", type=int, default=5000)
    parser.add_argument(
        "--k-start",
        type=int,
        default=None,
        help="Optional K-start override. By default starts at bacteria feature count.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save K-grid table as CSV in run directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress checkpoint progress messages.",
    )
    return parser


def main() -> None:
    from microbiome_knockoffs.evaluation_classifier_comparison import (
        ClassifierComparisonConfig,
        plot_classifier_comparison,
        run_classifier_comparison,
    )

    args = build_parser().parse_args()
    verbose = not args.quiet
    script_start = perf_counter()

    _log("Starting knockoff classifier comparison.", verbose)
    _log(f"Study: {args.study}", verbose)
    _log(f"Run folder: {args.run_folder}", verbose)
    _log(f"Classifier: {CLASSIFIER_NAME} | Metric: {METRIC_NAME} | BH test: {BH_TEST_NAME}", verbose)

    config = ClassifierComparisonConfig(
        base_dir=Path(args.base_dir),
        study_name=args.study,
        run_folder=args.run_folder,
        random_state=args.random_state,
        test_size=args.test_size,
        k_grid_points=args.k_grid_points,
        random_trials=args.random_trials,
        k_end=args.k_end,
        k_start=args.k_start,
    )

    _log(f"Resolved run directory: {config.run_dir}", verbose)
    _log(
        "Running feature selection and classifier scoring across all enabled methods and K values...",
        verbose,
    )
    comparison_start = perf_counter()

    results = run_classifier_comparison(
        config,
        classifier_name=CLASSIFIER_NAME,
        metric_name=METRIC_NAME,
        enabled_methods=ENABLED_METHODS,
        method_labels=METHOD_LABELS,
        bh_test=BH_TEST_NAME,
    )
    comparison_elapsed = perf_counter() - comparison_start
    _log(
        f"Comparison complete in {comparison_elapsed:.1f}s. Generated {len(results)} rows.",
        verbose,
    )

    plot_path = config.run_dir / f"classifier_comparison_{config.study_name}.png"
    _log(f"Generating summary plot: {plot_path}", verbose)
    plot_classifier_comparison(
        results=results,
        study_name=config.study_name,
        save_path=plot_path,
        metric_name=METRIC_NAME,
    )
    _log("Plot generation complete.", verbose)

    pivot = (
        results.pivot_table(index="K", columns="method_label", values="score", aggfunc="mean")
        .sort_index()
    )
    print(pivot.to_string())
    print(f"Saved plot to: {plot_path}")

    if args.save_csv:
        csv_path = config.run_dir / f"classifier_comparison_{config.study_name}.csv"
        _log(f"Saving CSV table: {csv_path}", verbose)
        results.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

    _log(f"Run finished in {perf_counter() - script_start:.1f}s.", verbose)


if __name__ == "__main__":
    main()
