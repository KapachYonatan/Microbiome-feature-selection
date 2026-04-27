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
CLASSIFIER_NAME = "lgbm"
BH_TEST_NAME = "mannwhitney"
DEFAULT_OUTPUT_BASE_DIR = Path("/home/kapachy/Microbium/projects/cMD_downloads")

ENABLED_METHODS = [
    "bh_raw",
    "bh_clean",
    "bh_filtered",
    "knockoff_topk",
    "bacteria_bh",
    "gene_clustered_bh",
]

METHOD_LABELS = {
    "bh_raw": "BH Top-K (Raw MTX)",
    "bh_clean": "BH Top-K (X_clean)",
    "bh_filtered": "BH Top-K (X_filtered)",
    "knockoff_topk": "Knockoff Top-K",
    "bacteria_bh": "BH Top-K (Bacteria)",
    "gene_clustered_bh": "BH Top-K (Gene-Clustered)",
}


def _log(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare knockoff-selected features against BH baselines.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument(
        "--output-base-dir",
        default=str(DEFAULT_OUTPUT_BASE_DIR),
        help="Base directory for saved outputs",
    )
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--run-folder", required=True, help="Run folder under study/runs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--k-min", type=int, default=40, help="Minimum K value (default: 40)")
    parser.add_argument("--k-max", type=int, default=1000, help="Maximum K value (default: 1000)")
    parser.add_argument("--k-grid-points", type=int, default=10, help="Number of K grid points (default: 10)")
    parser.add_argument("--random-trials", type=int, default=10, help="Number of random trials (default: 10)")
    parser.add_argument("--save-csv", action="store_true", help="Also save results as CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
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

    _log(f"Study: {args.study} | Run folder: {args.run_folder}", verbose)
    _log(f"Classifier: {CLASSIFIER_NAME} | BH test: {BH_TEST_NAME} | K: [{args.k_min}, {args.k_max}] | grid={args.k_grid_points} | trials={args.random_trials}", verbose)

    config = ClassifierComparisonConfig(
        base_dir=Path(args.base_dir),
        study_name=args.study,
        run_folder=args.run_folder,
        random_state=args.random_state,
        test_size=args.test_size,
        k_min=args.k_min,
        k_max=args.k_max,
        k_grid_points=args.k_grid_points,
        random_trials=args.random_trials,
    )

    output_dir = Path(args.output_base_dir) / args.study
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output directory: {output_dir}", verbose)

    results = run_classifier_comparison(
        config,
        classifier_name=CLASSIFIER_NAME,
        enabled_methods=ENABLED_METHODS,
        method_labels=METHOD_LABELS,
        bh_test=BH_TEST_NAME,
    )
    _log(f"Comparison complete. {len(results)} result rows.", verbose)

    file_suffix = f"{config.study_name}_{config.run_folder}"
    plot_path = output_dir / f"classifier_comparison_{file_suffix}.png"
    _log(f"Saving plot: {plot_path}", verbose)
    plot_classifier_comparison(results=results, study_name=config.study_name, save_path=plot_path)
    print(f"Saved plot to: {plot_path}")

    for metric in results["metric_name"].unique():
        pivot = (
            results[results["metric_name"] == metric]
            .pivot_table(index="K", columns="method_label", values="score", aggfunc="mean")
            .sort_index()
        )
        print(f"\n=== {metric} ===")
        print(pivot.to_string())

    if args.save_csv:
        csv_path = output_dir / f"classifier_comparison_{file_suffix}.csv"
        results.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

    _log(f"Finished in {perf_counter() - script_start:.1f}s.", verbose)


if __name__ == "__main__":
    main()
