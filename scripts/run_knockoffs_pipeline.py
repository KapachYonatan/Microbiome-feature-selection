#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the modular knockoff pipeline CLI."""

    parser = argparse.ArgumentParser(
        description="Run the Microbiome Knockoffs pipeline for one study.",
    )
    parser.add_argument("--study", required=True, help="Study name under base directory")
    parser.add_argument(
        "--base-dir",
        default="/home1/kapachy/Microbium/projects/cMD_downloads",
        help="Base directory containing study folders",
    )
    parser.add_argument("--sparsity-threshold", type=float, default=0.85)
    parser.add_argument("--k-neighbors", type=int, default=40)
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--num-shuffles", type=int, default=20)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--calibration-features", type=int, default=2000)
    parser.add_argument("--calibration-trials", type=int, default=50)
    parser.add_argument("--cluster-batch-size", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    from microbiome_knockoffs.contracts import RunConfig
    from microbiome_knockoffs.pipeline_orchestrator import run_pipeline

    parser = build_parser()
    args = parser.parse_args()

    config = RunConfig(
        study_name=args.study,
        base_dir=Path(args.base_dir),
        sparsity_threshold=args.sparsity_threshold,
        k_neighbors=args.k_neighbors,
        target_fdr=args.target_fdr,
        num_shuffles=args.num_shuffles,
        correlation_threshold=args.correlation_threshold,
        calibration_features=args.calibration_features,
        calibration_trials=args.calibration_trials,
        cluster_batch_size=args.cluster_batch_size,
        random_seed=args.seed,
    )

    artifacts = run_pipeline(config)

    print("Generated artifacts:")
    print(f"  metadata:         {artifacts.metadata_path}")
    print(f"  filtered matrix:  {artifacts.x_filtered_path}")
    print(f"  filtered genes:   {artifacts.genes_filtered_path}")
    print(f"  clusters:         {artifacts.clusters_path}")
    print(f"  binary matrix:    {artifacts.x_binary_path}")
    print(f"  knockoff matrix:  {artifacts.x_knockoff_path}")
    print(f"  covariance plot:  {artifacts.cov_plot_path}")
    print(f"  rsp results:      {artifacts.rsp_results_path}")
    print(f"  rsp plot:         {artifacts.rsp_plot_path}")


if __name__ == "__main__":
    main()
