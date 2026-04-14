#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_study_run_pairs(entries: list[str] | None) -> dict[str, str]:
    """Parse study:run_folder entries into a dictionary."""

    mapping: dict[str, str] = {}
    if not entries:
        return mapping

    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid --study-run value '{entry}'. Expected format study:run_folder")
        study, run = entry.split(":", 1)
        study = study.strip()
        run = run.strip()
        if not study or not run:
            raise ValueError(f"Invalid --study-run value '{entry}'.")
        mapping[study] = run
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate merged and RSP comparison visualizations.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument("--studies", nargs="+", required=True, help="Study names to include (up to 4 for grid)")
    parser.add_argument(
        "--study-run",
        action="append",
        help="Optional study to run-folder mapping, format study:run_folder. Repeat per study.",
    )
    parser.add_argument(
        "--mode",
        choices=["merge-cov", "rsp-grid", "all"],
        default="all",
        help="Visualization mode to run",
    )
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Optional BH cache JSON path. Defaults to {base_dir}/bh_rsp_fixed_fdr_cache.json",
    )
    parser.add_argument(
        "--cov-output",
        default=None,
        help="Optional merged covariance output path. Defaults to {base_dir}/combined_cov_preservation.png",
    )
    parser.add_argument(
        "--rsp-output",
        default=None,
        help="Optional RSP grid output path. Defaults to {base_dir}/grid_rsp_plot.png",
    )
    return parser


def main() -> None:
    from microbiome_knockoffs.visualization_plots import (
        merge_images_2x2,
        plot_rsp_grid,
        resolve_run_file,
    )

    args = build_parser().parse_args()

    base_dir = Path(args.base_dir)
    studies = args.studies
    run_map = parse_study_run_pairs(args.study_run)

    if len(studies) > 4:
        raise ValueError("At most 4 studies are supported in the current 2x2 grid layout")

    cache_file = Path(args.cache_file) if args.cache_file else (base_dir / "bh_rsp_fixed_fdr_cache.json")
    cov_output = Path(args.cov_output) if args.cov_output else (base_dir / "combined_cov_preservation.png")
    rsp_output = Path(args.rsp_output) if args.rsp_output else (base_dir / "grid_rsp_plot.png")

    if args.mode in {"merge-cov", "all"}:
        cov_paths = [
            resolve_run_file(base_dir, study, "cov_preservation.png", run_map.get(study))
            for study in studies
        ]
        if len(cov_paths) != 4:
            raise ValueError("merge-cov mode requires exactly 4 studies for 2x2 output")
        merged_path = merge_images_2x2(cov_paths, cov_output)
        print(f"Merged covariance image saved to: {merged_path}")

    if args.mode in {"rsp-grid", "all"}:
        rsp_path = plot_rsp_grid(
            base_dir=base_dir,
            studies=studies,
            run_map=run_map,
            cache_file=cache_file,
            save_path=rsp_output,
            target_fdr=args.target_fdr,
            n_shuffles=args.n_shuffles,
            seed=args.seed,
        )
        print(f"RSP grid plot saved to: {rsp_path}")


if __name__ == "__main__":
    main()
