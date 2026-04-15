#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.analysis_rsp import calculate_and_plot_rsp


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _build_paths(base_dir: Path, study: str, run_folder: str) -> dict[str, Path]:
    study_dir = base_dir / study
    run_dir = study_dir / "runs" / run_folder

    return {
        "study_dir": study_dir,
        "run_dir": run_dir,
        "x_binary": run_dir / "X_binary.npy",
        "x_knockoff": run_dir / "X_knockoff_binary.npy",
        "y_clean": study_dir / "y_clean.npy",
        "metadata": run_dir / "run_metadata.json",
        "rsp_results": run_dir / "rsp_results.npy",
        "rsp_plot": run_dir / "rsp_plot.png",
    }


def recompute_rsp(
    base_dir: Path,
    study: str,
    run_folder: str,
    *,
    backup: bool,
) -> Path:
    paths = _build_paths(base_dir=base_dir, study=study, run_folder=run_folder)

    _required_file(paths["x_binary"])
    _required_file(paths["x_knockoff"])
    _required_file(paths["y_clean"])
    _required_file(paths["metadata"])
    _required_file(paths["rsp_results"])

    X_binary = np.load(paths["x_binary"])
    X_knockoff = np.load(paths["x_knockoff"])
    y = np.load(paths["y_clean"])

    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    params = metadata.get("parameters", {})
    target_fdr = float(params["target_fdr"])
    num_shuffles = int(params["num_shuffles"])
    random_seed = int(params["random_seed"])

    if backup:
        backup_path = paths["rsp_results"].with_name("rsp_results.pre_recompute.npy")
        shutil.copy2(paths["rsp_results"], backup_path)
        print(f"Backed up existing rsp_results to: {backup_path}")

    rng = np.random.default_rng(random_seed)
    rsp_results_new = asdict(
        calculate_and_plot_rsp(
            X=X_binary,
            X_tilde=X_knockoff,
            y=y,
            target_fdr=target_fdr,
            num_shuffles=num_shuffles,
            save_path=str(paths["rsp_plot"]),
            rng=rng,
        )
    )

    # Overwrite the faulty artifact in-place with the newly recomputed payload.
    np.save(paths["rsp_results"], rsp_results_new)

    print(f"Updated: {paths['rsp_results']}")
    print(f"New keys: {list(rsp_results_new.keys())}")
    print(
        f"RP={rsp_results_new['RP']} SP={rsp_results_new['SP']:.6f} "
        f"threshold={rsp_results_new['threshold']:.6f}"
    )
    print(f"feature_index_map entries: {len(rsp_results_new['feature_index_map'])}")

    return paths["rsp_results"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute rsp_results.npy from existing run artifacts (X_binary, "
            "X_knockoff_binary, y_clean, run_metadata) without rerunning the full pipeline."
        )
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing study folders (e.g. /path/to/projects/cMD_downloads)",
    )
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--run-folder", required=True, help="Run folder under <study>/runs")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not save rsp_results.pre_recompute.npy before overwriting rsp_results.npy",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    recompute_rsp(
        base_dir=Path(args.base_dir),
        study=args.study,
        run_folder=args.run_folder,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
