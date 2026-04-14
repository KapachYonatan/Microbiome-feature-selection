#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.contracts import RunConfig


@dataclass(frozen=True)
class RunSummary:
    """Compact run summary used for flow-compatibility checks.

    Output structure:
    - stores artifact shapes and key summary statistics for one run directory.
    """

    run_dir: str
    x_filtered_shape: tuple[int, int]
    x_binary_shape: tuple[int, int]
    x_knockoff_shape: tuple[int, int]
    n_genes_filtered: int
    rp: int
    sp: float
    threshold: float
    has_cov_plot: bool
    has_rsp_plot: bool


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_run_summary(run_dir: Path) -> RunSummary:
    """Load a run folder and return structural summary metrics.

    Required files in run_dir:
    - X_filtered.npz
    - genes_filtered.txt
    - X_binary.npy
    - X_knockoff_binary.npy
    - rsp_results.npy
    """

    required_files = [
        run_dir / "X_filtered.npz",
        run_dir / "genes_filtered.txt",
        run_dir / "X_binary.npy",
        run_dir / "X_knockoff_binary.npy",
        run_dir / "rsp_results.npy",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    X_filtered = sparse.load_npz(run_dir / "X_filtered.npz")
    genes_filtered = np.loadtxt(run_dir / "genes_filtered.txt", dtype=str)
    X_binary = np.load(run_dir / "X_binary.npy")
    X_knockoff = np.load(run_dir / "X_knockoff_binary.npy")
    rsp_result = np.load(run_dir / "rsp_results.npy", allow_pickle=True).item()

    if np.ndim(genes_filtered) == 0:
        genes_filtered = np.array([str(genes_filtered)])

    rp_val = rsp_result.get("RP")
    if rp_val is None:
        selected = np.asarray(rsp_result.get("selected_indices", []), dtype=int)
        rp_val = int(selected.size)

    return RunSummary(
        run_dir=str(run_dir),
        x_filtered_shape=(int(X_filtered.shape[0]), int(X_filtered.shape[1])),
        x_binary_shape=(int(X_binary.shape[0]), int(X_binary.shape[1])),
        x_knockoff_shape=(int(X_knockoff.shape[0]), int(X_knockoff.shape[1])),
        n_genes_filtered=int(len(genes_filtered)),
        rp=int(rp_val),
        sp=_safe_float(rsp_result.get("SP")),
        threshold=_safe_float(rsp_result.get("threshold")),
        has_cov_plot=(run_dir / "cov_preservation.png").exists(),
        has_rsp_plot=(run_dir / "rsp_plot.png").exists(),
    )


def compare_summaries(reference: RunSummary, candidate: RunSummary) -> list[dict[str, object]]:
    """Compare key structure/flow outputs between legacy and new runs."""

    checks: list[dict[str, object]] = []

    def add_check(name: str, ref_value: object, new_value: object, passed: bool) -> None:
        checks.append(
            {
                "check": name,
                "reference": ref_value,
                "candidate": new_value,
                "passed": bool(passed),
            }
        )

    add_check(
        "X_filtered sample count",
        reference.x_filtered_shape[0],
        candidate.x_filtered_shape[0],
        reference.x_filtered_shape[0] == candidate.x_filtered_shape[0],
    )
    add_check(
        "X_filtered feature count",
        reference.x_filtered_shape[1],
        candidate.x_filtered_shape[1],
        reference.x_filtered_shape[1] == candidate.x_filtered_shape[1],
    )
    add_check(
        "genes_filtered count",
        reference.n_genes_filtered,
        candidate.n_genes_filtered,
        reference.n_genes_filtered == candidate.n_genes_filtered,
    )
    add_check(
        "X_binary shape",
        reference.x_binary_shape,
        candidate.x_binary_shape,
        reference.x_binary_shape == candidate.x_binary_shape,
    )
    add_check(
        "X_knockoff shape",
        reference.x_knockoff_shape,
        candidate.x_knockoff_shape,
        reference.x_knockoff_shape == candidate.x_knockoff_shape,
    )
    add_check(
        "Covariance plot exists",
        reference.has_cov_plot,
        candidate.has_cov_plot,
        candidate.has_cov_plot,
    )
    add_check(
        "RSP plot exists",
        reference.has_rsp_plot,
        candidate.has_rsp_plot,
        candidate.has_rsp_plot,
    )

    return checks


def print_report(reference: RunSummary, candidate: RunSummary, checks: list[dict[str, object]]) -> None:
    """Print comparison results to stdout."""

    print("\n=== Legacy Flow Compatibility Report ===")
    print(f"Reference run: {reference.run_dir}")
    print(f"Candidate run: {candidate.run_dir}\n")

    for row in checks:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{status}] {row['check']}: "
            f"reference={row['reference']} candidate={row['candidate']}"
        )

    failures = [row for row in checks if not row["passed"]]
    print(f"\nTotal checks: {len(checks)}")
    print(f"Passed: {len(checks) - len(failures)}")
    print(f"Failed: {len(failures)}")



def build_parser() -> argparse.ArgumentParser:
    """Build parser for end-to-end legacy flow compatibility testing."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the new modular pipeline with legacy parameters and compare artifacts "
            "against a reference run directory."
        )
    )

    parser.add_argument(
        "--base-dir",
        default="/home1/kapachy/Microbium/projects/cMD_downloads",
        help="Base directory that contains study folders",
    )
    parser.add_argument(
        "--study",
        default="WirbelJ_2018",
        help="Study name",
    )
    parser.add_argument(
        "--reference-run-dir",
        default=(
            "/home1/kapachy/Microbium/projects/cMD_downloads/WirbelJ_2018/runs/"
            "WirbelJ_2018_sparse085_k40_20260324_153038"
        ),
        help="Legacy run folder to compare against",
    )

    # Legacy-equivalent defaults from combined_knockoffs_flow.py
    parser.add_argument("--sparsity-threshold", type=float, default=0.85)
    parser.add_argument("--k-neighbors", type=int, default=40)
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--num-shuffles", type=int, default=20)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--calibration-features", type=int, default=2000)
    parser.add_argument("--calibration-trials", type=int, default=50)
    parser.add_argument("--cluster-batch-size", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip pipeline run and only compare an existing candidate run",
    )
    parser.add_argument(
        "--candidate-run-dir",
        default=None,
        help="Candidate run directory used with --compare-only",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path (default: candidate_run_dir/flow_compat_report.json)",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    reference_run_dir = Path(args.reference_run_dir)
    if not reference_run_dir.is_dir():
        raise FileNotFoundError(f"Reference run directory does not exist: {reference_run_dir}")

    candidate_run_dir: Path

    if args.compare_only:
        if not args.candidate_run_dir:
            raise ValueError("--candidate-run-dir is required when using --compare-only")
        candidate_run_dir = Path(args.candidate_run_dir)
        if not candidate_run_dir.is_dir():
            raise FileNotFoundError(f"Candidate run directory does not exist: {candidate_run_dir}")
    else:
        from microbiome_knockoffs.pipeline_orchestrator import run_pipeline

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
        candidate_run_dir = artifacts.run_dir

    reference_summary = load_run_summary(reference_run_dir)
    candidate_summary = load_run_summary(candidate_run_dir)

    checks = compare_summaries(reference_summary, candidate_summary)
    print_report(reference_summary, candidate_summary, checks)

    report = {
        "reference": asdict(reference_summary),
        "candidate": asdict(candidate_summary),
        "checks": checks,
    }

    report_path = Path(args.report_path) if args.report_path else (candidate_run_dir / "flow_compat_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved JSON report to: {report_path}")


if __name__ == "__main__":
    main()
