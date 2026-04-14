from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse

from .contracts import PipelineArtifacts, RunConfig, StudyData


def create_run_directory(config: RunConfig) -> Path:
    """Create run output directory and return its path.

    Input:
    - config.study_dir points to a valid study folder.

    Output:
    - Path to {study_dir}/runs/{study}_{timestamp}.
    """

    runs_dir = config.study_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{config.study_name}_{timestamp}"

    run_dir = runs_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    config: RunConfig,
    model_type: str,
    distribution_learner: str,
    tuning_backend: str,
) -> dict:
    """Persist run metadata JSON and return the in-memory metadata object."""

    metadata = {
        "run_info": {
            "timestamp_start": datetime.now().isoformat(),
            "study_name": config.study_name,
        },
        "parameters": {
            "sparsity_threshold": config.sparsity_threshold,
            "k_neighbors": config.k_neighbors,
            "correlation_threshold": config.correlation_threshold,
            "target_fdr": config.target_fdr,
            "num_shuffles": config.num_shuffles,
            "calibration": {
                "n_calibration": config.calibration_features,
                "n_trials": config.calibration_trials,
            },
            "model_type": model_type,
            "distribution_learner": distribution_learner,
            "tuning_backend": tuning_backend,
            "random_seed": config.random_seed,
        },
    }

    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return metadata


def finalize_run_metadata(run_dir: Path, metadata: dict, status: str = "completed") -> Path:
    """Update run metadata with end timestamp and final status.

    Output:
    - Path to the saved metadata JSON.
    """

    metadata["run_info"]["timestamp_end"] = datetime.now().isoformat()
    metadata["run_info"]["status"] = status

    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return metadata_path


def load_study_data(config: RunConfig) -> StudyData:
    """Load study input arrays from disk.

    Input files:
    - X_clean.npz, y_clean.npy, genes_clean.txt in config.study_dir.

    Output:
    - StudyData with aligned sample/feature dimensions.
    """

    if not config.study_dir.is_dir():
        raise FileNotFoundError(f"Study directory does not exist: {config.study_dir}")

    X_sparse = sparse.load_npz(config.study_dir / "X_clean.npz").tocsr()
    y = np.load(config.study_dir / "y_clean.npy")
    feature_names = np.loadtxt(config.study_dir / "genes_clean.txt", dtype=str)

    if X_sparse.shape[0] != y.shape[0]:
        raise ValueError(
            "Sample mismatch between X_clean and y_clean: "
            f"{X_sparse.shape[0]} != {y.shape[0]}"
        )

    if X_sparse.shape[1] != feature_names.shape[0]:
        raise ValueError(
            "Feature mismatch between X_clean and genes_clean: "
            f"{X_sparse.shape[1]} != {feature_names.shape[0]}"
        )

    return StudyData(X_sparse=X_sparse, y=y, feature_names=feature_names)


def save_filtering_outputs(
    run_dir: Path,
    X_filtered: sparse.csr_matrix,
    feature_names_filtered: np.ndarray,
    clusters: dict[str, list[str]],
) -> tuple[Path, Path, Path]:
    """Save feature filtering artifacts.

    Output paths:
    - X_filtered.npz, genes_filtered.txt, gene_filtered_clusters.json.
    """

    x_filtered_path = run_dir / "X_filtered.npz"
    genes_filtered_path = run_dir / "genes_filtered.txt"
    clusters_path = run_dir / "gene_filtered_clusters.json"

    sparse.save_npz(x_filtered_path, X_filtered)
    np.savetxt(genes_filtered_path, feature_names_filtered, fmt="%s")
    with clusters_path.open("w", encoding="utf-8") as handle:
        json.dump(clusters, handle, indent=4)

    return x_filtered_path, genes_filtered_path, clusters_path


def save_knockoff_outputs(
    run_dir: Path,
    X_binary: np.ndarray,
    X_knockoff: np.ndarray,
) -> tuple[Path, Path]:
    """Save binary transformed data and generated knockoffs."""

    x_binary_path = run_dir / "X_binary.npy"
    x_knockoff_path = run_dir / "X_knockoff_binary.npy"

    np.save(x_binary_path, X_binary)
    np.save(x_knockoff_path, X_knockoff)

    return x_binary_path, x_knockoff_path


def save_rsp_outputs(run_dir: Path, rsp_result: dict) -> Path:
    """Save RSP dictionary as a numpy object array file."""

    rsp_results_path = run_dir / "rsp_results.npy"
    np.save(rsp_results_path, rsp_result)
    return rsp_results_path


def build_pipeline_artifacts(
    run_dir: Path,
    metadata_path: Path,
    x_filtered_path: Path,
    genes_filtered_path: Path,
    clusters_path: Path,
    x_binary_path: Path,
    x_knockoff_path: Path,
    cov_plot_path: Path,
    rsp_results_path: Path,
    rsp_plot_path: Path,
) -> PipelineArtifacts:
    """Construct a typed pipeline artifact summary object."""

    return PipelineArtifacts(
        run_dir=run_dir,
        metadata_path=metadata_path,
        x_filtered_path=x_filtered_path,
        genes_filtered_path=genes_filtered_path,
        clusters_path=clusters_path,
        x_binary_path=x_binary_path,
        x_knockoff_path=x_knockoff_path,
        cov_plot_path=cov_plot_path,
        rsp_results_path=rsp_results_path,
        rsp_plot_path=rsp_plot_path,
    )
