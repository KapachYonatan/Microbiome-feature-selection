from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse


NDArray = np.ndarray
SparseMatrix = sparse.csr_matrix


@dataclass(frozen=True)
class RunConfig:
    """Pipeline run configuration.

    Input structure:
    - study_name: dataset folder name under base_dir.
    - base_dir: root directory containing per-study folders.

    Output usage:
    - Provides all stage parameters and derived paths consumed by pipeline modules.
    """

    study_name: str
    base_dir: Path
    sparsity_threshold: float = 0.85
    k_neighbors: int = 40
    target_fdr: float = 0.05
    num_shuffles: int = 20
    correlation_threshold: float = 0.95
    calibration_features: int = 2000
    calibration_trials: int = 50
    cluster_batch_size: int = 64000
    random_seed: int = 42
    deterministic_mode: bool = False
    faiss_mode: Literal["hnsw", "flat"] = "hnsw"
    faiss_threads: int = 1
    filter_n_jobs: int | None = None
    use_optuna_tuning: bool = True
    classifier_params: dict[str, Any] | None = None
    regressor_params: dict[str, Any] | None = None

    @property
    def study_dir(self) -> Path:
        return self.base_dir / self.study_name

    @property
    def effective_filter_n_jobs(self) -> int:
        if self.filter_n_jobs is not None:
            return int(self.filter_n_jobs)
        return 1 if self.deterministic_mode else -1


@dataclass(frozen=True)
class StudyData:
    """Loaded study arrays.

    Input source:
    - X_clean.npz, y_clean.npy, genes_clean.txt from a study directory.

    Output structure:
    - X_sparse: csr_matrix with shape (n_samples, n_features).
    - y: ndarray with shape (n_samples,).
    - feature_names: ndarray[str] with shape (n_features,).
    """

    X_sparse: SparseMatrix
    y: NDArray
    feature_names: NDArray


@dataclass(frozen=True)
class FilteringArtifacts:
    """Feature filtering outputs.

    Output structure:
    - X_filtered: csr_matrix with shape (n_samples, n_filtered_features).
    - feature_names_filtered: ndarray[str] with shape (n_filtered_features,).
    - leaders: ndarray[int] with indices in original feature space.
    - clusters: map[int, ndarray[int]] leader to member indices.
    """

    X_filtered: SparseMatrix
    feature_names_filtered: NDArray
    leaders: NDArray
    clusters: dict[int, NDArray]


@dataclass(frozen=True)
class CalibrationBatch:
    """Prepared grouped calibration arrays for tuning.

    Output structure:
    - X_clf, y_clf, groups_clf for support classifier tuning.
    - X_reg, y_reg, groups_reg for value regressor tuning. Can be None when unavailable.
    """

    X_clf: NDArray
    y_clf: NDArray
    groups_clf: NDArray
    X_reg: NDArray | None
    y_reg: NDArray | None
    groups_reg: NDArray | None


@dataclass(frozen=True)
class TuningResult:
    """Tuning outputs for distribution model components."""

    classifier_params: dict[str, Any]
    regressor_params: dict[str, Any]
    classifier_score: float
    regressor_score: float


@dataclass
class KnockoffOutputs:
    """Generated knockoff arrays and runtime logs.

    Output structure:
    - X_transformed: ndarray with shape (n_samples, n_features).
    - X_knockoff: ndarray with shape (n_samples, n_features).
    - logs: list of fallback/runtime events.
    """

    X_transformed: NDArray
    X_knockoff: NDArray
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RSPResult:
    """RSP analysis outputs."""

    W_real: NDArray
    selected_indices: NDArray
    rsp: NDArray
    beta_values: NDArray
    RP: int
    SP: float
    threshold: float


@dataclass(frozen=True)
class PipelineArtifacts:
    """Top-level file outputs from a pipeline run."""

    run_dir: Path
    metadata_path: Path
    x_filtered_path: Path
    genes_filtered_path: Path
    clusters_path: Path
    x_binary_path: Path
    x_knockoff_path: Path
    cov_plot_path: Path
    rsp_results_path: Path
    rsp_plot_path: Path


def validate_run_config(config: RunConfig) -> None:
    """Validate deterministic/tuning configuration combinations."""

    if config.faiss_mode not in {"hnsw", "flat"}:
        raise ValueError(f"Invalid faiss_mode: {config.faiss_mode}")

    if int(config.faiss_threads) < 1:
        raise ValueError(f"faiss_threads must be >= 1, got {config.faiss_threads}")

    if int(config.k_neighbors) < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {config.k_neighbors}")

    if config.classifier_params is not None and not isinstance(config.classifier_params, dict):
        raise TypeError("classifier_params must be a dict or None")

    if config.regressor_params is not None and not isinstance(config.regressor_params, dict):
        raise TypeError("regressor_params must be a dict or None")

    if config.filter_n_jobs is not None and int(config.filter_n_jobs) == 0:
        raise ValueError("filter_n_jobs cannot be 0")
