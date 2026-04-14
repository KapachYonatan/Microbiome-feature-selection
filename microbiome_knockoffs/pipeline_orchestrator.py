from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_covariance import plot_cov_preservation
from .analysis_rsp import calculate_and_plot_rsp
from .contracts import PipelineArtifacts, RunConfig
from .filtering_star import build_named_clusters, run_feature_filtering
from .io_data import (
    build_pipeline_artifacts,
    create_run_directory,
    finalize_run_metadata,
    load_study_data,
    save_filtering_outputs,
    save_knockoff_outputs,
    save_rsp_outputs,
    save_run_metadata,
)
from .knockoffs import (
    BinaryKnockoffGenerator,
    FaissFlatIPIndex,
    FaissHNSWIndex,
    HurdleLGBMDistribution,
    NoOpTuner,
    OptunaLGBMTuner,
)
from .logging_utils import log_checkpoint


def _configure_runtime(config: RunConfig) -> None:
    """Apply deterministic thread settings where requested."""

    if config.deterministic_mode:
        os.environ["OMP_NUM_THREADS"] = str(config.faiss_threads)
        os.environ["MKL_NUM_THREADS"] = str(config.faiss_threads)

    try:
        import faiss

        faiss.omp_set_num_threads(int(config.faiss_threads))
    except Exception:
        # Keep pipeline runnable even when faiss runtime controls are unavailable.
        pass


def run_pipeline(config: RunConfig) -> PipelineArtifacts:
    """Run the full knockoff pipeline for one study.

    Input:
    - config: RunConfig with all run-time parameters.

    Output:
    - PipelineArtifacts containing run directory and generated file paths.
    """

    np.random.seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)
    _configure_runtime(config)

    log_checkpoint(f"Microbiome Knockoffs Pipeline - Study: {config.study_name}", section=True)
    print(f"Study directory: {config.study_dir}\n")

    run_dir = create_run_directory(config)

    tuning_backend = "OptunaLGBMTuner" if config.use_optuna_tuning else "NoOpTuner"

    metadata = save_run_metadata(
        run_dir=run_dir,
        config=config,
        model_type="BinaryKnockoffGenerator",
        distribution_learner="HurdleLGBMDistribution",
        tuning_backend=tuning_backend,
        deterministic_mode=config.deterministic_mode,
        faiss_mode=config.faiss_mode,
        faiss_threads=config.faiss_threads,
        use_optuna_tuning=config.use_optuna_tuning,
        classifier_params=config.classifier_params,
        regressor_params=config.regressor_params,
    )

    print("Determinism profile:")
    print(f"  deterministic_mode: {config.deterministic_mode}")
    print(f"  faiss_mode: {config.faiss_mode}")
    print(f"  faiss_threads: {config.faiss_threads}")
    print(f"  filter_n_jobs: {config.effective_filter_n_jobs}")
    print(f"  use_optuna_tuning: {config.use_optuna_tuning}\n")

    log_checkpoint("Step 1: Loading data", section=True)
    study = load_study_data(config)
    print(f"Data shape: {study.X_sparse.shape}")
    print(f"Label shape: {study.y.shape}")
    print(f"Features shape: {study.feature_names.shape}\n")

    log_checkpoint("Step 2: Feature filtering", section=True)
    filtered = run_feature_filtering(
        study.X_sparse,
        study.feature_names,
        correlation_threshold=config.correlation_threshold,
        batch_size=config.cluster_batch_size,
        n_jobs=config.effective_filter_n_jobs,
    )
    named_clusters = build_named_clusters(filtered.clusters, study.feature_names)
    x_filtered_path, genes_filtered_path, clusters_path = save_filtering_outputs(
        run_dir,
        filtered.X_filtered,
        filtered.feature_names_filtered,
        named_clusters,
    )

    log_checkpoint("Step 3: Knockoff generation", section=True)
    if config.faiss_mode == "flat":
        neighbor_index = FaissFlatIPIndex(vector_dim=filtered.X_filtered.shape[0])
    else:
        neighbor_index = FaissHNSWIndex(vector_dim=filtered.X_filtered.shape[0])

    if config.use_optuna_tuning:
        tuner = OptunaLGBMTuner(
            default_classifier_params=config.classifier_params,
            default_regressor_params=config.regressor_params,
        )
    else:
        tuner = NoOpTuner(
            classifier_params=config.classifier_params,
            regressor_params=config.regressor_params,
        )

    generator = BinaryKnockoffGenerator(
        X=filtered.X_filtered.toarray(),
        sparsity_threshold=config.sparsity_threshold,
        k_neighbors=config.k_neighbors,
        random_seed=config.random_seed,
        deterministic_mode=config.deterministic_mode,
        neighbor_index=neighbor_index,
        distribution_learner=HurdleLGBMDistribution(),
        tuner=tuner,
    )
    knockoff_outputs = generator.generate(
        n_calibration=config.calibration_features,
        n_trials=config.calibration_trials,
        tune=config.use_optuna_tuning,
    )

    print(f"Generation complete. Total fallback events: {len(knockoff_outputs.logs)}")
    if knockoff_outputs.logs:
        log_df = pd.DataFrame(knockoff_outputs.logs)
        print("\nLog summary by status:")
        print(log_df.groupby(["step", "status"]).size())

    x_binary_path, x_knockoff_path = save_knockoff_outputs(
        run_dir,
        knockoff_outputs.X_transformed,
        knockoff_outputs.X_knockoff,
    )

    log_checkpoint("Step 4: Covariance preservation", section=True)
    cov_plot_path = run_dir / "cov_preservation.png"
    preservation_score = plot_cov_preservation(
        knockoff_outputs.X_transformed,
        knockoff_outputs.X_knockoff,
        save_path=str(cov_plot_path),
        rng=rng,
    )
    print(f"Preservation score: {preservation_score:.4f}\n")

    log_checkpoint("Step 5: RSP analysis", section=True)
    rsp_plot_path = run_dir / "rsp_plot.png"
    rsp_result = calculate_and_plot_rsp(
        knockoff_outputs.X_transformed,
        knockoff_outputs.X_knockoff,
        study.y,
        target_fdr=config.target_fdr,
        num_shuffles=config.num_shuffles,
        save_path=str(rsp_plot_path),
        rng=rng,
    )
    rsp_results_path = save_rsp_outputs(run_dir, asdict(rsp_result))

    metadata_path = finalize_run_metadata(run_dir, metadata, status="completed")

    log_checkpoint("Pipeline complete", section=True)
    print(f"Run directory: {run_dir}\n")

    return build_pipeline_artifacts(
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
