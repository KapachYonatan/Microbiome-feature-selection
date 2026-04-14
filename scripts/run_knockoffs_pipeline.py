#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the modular knockoff pipeline CLI."""

    parser = argparse.ArgumentParser(
        description="Run the Microbiome Knockoffs pipeline for one study.",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config file path")
    parser.add_argument("--study", default=None, help="Study name under base directory")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory containing study folders",
    )
    parser.add_argument("--sparsity-threshold", type=float, default=None)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--target-fdr", type=float, default=None)
    parser.add_argument("--num-shuffles", type=int, default=None)
    parser.add_argument("--correlation-threshold", type=float, default=None)
    parser.add_argument("--calibration-features", type=int, default=None)
    parser.add_argument("--calibration-trials", type=int, default=None)
    parser.add_argument("--cluster-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--deterministic-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable deterministic execution profile",
    )
    parser.add_argument("--faiss-mode", choices=["hnsw", "flat"], default=None)
    parser.add_argument("--faiss-threads", type=int, default=None)
    parser.add_argument("--filter-n-jobs", type=int, default=None)

    parser.add_argument(
        "--use-optuna-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Optuna tuning",
    )
    parser.add_argument(
        "--no-optuna-tuning",
        action="store_true",
        default=None,
        help="Convenience alias to disable Optuna tuning",
    )
    parser.add_argument(
        "--classifier-params-json",
        default=None,
        help="JSON dict for classifier params",
    )
    parser.add_argument(
        "--regressor-params-json",
        default=None,
        help="JSON dict for regressor params",
    )
    return parser


def _load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping/object")
    return data


def _read_section(data: dict, section: str) -> dict:
    section_data = data.get(section, {})
    return section_data if isinstance(section_data, dict) else {}


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _parse_json_dict(raw: str | None, arg_name: str) -> dict | None:
    if raw is None:
        return None

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object")
    return parsed


def main() -> None:
    from microbiome_knockoffs.contracts import RunConfig, validate_run_config
    from microbiome_knockoffs.pipeline_orchestrator import run_pipeline

    parser = build_parser()
    args = parser.parse_args()

    config_data = _load_yaml_config(args.config)
    pipeline_cfg = _read_section(config_data, "pipeline")
    determinism_cfg = _read_section(config_data, "determinism")
    tuning_cfg = _read_section(config_data, "tuning")

    study_name = _coalesce(args.study, pipeline_cfg.get("study"))
    if not study_name:
        parser.error("Study name is required. Provide --study or set pipeline.study in --config.")

    deterministic_mode = _coalesce(args.deterministic_mode, determinism_cfg.get("deterministic_mode"))
    deterministic_mode = bool(deterministic_mode) if deterministic_mode is not None else False

    use_optuna_tuning_cli = args.use_optuna_tuning
    if args.no_optuna_tuning:
        use_optuna_tuning_cli = False

    default_optuna = False if deterministic_mode else True
    use_optuna_tuning = _coalesce(
        use_optuna_tuning_cli,
        tuning_cfg.get("use_optuna_tuning"),
        default_optuna,
    )

    classifier_params = _coalesce(
        _parse_json_dict(args.classifier_params_json, "--classifier-params-json"),
        tuning_cfg.get("classifier_params"),
    )
    regressor_params = _coalesce(
        _parse_json_dict(args.regressor_params_json, "--regressor-params-json"),
        tuning_cfg.get("regressor_params"),
    )

    default_faiss_mode = "flat" if deterministic_mode else "hnsw"

    config = RunConfig(
        study_name=study_name,
        base_dir=Path(
            _coalesce(
                args.base_dir,
                pipeline_cfg.get("base_dir"),
                "/home1/kapachy/Microbium/projects/cMD_downloads",
            )
        ),
        sparsity_threshold=float(_coalesce(args.sparsity_threshold, pipeline_cfg.get("sparsity_threshold"), 0.85)),
        k_neighbors=int(_coalesce(args.k_neighbors, pipeline_cfg.get("k_neighbors"), 40)),
        target_fdr=float(_coalesce(args.target_fdr, pipeline_cfg.get("target_fdr"), 0.05)),
        num_shuffles=int(_coalesce(args.num_shuffles, pipeline_cfg.get("num_shuffles"), 20)),
        correlation_threshold=float(
            _coalesce(args.correlation_threshold, pipeline_cfg.get("correlation_threshold"), 0.95)
        ),
        calibration_features=int(_coalesce(args.calibration_features, pipeline_cfg.get("calibration_features"), 2000)),
        calibration_trials=int(_coalesce(args.calibration_trials, pipeline_cfg.get("calibration_trials"), 50)),
        cluster_batch_size=int(_coalesce(args.cluster_batch_size, pipeline_cfg.get("cluster_batch_size"), 64000)),
        random_seed=int(_coalesce(args.seed, pipeline_cfg.get("seed"), 42)),
        deterministic_mode=deterministic_mode,
        faiss_mode=str(_coalesce(args.faiss_mode, determinism_cfg.get("faiss_mode"), default_faiss_mode)),
        faiss_threads=int(_coalesce(args.faiss_threads, determinism_cfg.get("faiss_threads"), 1)),
        filter_n_jobs=_coalesce(args.filter_n_jobs, determinism_cfg.get("filter_n_jobs")),
        use_optuna_tuning=bool(use_optuna_tuning),
        classifier_params=classifier_params,
        regressor_params=regressor_params,
    )

    validate_run_config(config)

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
