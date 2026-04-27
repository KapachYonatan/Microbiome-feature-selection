#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.knockoffs import FaissFlatIPIndex, FaissHNSWIndex


DEFAULT_PARAM_SETS = [
    {
        "id": "baseline_shallow",
        "classifier": {
            "objective": "binary",
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbose": -1,
            "n_jobs": 1,
        },
    },
    {
        "id": "sparse_imbalanced",
        "classifier": {
            "objective": "binary",
            "max_depth": 5,
            "num_leaves": 15,
            "is_unbalance": True,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
        },
    },
    {
        "id": "sparse_imbalanced_plus",
        "classifier": {
            "objective": "binary",
            "metric": "auc",
            "max_depth": 5,
            "num_leaves": 15,
            "scale_pos_weight": 5.0,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "verbose": -1,
            "n_jobs": 1,
        },
    },
]


@dataclass(frozen=True)
class SandboxConfig:
    base_dir: Path
    study_name: str
    run_folder: str
    sampled_target_features: int
    k_neighbors: int
    faiss_mode: str
    faiss_threads: int
    random_seed: int
    test_size: float
    output_prefix: str
    param_sets_json: str
    quiet: bool

    @property
    def study_dir(self) -> Path:
        return self.base_dir / self.study_name

    @property
    def run_dir(self) -> Path:
        return self.study_dir / "runs" / self.run_folder


@dataclass(frozen=True)
class ParameterSet:
    config_id: str
    classifier_params: dict


def _log(msg: str, quiet: bool) -> None:
    if quiet:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _as_1d_str(arr: np.ndarray) -> np.ndarray:
    if np.ndim(arr) == 0:
        return np.array([str(arr)])
    return arr.astype(str)


@contextmanager
def _suppress_lgbm_feature_name_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        yield


def _configure_faiss_runtime(faiss_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(faiss_threads)
    os.environ["MKL_NUM_THREADS"] = str(faiss_threads)
    try:
        import faiss

        faiss.omp_set_num_threads(int(faiss_threads))
    except Exception:
        pass


def _load_run_matrices(config: SandboxConfig) -> tuple[np.ndarray, np.ndarray]:
    x_binary_path = config.run_dir / "X_binary.npy"
    genes_filtered_path = config.run_dir / "genes_filtered.txt"

    missing = [
        str(path)
        for path in (x_binary_path, genes_filtered_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required run artifacts: {missing}")

    X_binary = np.load(x_binary_path).astype(np.float32, copy=False)
    genes_filtered = _as_1d_str(np.loadtxt(genes_filtered_path, dtype=str))

    if X_binary.ndim != 2:
        raise ValueError(f"Expected 2D X_binary.npy, got shape {X_binary.shape}")
    if X_binary.shape[1] != genes_filtered.shape[0]:
        raise ValueError(
            "Feature mismatch between X_binary and genes_filtered: "
            f"{X_binary.shape[1]} != {genes_filtered.shape[0]}"
        )

    return X_binary, genes_filtered


def _merge_classifier_params(user_params: dict, seed: int) -> dict:
    params = {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.1,
        "objective": "binary",
        "verbose": -1,
        "n_jobs": 1,
        "random_state": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    params.update(user_params)
    return params


def _parse_param_sets(raw_json: str, seed: int) -> list[ParameterSet]:
    parsed = json.loads(raw_json)
    if isinstance(parsed, dict) and "param_sets" in parsed:
        parsed = parsed["param_sets"]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--param-sets-json must decode to a non-empty JSON list")

    param_sets: list[ParameterSet] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Parameter set #{idx} must be an object")

        config_id = str(item.get("id", f"config_{idx}"))
        clf = item.get("classifier", item.get("classifier_params", {}))
        if not isinstance(clf, dict):
            raise TypeError(f"classifier params for {config_id} must be an object")

        param_sets.append(
            ParameterSet(
                config_id=config_id,
                classifier_params=_merge_classifier_params(clf, seed=seed),
            )
        )

    return param_sets


def _build_neighbor_index(search_vectors: np.ndarray, faiss_mode: str):
    vector_dim = int(search_vectors.shape[1])
    if faiss_mode == "flat":
        index = FaissFlatIPIndex(vector_dim=vector_dim)
    else:
        index = FaissHNSWIndex(vector_dim=vector_dim)
    index.fit(search_vectors)
    return index


def _evaluate_support(
    S_matrix: np.ndarray,
    target_values: np.ndarray,
    clf_params: dict,
    test_size: float,
    seed: int,
) -> dict[str, float | str]:
    y_binary = (target_values > 0).astype(np.int32)
    unique = np.unique(y_binary)
    if unique.size < 2:
        return {
            "support_status": "degenerate_target",
            "support_accuracy": np.nan,
            "support_recall": np.nan,
            "support_precision": np.nan,
            "support_roc_auc": np.nan,
            "support_pr_auc": np.nan,
        }

    class_counts = np.bincount(y_binary, minlength=2)
    if int(np.min(class_counts)) < 2:
        return {
            "support_status": "too_few_minority",
            "support_accuracy": np.nan,
            "support_recall": np.nan,
            "support_precision": np.nan,
            "support_roc_auc": np.nan,
            "support_pr_auc": np.nan,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        S_matrix,
        y_binary,
        test_size=test_size,
        random_state=seed,
        stratify=y_binary,
    )

    model = lgb.LGBMClassifier(**clf_params)
    try:
        with _suppress_lgbm_feature_name_warnings():
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
    except Exception as exc:
        return {
            "support_status": f"fit_failed:{type(exc).__name__}",
            "support_accuracy": np.nan,
            "support_recall": np.nan,
            "support_precision": np.nan,
            "support_roc_auc": np.nan,
            "support_pr_auc": np.nan,
        }

    y_pred = (y_score >= 0.5).astype(np.int32)

    roc_auc = np.nan
    pr_auc = np.nan
    if np.unique(y_test).size == 2:
        roc_auc = float(roc_auc_score(y_test, y_score))
        pr_auc = float(average_precision_score(y_test, y_score))

    return {
        "support_status": "ok",
        "support_accuracy": float(accuracy_score(y_test, y_pred)),
        "support_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "support_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "support_roc_auc": roc_auc,
        "support_pr_auc": pr_auc,
    }


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "support_accuracy",
        "support_recall",
        "support_precision",
        "support_roc_auc",
        "support_pr_auc",
    ]
    rename_map = {
        "support_accuracy_mean": "Acc (avg)",
        "support_accuracy_std": "Acc (std)",
        "support_recall_mean": "Rec (avg)",
        "support_recall_std": "Rec (std)",
        "support_precision_mean": "Prec (avg)",
        "support_precision_std": "Prec (std)",
        "support_roc_auc_mean": "Roc_auc (avg)",
        "support_roc_auc_std": "Roc_auc (std)",
        "support_pr_auc_mean": "Pr_auc (avg)",
        "support_pr_auc_std": "Pr_auc (std)",
    }

    grouped = df.groupby("config_id")
    summary = grouped[cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "config_id" if col == ("config_id", "") else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.rename(columns=rename_map)

    numeric_cols = [c for c in summary.columns if c != "config_id"]
    summary[numeric_cols] = summary[numeric_cols].round(3)

    return summary


def run_sandbox(config: SandboxConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    t0 = perf_counter()
    _log("Loading run artifacts (X_binary + genes_filtered).", config.quiet)
    X_binary, genes_filtered = _load_run_matrices(config)
    n_samples, n_features = X_binary.shape

    _log(f"Matrix loaded: {n_samples} samples x {n_features} features", config.quiet)
    _log(f"Parsing parameter sets JSON.", config.quiet)
    param_sets = _parse_param_sets(config.param_sets_json, seed=config.random_seed)

    _log(f"Configuring FAISS runtime threads = {config.faiss_threads}", config.quiet)
    _configure_faiss_runtime(config.faiss_threads)

    _log("Preparing feature vectors and building FAISS index.", config.quiet)
    raw_vectors = np.ascontiguousarray(X_binary.T.astype(np.float32, copy=False))
    search_vectors = normalize(raw_vectors, axis=1, norm="l2")
    search_vectors = np.ascontiguousarray(search_vectors.astype(np.float32, copy=False))
    neighbor_index = _build_neighbor_index(search_vectors, faiss_mode=config.faiss_mode)

    rng = np.random.default_rng(config.random_seed)
    n_sample = min(config.sampled_target_features, n_features)
    sampled_targets = rng.choice(n_features, size=n_sample, replace=False)

    _log(
        f"Evaluating {n_sample} sampled features with k={config.k_neighbors} across {len(param_sets)} parameter sets.",
        config.quiet,
    )

    rows: list[dict[str, object]] = []
    for feature_idx in sampled_targets:
        query = search_vectors[int(feature_idx)].reshape(1, -1)
        _, neighbor_ids = neighbor_index.search(query, config.k_neighbors + 1)
        neighbors = [int(i) for i in neighbor_ids[0] if int(i) != int(feature_idx)][: config.k_neighbors]

        if not neighbors:
            for param_set in param_sets:
                rows.append(
                    {
                        "config_id": param_set.config_id,
                        "target_feature_idx": int(feature_idx),
                        "target_feature_name": str(genes_filtered[int(feature_idx)]),
                        "n_neighbors": 0,
                        "n_nonzero": int(np.sum(raw_vectors[int(feature_idx)] > 0)),
                        "support_status": "no_neighbors",
                        "support_accuracy": np.nan,
                        "support_recall": np.nan,
                        "support_precision": np.nan,
                        "support_roc_auc": np.nan,
                        "support_pr_auc": np.nan,
                    }
                )
            continue

        S_matrix = raw_vectors[neighbors].T
        target_values = raw_vectors[int(feature_idx)]

        for param_set in param_sets:
            base = {
                "config_id": param_set.config_id,
                "target_feature_idx": int(feature_idx),
                "target_feature_name": str(genes_filtered[int(feature_idx)]),
                "n_neighbors": len(neighbors),
                "n_nonzero": int(np.sum(target_values > 0)),
            }

            support = _evaluate_support(
                S_matrix=S_matrix,
                target_values=target_values,
                clf_params=param_set.classifier_params,
                test_size=config.test_size,
                seed=config.random_seed,
            )

            rows.append({**base, **support})

    detailed_df = pd.DataFrame(rows)
    summary_df = _summarize_results(detailed_df)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{config.output_prefix}_{stamp}"
    detailed_csv = config.run_dir / f"{prefix}_feature_results.csv"
    summary_csv = config.run_dir / f"{prefix}_summary.csv"

    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    sort_col = "Roc_auc (avg)"
    if sort_col in summary_df.columns:
        display_df = summary_df.sort_values(sort_col, ascending=False)
    else:
        display_df = summary_df

    print("\n=== LGBM Feature-Learning Sandbox Summary ===")
    print(display_df.to_string(index=False))
    print(f"\nSaved detailed CSV: {detailed_csv}")
    print(f"Saved summary CSV:  {summary_csv}")
    _log(f"Finished in {perf_counter() - t0:.1f}s", config.quiet)

    return detailed_df, summary_df, detailed_csv, summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sandbox for sweeping LGBM params on feature-distribution learning using "
            "pipeline-style FAISS Markov blankets from X_binary."
        )
    )
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--run-folder", required=True, help="Run folder under study/runs")
    parser.add_argument("--sampled-target-features", type=int, default=200)
    parser.add_argument("--k-neighbors", type=int, default=40)
    parser.add_argument("--faiss-mode", choices=["hnsw", "flat"], default="hnsw")
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-prefix", default="lgbm_feature_learning_sandbox")
    parser.add_argument(
        "--param-sets-json",
        default=json.dumps(DEFAULT_PARAM_SETS),
        help=(
            "JSON list of parameter-set objects. Each object supports keys: "
            "id, classifier."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = SandboxConfig(
        base_dir=Path(args.base_dir),
        study_name=args.study,
        run_folder=args.run_folder,
        sampled_target_features=int(args.sampled_target_features),
        k_neighbors=int(args.k_neighbors),
        faiss_mode=str(args.faiss_mode),
        faiss_threads=int(args.faiss_threads),
        random_seed=int(args.random_seed),
        test_size=float(args.test_size),
        output_prefix=str(args.output_prefix),
        param_sets_json=str(args.param_sets_json),
        quiet=bool(args.quiet),
    )

    if config.sampled_target_features < 1:
        raise ValueError("--sampled-target-features must be >= 1")
    if config.k_neighbors < 1:
        raise ValueError("--k-neighbors must be >= 1")
    if config.faiss_threads < 1:
        raise ValueError("--faiss-threads must be >= 1")
    if not (0.0 < config.test_size < 1.0):
        raise ValueError("--test-size must be in (0, 1)")

    run_sandbox(config)


if __name__ == "__main__":
    main()