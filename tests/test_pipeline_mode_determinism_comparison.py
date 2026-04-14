from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.contracts import RunConfig
from microbiome_knockoffs.pipeline_orchestrator import run_pipeline


def _write_small_study(base_dir: Path, study_name: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    study_dir = base_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 150
    n_features = 1000

    X = rng.poisson(lam=0.4, size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples, endpoint=False).astype(np.int32)
    genes = np.array([f"gene_{idx:03d}" for idx in range(n_features)], dtype=str)

    sparse.save_npz(study_dir / "X_clean.npz", sparse.csr_matrix(X))
    np.save(study_dir / "y_clean.npy", y)
    np.savetxt(study_dir / "genes_clean.txt", genes, fmt="%s")
    return study_dir


def _load_rsp(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    if hasattr(payload, "item"):
        return payload.item()
    raise TypeError("Expected a dict-like RSP payload")


def _compare_rsp_dicts(left: dict, right: dict) -> bool:
    return (
        np.array_equal(np.asarray(left["W_real"]), np.asarray(right["W_real"]))
        and np.array_equal(np.asarray(left["selected_indices"]), np.asarray(right["selected_indices"]))
        and np.array_equal(np.asarray(left["rsp"]), np.asarray(right["rsp"]))
        and np.array_equal(np.asarray(left["beta_values"]), np.asarray(right["beta_values"]))
        and int(left["RP"]) == int(right["RP"])
        and float(left["SP"]) == float(right["SP"])
        and float(left["threshold"]) == float(right["threshold"])
    )


def _compare_artifacts(run_a, run_b) -> dict[str, object]:
    x_binary_equal = np.array_equal(np.load(run_a.x_binary_path), np.load(run_b.x_binary_path))
    x_knockoff_equal = np.array_equal(np.load(run_a.x_knockoff_path), np.load(run_b.x_knockoff_path))
    genes_equal = np.array_equal(
        np.loadtxt(run_a.genes_filtered_path, dtype=str),
        np.loadtxt(run_b.genes_filtered_path, dtype=str),
    )

    with run_a.clusters_path.open("r", encoding="utf-8") as handle:
        clusters_a = json.load(handle)
    with run_b.clusters_path.open("r", encoding="utf-8") as handle:
        clusters_b = json.load(handle)
    clusters_equal = clusters_a == clusters_b

    rsp_equal = _compare_rsp_dicts(_load_rsp(run_a.rsp_results_path), _load_rsp(run_b.rsp_results_path))

    checks = {
        "x_binary_equal": x_binary_equal,
        "x_knockoff_equal": x_knockoff_equal,
        "genes_equal": genes_equal,
        "clusters_equal": clusters_equal,
        "rsp_equal": rsp_equal,
    }
    checks["determinism_score"] = int(sum(1 for value in checks.values() if value))
    return checks


def _run_mode_twice(base_dir: Path, mode_name: str, mode_overrides: dict) -> dict[str, object]:
    seed = 4242
    study_a = f"{mode_name}_a"
    study_b = f"{mode_name}_b"

    _write_small_study(base_dir, study_a, seed=seed)
    _write_small_study(base_dir, study_b, seed=seed)

    common = dict(
        sparsity_threshold=0.9,
        k_neighbors=4,
        target_fdr=0.1,
        num_shuffles=5,
        correlation_threshold=0.9,
        calibration_features=6,
        calibration_trials=0,
        cluster_batch_size=128,
        random_seed=2026,
        use_optuna_tuning=False,
    )

    config_a = RunConfig(study_name=study_a, base_dir=base_dir, **common, **mode_overrides)
    config_b = RunConfig(study_name=study_b, base_dir=base_dir, **common, **mode_overrides)

    run_a = run_pipeline(config_a)
    run_b = run_pipeline(config_b)

    return _compare_artifacts(run_a, run_b)


def test_compare_mode_determinism_small_random_data(tmp_path: Path):
    mode_overrides = {
        "deterministic": {
            "deterministic_mode": True,
            "faiss_mode": "flat",
            "faiss_threads": 1,
            "filter_n_jobs": 1,
        },
        "single_threaded": {
            "deterministic_mode": False,
            "faiss_mode": "hnsw",
            "faiss_threads": 1,
            "filter_n_jobs": 1,
        },
        "multi_threaded": {
            "deterministic_mode": False,
            "faiss_mode": "hnsw",
            "faiss_threads": 4,
            "filter_n_jobs": -1,
        },
    }

    results: dict[str, dict[str, object]] = {}
    for mode_name, overrides in mode_overrides.items():
        results[mode_name] = _run_mode_twice(tmp_path, mode_name, overrides)

    deterministic = results["deterministic"]
    assert deterministic["x_binary_equal"] is True
    assert deterministic["x_knockoff_equal"] is True
    assert deterministic["genes_equal"] is True
    assert deterministic["clusters_equal"] is True
    assert deterministic["rsp_equal"] is True

    assert int(deterministic["determinism_score"]) >= int(results["single_threaded"]["determinism_score"])
    assert int(deterministic["determinism_score"]) >= int(results["multi_threaded"]["determinism_score"])

    print(f"Mode determinism comparison: {results}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))