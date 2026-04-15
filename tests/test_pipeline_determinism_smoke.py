from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from microbiome_knockoffs.contracts import RunConfig
from microbiome_knockoffs.pipeline_orchestrator import run_pipeline


def _write_toy_study(base_dir: Path, study_name: str, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    study_dir = base_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 36
    n_features = 12

    X = rng.poisson(lam=0.35, size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples, endpoint=False).astype(np.int32)
    genes = np.array([f"gene_{i:03d}" for i in range(n_features)], dtype=str)

    sparse.save_npz(study_dir / "X_clean.npz", sparse.csr_matrix(X))
    np.save(study_dir / "y_clean.npy", y)
    np.savetxt(study_dir / "genes_clean.txt", genes, fmt="%s")

    return study_dir


def _load_rsp_dict(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=True)
    if hasattr(loaded, "item"):
        return loaded.item()
    raise TypeError("Expected saved RSP dict in numpy object file")


def _normalize_feature_index_map(payload: dict) -> list[tuple[int, tuple[float, bool]]]:
    if "feature_index_map" not in payload:
        raise KeyError("rsp_results payload missing feature_index_map")
    feature_map = payload["feature_index_map"]
    return [
        (int(feature_index), (float(values[0]), bool(values[1])))
        for feature_index, values in feature_map.items()
    ]


def test_pipeline_is_reproducible_in_deterministic_mode(tmp_path: Path):
    study_name = "toy_study"
    _write_toy_study(tmp_path, study_name)

    config = RunConfig(
        study_name=study_name,
        base_dir=tmp_path,
        sparsity_threshold=0.95,
        k_neighbors=4,
        target_fdr=0.1,
        num_shuffles=6,
        correlation_threshold=0.9,
        calibration_features=6,
        calibration_trials=0,
        cluster_batch_size=256,
        random_seed=2026,
        deterministic_mode=True,
        faiss_mode="flat",
        faiss_threads=1,
        filter_n_jobs=1,
        use_optuna_tuning=False,
    )

    run_a = run_pipeline(config)
    run_b = run_pipeline(config)

    x_binary_a = np.load(run_a.x_binary_path)
    x_binary_b = np.load(run_b.x_binary_path)
    assert np.array_equal(x_binary_a, x_binary_b)

    x_knockoff_a = np.load(run_a.x_knockoff_path)
    x_knockoff_b = np.load(run_b.x_knockoff_path)
    assert np.array_equal(x_knockoff_a, x_knockoff_b)

    genes_filtered_a = np.loadtxt(run_a.genes_filtered_path, dtype=str)
    genes_filtered_b = np.loadtxt(run_b.genes_filtered_path, dtype=str)
    assert np.array_equal(genes_filtered_a, genes_filtered_b)

    with run_a.clusters_path.open("r", encoding="utf-8") as handle:
        clusters_a = json.load(handle)
    with run_b.clusters_path.open("r", encoding="utf-8") as handle:
        clusters_b = json.load(handle)
    assert clusters_a == clusters_b

    rsp_a = _load_rsp_dict(run_a.rsp_results_path)
    rsp_b = _load_rsp_dict(run_b.rsp_results_path)
    assert "W_real" not in rsp_a and "W_real" not in rsp_b
    assert "selected_indices" not in rsp_a and "selected_indices" not in rsp_b
    assert _normalize_feature_index_map(rsp_a) == _normalize_feature_index_map(rsp_b)
    assert rsp_a["RP"] == rsp_b["RP"]
    np.testing.assert_allclose(np.asarray(rsp_a["rsp"]), np.asarray(rsp_b["rsp"]))
    np.testing.assert_allclose(np.asarray(rsp_a["beta_values"]), np.asarray(rsp_b["beta_values"]))
    assert float(rsp_a["SP"]) == float(rsp_b["SP"])
    assert float(rsp_a["threshold"]) == float(rsp_b["threshold"])

    with run_a.metadata_path.open("r", encoding="utf-8") as handle:
        metadata_a = json.load(handle)
    with run_b.metadata_path.open("r", encoding="utf-8") as handle:
        metadata_b = json.load(handle)

    params_a = metadata_a["parameters"]
    params_b = metadata_b["parameters"]
    assert params_a["deterministic_mode"] is True
    assert params_a["faiss_mode"] == "flat"
    assert params_a["use_optuna_tuning"] is False
    assert params_a == params_b