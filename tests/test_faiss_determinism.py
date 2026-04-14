from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.knockoffs.neighbor_index_faiss import FaissHNSWIndex


K_NEIGHBORS = 8

# Outer knobs for synthetic data generation.
N_GROUPS = 500
FEATURES_PER_GROUP = 500
N_FEATURES_GENERATED = N_GROUPS * FEATURES_PER_GROUP
N_SAMPLES = 300


def _validate_generation_params() -> None:
    expected = N_GROUPS * FEATURES_PER_GROUP
    if N_FEATURES_GENERATED != expected:
        raise ValueError(
            "N_FEATURES_GENERATED must equal N_GROUPS * FEATURES_PER_GROUP "
            f"({N_FEATURES_GENERATED} != {N_GROUPS} * {FEATURES_PER_GROUP})"
        )


def _set_single_thread_runtime() -> None:
    """Reduce scheduler noise when evaluating determinism and timings."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception:
        # The faiss wrapper may vary by build; test still runs with env defaults.
        pass


def _build_correlated_feature_vectors(
    seed: int,
    n_samples: int = N_SAMPLES,
    n_groups: int = N_GROUPS,
    features_per_group: int = FEATURES_PER_GROUP,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature vectors with clear correlation neighborhoods."""

    rng = np.random.default_rng(seed)
    latent = rng.normal(loc=0.0, scale=1.0, size=(n_groups, n_samples)).astype(np.float32)

    vectors: list[np.ndarray] = []
    group_ids: list[int] = []
    for group_id in range(n_groups):
        group_signal = latent[group_id]
        for _ in range(features_per_group):
            noise = rng.normal(loc=0.0, scale=0.18, size=n_samples).astype(np.float32)
            vectors.append(group_signal + noise)
            group_ids.append(group_id)

    raw_vectors = np.asarray(vectors, dtype=np.float32)
    return raw_vectors, np.asarray(group_ids, dtype=np.int32)


def _build_knockoff_like_vectors(
    raw_vectors: np.ndarray,
    seed: int,
    n_insert: int = N_FEATURES_GENERATED,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate online knockoff insertions via correlated-noise perturbations."""

    rng = np.random.default_rng(seed)
    n_insert = min(n_insert, raw_vectors.shape[0])
    parent_indices = rng.choice(raw_vectors.shape[0], size=n_insert, replace=False).astype(np.int32)

    inserted: list[np.ndarray] = []
    for idx in parent_indices:
        parent = raw_vectors[idx]
        sigma = float(np.std(parent))
        noise = rng.normal(loc=0.0, scale=max(1e-6, sigma * 0.25), size=parent.shape[0]).astype(np.float32)
        inserted.append(parent + noise)

    return np.asarray(inserted, dtype=np.float32), parent_indices


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    out = normalize(vectors, axis=1, norm="l2")
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def _exact_inner_product_search(
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = query_vectors @ index_vectors.T
    topk_indices = np.argsort(sims, axis=1)[:, ::-1][:, :k]
    topk_scores = np.take_along_axis(sims, topk_indices, axis=1)
    return topk_scores.astype(np.float32), topk_indices.astype(np.int64)


def _drop_self_neighbors(
    distances: np.ndarray,
    indices: np.ndarray,
    query_global_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    kept_distances = np.empty((indices.shape[0], k), dtype=np.float32)
    kept_indices = np.empty((indices.shape[0], k), dtype=np.int64)

    for row_idx, global_id in enumerate(query_global_ids):
        mask = indices[row_idx] != global_id
        row_indices = indices[row_idx][mask][:k]
        row_distances = distances[row_idx][mask][:k]

        if row_indices.shape[0] != k:
            raise ValueError("Not enough non-self neighbors returned by search")

        kept_indices[row_idx] = row_indices
        kept_distances[row_idx] = row_distances

    return kept_distances, kept_indices


def _recall_at_k(predicted: np.ndarray, reference: np.ndarray) -> float:
    recalls = []
    for pred_row, ref_row in zip(predicted, reference):
        pred_set = set(int(v) for v in pred_row.tolist())
        ref_set = set(int(v) for v in ref_row.tolist())
        recalls.append(len(pred_set & ref_set) / float(len(ref_set)))
    return float(np.mean(recalls))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def _median_neighbor_correlation(
    raw_queries: np.ndarray,
    raw_index_vectors: np.ndarray,
    neighbor_indices: np.ndarray,
) -> float:
    values: list[float] = []
    for row_idx in range(raw_queries.shape[0]):
        query = raw_queries[row_idx]
        for n_idx in neighbor_indices[row_idx]:
            values.append(_pearson_corr(query, raw_index_vectors[int(n_idx)]))
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _run_faiss_experiment(seed: int) -> dict[str, object]:
    _set_single_thread_runtime()
    _validate_generation_params()

    raw_base, _ = _build_correlated_feature_vectors(
        seed=seed,
        n_samples=N_SAMPLES,
        n_groups=N_GROUPS,
        features_per_group=FEATURES_PER_GROUP,
    )
    if raw_base.shape[0] != N_FEATURES_GENERATED:
        raise ValueError(
            f"Generated {raw_base.shape[0]} features, expected {N_FEATURES_GENERATED}"
        )
    search_base = _normalize_rows(raw_base)

    index = FaissHNSWIndex(
        vector_dim=search_base.shape[1],
        m=32,
        ef_construction=200,
        ef_search=100,
    )

    fit_t0 = time.perf_counter()
    index.fit(search_base)
    fit_seconds = time.perf_counter() - fit_t0

    rng = np.random.default_rng(seed + 1)
    n_queries = min(64, search_base.shape[0])
    query_ids = rng.choice(search_base.shape[0], size=n_queries, replace=False).astype(np.int64)
    query_vectors = search_base[query_ids]

    pre_t0 = time.perf_counter()
    approx_pre_d_full, approx_pre_i_full = index.search(query_vectors, K_NEIGHBORS + 1)
    search_pre_seconds = time.perf_counter() - pre_t0
    approx_pre_d, approx_pre_i = _drop_self_neighbors(
        approx_pre_d_full,
        approx_pre_i_full,
        query_global_ids=query_ids,
        k=K_NEIGHBORS,
    )

    exact_pre_d_full, exact_pre_i_full = _exact_inner_product_search(
        search_base,
        query_vectors,
        K_NEIGHBORS + 1,
    )
    exact_pre_d, exact_pre_i = _drop_self_neighbors(
        exact_pre_d_full,
        exact_pre_i_full,
        query_global_ids=query_ids,
        k=K_NEIGHBORS,
    )

    raw_insert, parent_indices = _build_knockoff_like_vectors(
        raw_base,
        seed=seed + 2,
        n_insert=N_FEATURES_GENERATED,
    )
    if raw_insert.shape[0] != raw_base.shape[0]:
        raise ValueError(
            "Knockoff count must match base feature count "
            f"({raw_insert.shape[0]} != {raw_base.shape[0]})"
        )
    search_insert = _normalize_rows(raw_insert)

    insert_t0 = time.perf_counter()
    for row in search_insert:
        index.add(row.reshape(1, -1))
    insert_seconds_total = time.perf_counter() - insert_t0

    combined_search = np.vstack([search_base, search_insert]).astype(np.float32)
    combined_raw = np.vstack([raw_base, raw_insert]).astype(np.float32)

    inserted_start = search_base.shape[0]
    inserted_ids = np.arange(inserted_start, inserted_start + search_insert.shape[0], dtype=np.int64)
    post_local_query_count = min(32, search_insert.shape[0])
    post_local_query_ids = np.arange(post_local_query_count, dtype=np.int64)
    post_query_vectors = search_insert[post_local_query_ids]
    post_query_global_ids = inserted_ids[post_local_query_ids]

    post_t0 = time.perf_counter()
    approx_post_d_full, approx_post_i_full = index.search(post_query_vectors, K_NEIGHBORS + 1)
    search_post_seconds = time.perf_counter() - post_t0
    approx_post_d, approx_post_i = _drop_self_neighbors(
        approx_post_d_full,
        approx_post_i_full,
        query_global_ids=post_query_global_ids,
        k=K_NEIGHBORS,
    )

    exact_post_d_full, exact_post_i_full = _exact_inner_product_search(
        combined_search,
        post_query_vectors,
        K_NEIGHBORS + 1,
    )
    exact_post_d, exact_post_i = _drop_self_neighbors(
        exact_post_d_full,
        exact_post_i_full,
        query_global_ids=post_query_global_ids,
        k=K_NEIGHBORS,
    )

    parent_hits = []
    for row_idx in range(post_local_query_count):
        expected_parent = int(parent_indices[row_idx])
        parent_hits.append(expected_parent in set(int(v) for v in approx_post_i[row_idx].tolist()))

    quality = {
        "recall_pre_at_k": _recall_at_k(approx_pre_i, exact_pre_i),
        "recall_post_at_k": _recall_at_k(approx_post_i, exact_post_i),
        "median_corr_pre": _median_neighbor_correlation(raw_base[query_ids], raw_base, approx_pre_i),
        "median_corr_post": _median_neighbor_correlation(raw_insert[post_local_query_ids], combined_raw, approx_post_i),
        "insert_parent_hit_rate": float(np.mean(np.asarray(parent_hits, dtype=np.float32))),
    }

    timings = {
        "fit_seconds": float(fit_seconds),
        "insert_seconds_total": float(insert_seconds_total),
        "insert_seconds_per_vector": float(insert_seconds_total / max(1, search_insert.shape[0])),
        "search_pre_seconds": float(search_pre_seconds),
        "search_pre_seconds_per_query": float(search_pre_seconds / max(1, query_vectors.shape[0])),
        "search_post_seconds": float(search_post_seconds),
        "search_post_seconds_per_query": float(search_post_seconds / max(1, post_query_vectors.shape[0])),
    }

    return {
        "pre_indices": approx_pre_i,
        "pre_distances": approx_pre_d,
        "post_indices": approx_post_i,
        "post_distances": approx_post_d,
        "quality": quality,
        "timings": timings,
    }


@pytest.mark.parametrize("seed", [20260414])
def test_faiss_hnsw_determinism_speed_and_quality(seed: int) -> None:
    """Validate deterministic search outputs, phase timings, and neighbor quality."""

    run_a = _run_faiss_experiment(seed)
    run_b = _run_faiss_experiment(seed)

    assert np.array_equal(run_a["pre_indices"], run_b["pre_indices"])
    assert np.array_equal(run_a["post_indices"], run_b["post_indices"])
    assert np.allclose(run_a["pre_distances"], run_b["pre_distances"], rtol=1e-6, atol=1e-7)
    assert np.allclose(run_a["post_distances"], run_b["post_distances"], rtol=1e-6, atol=1e-7)

    timings = run_a["timings"]
    for key in (
        "fit_seconds",
        "insert_seconds_total",
        "insert_seconds_per_vector",
        "search_pre_seconds",
        "search_pre_seconds_per_query",
        "search_post_seconds",
        "search_post_seconds_per_query",
    ):
        value = float(timings[key])
        assert np.isfinite(value)
        assert value >= 0.0

    quality = run_a["quality"]
    assert float(quality["recall_pre_at_k"]) >= 0.90
    assert float(quality["recall_post_at_k"]) >= 0.85
    assert float(quality["median_corr_pre"]) >= 0.50
    assert float(quality["median_corr_post"]) >= 0.45
    assert float(quality["insert_parent_hit_rate"]) >= 0.70

    print("deterministic=True")
    print(f"timings={timings}")
    print(f"quality={quality}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))