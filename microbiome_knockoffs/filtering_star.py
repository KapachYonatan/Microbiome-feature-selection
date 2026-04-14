from __future__ import annotations

import gc

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .contracts import FilteringArtifacts


def cluster_features_star_optimized(
    X: sparse.csr_matrix | np.ndarray,
    correlation_threshold: float = 0.95,
    batch_size: int = 5000,
    n_jobs: int = -1,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Perform variance-priority star clustering over features.

    Input:
    - X: matrix with shape (n_samples, n_features).
    - correlation_threshold: minimum leader-member Pearson correlation.
    - batch_size: neighbor graph query batch size.

    Output:
    - leaders: ndarray[int] sorted feature indices to keep.
    - clusters: dict[leader_idx -> ndarray[member_indices]].
    """

    _, n_features = X.shape
    print(f"Starting Star Clustering (threshold={correlation_threshold})...")

    if sparse.issparse(X):
        mean = np.array(X.mean(axis=0)).flatten()
        variances = np.array(X.power(2).mean(axis=0)).flatten() - mean ** 2
        X_feat = X.T
    else:
        variances = np.var(X, axis=0)
        X_feat = X.T

    sorted_indices = np.argsort(variances)[::-1]

    X_dense = X_feat.toarray() if sparse.issparse(X_feat) else X_feat
    X_norm = normalize(X_dense, axis=1, norm="l2")
    del X_dense
    gc.collect()

    radius = np.sqrt(2 * (1 - correlation_threshold))
    nn_model = NearestNeighbors(radius=radius, metric="euclidean", n_jobs=n_jobs, algorithm="auto")
    nn_model.fit(X_norm)

    sparse_graphs = []
    n_batches = int(np.ceil(n_features / batch_size))

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n_features)
        batch_graph = nn_model.radius_neighbors_graph(X_norm[start:end], mode="distance")
        sparse_graphs.append(batch_graph)

        if batch_idx % 10 == 0:
            print(f"  Neighbor batch {batch_idx + 1}/{n_batches}")
            gc.collect()

    adjacency_matrix = sparse.vstack(sparse_graphs, format="csr")
    del sparse_graphs, nn_model, X_norm
    gc.collect()

    is_covered = np.zeros(n_features, dtype=bool)
    leaders: list[int] = []
    clusters: dict[int, np.ndarray] = {}

    for current_idx in sorted_indices:
        if is_covered[current_idx]:
            continue

        leaders.append(int(current_idx))
        neighbors = adjacency_matrix[current_idx].indices
        uncovered_neighbors = neighbors[~is_covered[neighbors]]
        clusters[int(current_idx)] = uncovered_neighbors
        is_covered[uncovered_neighbors] = True

    print(
        "Star Clustering complete: "
        f"original={n_features}, kept={len(leaders)}, "
        f"compression={100 * (1 - len(leaders) / n_features):.2f}%"
    )

    return np.array(sorted(leaders), dtype=np.int32), clusters


def build_named_clusters(
    clusters: dict[int, np.ndarray],
    feature_names: np.ndarray,
) -> dict[str, list[str]]:
    """Convert integer-index clusters into feature-name mapping."""

    named_clusters: dict[str, list[str]] = {}
    for leader_idx in sorted(clusters.keys()):
        member_indices = clusters[leader_idx]
        leader_name = str(feature_names[leader_idx])
        member_names = [str(feature_names[idx]) for idx in member_indices]
        named_clusters[leader_name] = member_names
    return named_clusters


def run_feature_filtering(
    X_sparse: sparse.csr_matrix,
    feature_names: np.ndarray,
    correlation_threshold: float = 0.95,
    batch_size: int = 64000,
    n_jobs: int = -1,
) -> FilteringArtifacts:
    """Run feature filtering and return in-memory filtering artifacts.

    Input:
    - X_sparse: csr_matrix (n_samples, n_features).
    - feature_names: ndarray[str] (n_features,).

    Output:
    - FilteringArtifacts with filtered matrix, names, leaders, and cluster map.
    """

    leaders, clusters = cluster_features_star_optimized(
        X_sparse,
        correlation_threshold=correlation_threshold,
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    X_filtered = X_sparse[:, leaders].copy().tocsr()
    feature_names_filtered = feature_names[leaders]

    return FilteringArtifacts(
        X_filtered=X_filtered,
        feature_names_filtered=feature_names_filtered,
        leaders=leaders,
        clusters=clusters,
    )
