import numpy as np
from scipy import sparse

from microbiome_knockoffs.evaluation_classifier_comparison import (
    _align_raw_sample_rows,
    build_k_grid,
    build_gene_feature_matrix,
    build_kmeans_clustered_feature_matrix,
    method_registry,
)


def test_build_gene_feature_matrix_aggregates_by_gene_token():
    X_source = sparse.csr_matrix(
        np.array(
            [
                [1.0, 2.0, 5.0, 7.0],
                [2.0, 4.0, 6.0, 8.0],
            ],
            dtype=np.float32,
        )
    )
    feature_names = np.array(["g1|b1", "g1|b2", "g2|b1", "g3|b3"], dtype=str)

    X_gene, gene_names = build_gene_feature_matrix(X_source, feature_names)

    assert gene_names.tolist() == ["g1", "g2", "g3"]
    assert X_gene.shape == (2, 3)
    assert np.allclose(
        X_gene.toarray(),
        np.array(
            [
                [1.5, 5.0, 7.0],
                [3.0, 6.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_kmeans_clustered_feature_matrix_respects_target_k():
    X_gene = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    clustered_matrix, cluster_names = build_kmeans_clustered_feature_matrix(
        X_gene,
        target_k=3,
        random_state=42,
    )

    assert clustered_matrix.shape == (4, 3)
    assert cluster_names.tolist() == ["gene_cluster_0", "gene_cluster_1", "gene_cluster_2"]
    assert np.isfinite(clustered_matrix.data).all()


def test_build_kmeans_clustered_feature_matrix_caps_clusters_to_feature_count():
    X_gene = sparse.csr_matrix(np.eye(3, dtype=np.float32))

    clustered_matrix, cluster_names = build_kmeans_clustered_feature_matrix(
        X_gene,
        target_k=10,
        random_state=42,
    )

    assert clustered_matrix.shape == (3, 3)
    assert len(cluster_names) == 3


def test_method_registry_contains_gene_clustered_random():
    method = method_registry()["gene_clustered_random"]

    assert method.kind == "random"
    assert method.source == "gene_clustered"


def test_align_raw_sample_rows_trims_extra_rows():
    X_raw = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3))

    aligned = _align_raw_sample_rows(X_raw, y_sample_count=3)

    assert aligned.shape == (3, 3)
    assert np.allclose(aligned.toarray(), X_raw.toarray()[:3, :])


def test_align_raw_sample_rows_raises_when_raw_has_fewer_rows():
    X_raw = sparse.csr_matrix(np.arange(6, dtype=np.float32).reshape(2, 3))

    try:
        _align_raw_sample_rows(X_raw, y_sample_count=3)
    except ValueError as exc:
        assert "fewer rows than y_clean" in str(exc)
    else:
        raise AssertionError("Expected mismatch error when raw has fewer rows")


def test_build_k_grid_uses_requested_end_without_capacity_clamp():
    k_values = build_k_grid(k_start=182, k_end_requested=5000, k_grid_points=10)

    assert k_values[0] == 182
    assert k_values[-1] == 5000
    assert len(k_values) >= 2


def test_build_k_grid_raises_when_end_below_start():
    try:
        build_k_grid(k_start=300, k_end_requested=200, k_grid_points=10)
    except ValueError as exc:
        assert "Requested K_end" in str(exc)
    else:
        raise AssertionError("Expected ValueError when K_end is smaller than K_start")