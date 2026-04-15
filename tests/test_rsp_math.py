import numpy as np

from microbiome_knockoffs.analysis_rsp import build_feature_index_map, calculate_threshold


def test_calculate_threshold_returns_inf_when_no_selection():
    W = np.array([-0.5, -0.2, -0.1], dtype=float)
    threshold = calculate_threshold(W, fdr=0.1, offset=1)
    assert threshold == float("inf")


def test_calculate_threshold_finds_valid_cutoff():
    W = np.array([1.5, 1.0, 0.7, -0.1, -0.2, 0.0], dtype=float)
    threshold = calculate_threshold(W, fdr=0.5, offset=1)
    assert np.isfinite(threshold)
    assert threshold > 0


def test_build_feature_index_map_sorted_by_descending_w_with_tie_break():
    W = np.array([0.2, 0.9, 0.9, -0.1, 1.5], dtype=float)
    is_significant = np.array([False, True, True, False, True], dtype=bool)

    feature_map = build_feature_index_map(W, is_significant)

    assert list(feature_map.keys()) == [4, 1, 2, 0, 3]


def test_build_feature_index_map_significance_matches_selected_membership():
    W = np.array([0.4, -0.2, 1.1, 0.0], dtype=float)
    is_significant = np.array([True, False, True, False], dtype=bool)

    feature_map = build_feature_index_map(W, is_significant)

    assert feature_map[2] == (1.1, True)
    assert feature_map[0] == (0.4, True)
    assert feature_map[3] == (0.0, False)
    assert feature_map[1] == (-0.2, False)


def test_build_feature_index_map_validates_vector_shapes():
    W = np.array([0.4, -0.2, 1.1], dtype=float)
    is_significant = np.array([True, False], dtype=bool)

    try:
        build_feature_index_map(W, is_significant)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected shape validation error")
