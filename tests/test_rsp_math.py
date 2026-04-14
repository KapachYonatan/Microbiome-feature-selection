import numpy as np

from microbiome_knockoffs.analysis_rsp import calculate_threshold, select_features


def test_calculate_threshold_returns_inf_when_no_selection():
    W = np.array([-0.5, -0.2, -0.1], dtype=float)
    threshold = calculate_threshold(W, fdr=0.1, offset=1)
    assert threshold == float("inf")


def test_select_features_sorted_descending_by_w():
    W = np.array([0.2, 0.9, 0.4, 1.1, -0.3], dtype=float)
    threshold = 0.3
    selected = select_features(W, threshold)
    assert selected.tolist() == [3, 1, 2]


def test_select_features_tie_breaks_by_feature_index():
    W = np.array([0.7, 0.9, 0.9, 0.2, 0.9], dtype=float)
    threshold = 0.7
    selected = select_features(W, threshold)
    assert selected.tolist() == [1, 2, 4, 0]
    selected_w = W[selected]
    assert np.all(selected_w[:-1] >= selected_w[1:])


def test_calculate_threshold_finds_valid_cutoff():
    W = np.array([1.5, 1.0, 0.7, -0.1, -0.2, 0.0], dtype=float)
    threshold = calculate_threshold(W, fdr=0.5, offset=1)
    assert np.isfinite(threshold)
    assert threshold > 0
