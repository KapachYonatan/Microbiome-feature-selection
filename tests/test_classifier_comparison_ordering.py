import numpy as np

from microbiome_knockoffs.evaluation_classifier_comparison import _ordered_knockoff_selected_indices


def test_ordered_knockoff_selected_indices_sorts_by_descending_w_real():
    rsp_results = {
        "selected_indices": np.array([2, 0, 3, 1], dtype=int),
        "W_real": np.array([0.8, 0.1, 1.7, 0.5], dtype=float),
    }

    ordered = _ordered_knockoff_selected_indices(rsp_results)

    assert ordered.tolist() == [2, 0, 3, 1]


def test_ordered_knockoff_selected_indices_falls_back_when_w_real_missing():
    rsp_results = {
        "selected_indices": np.array([2, 0, 3, 1], dtype=int),
    }

    ordered = _ordered_knockoff_selected_indices(rsp_results)

    assert ordered.tolist() == [1, 3, 0, 2]


def test_ordered_knockoff_selected_indices_ties_break_by_feature_index():
    rsp_results = {
        "selected_indices": np.array([3, 2, 0, 1], dtype=int),
        "W_real": np.array([1.0, 0.3, 1.0, 1.0], dtype=float),
    }

    ordered = _ordered_knockoff_selected_indices(rsp_results)

    # Tied W_real among {0,2,3} should be ordered by ascending index.
    assert ordered.tolist() == [0, 2, 3, 1]