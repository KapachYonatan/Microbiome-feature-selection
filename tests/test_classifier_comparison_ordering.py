import numpy as np
import pytest

from microbiome_knockoffs.evaluation_classifier_comparison import _ordered_knockoff_selected_indices


def test_ordered_knockoff_selected_indices_sorts_by_descending_w_real():
    rsp_results = {
        "feature_index_map": {
            2: (1.7, True),
            0: (0.8, True),
            3: (0.5, True),
            1: (0.1, False),
        },
    }

    ordered = _ordered_knockoff_selected_indices(rsp_results)

    assert ordered.tolist() == [2, 0, 3]


def test_ordered_knockoff_selected_indices_raises_when_feature_map_missing():
    rsp_results = {}

    with pytest.raises(ValueError, match="missing non-empty feature_index_map"):
        _ordered_knockoff_selected_indices(rsp_results)


def test_ordered_knockoff_selected_indices_ties_break_by_feature_index():
    rsp_results = {
        "feature_index_map": {
            0: (1.0, True),
            2: (1.0, True),
            3: (1.0, True),
            1: (0.3, False),
        },
    }

    ordered = _ordered_knockoff_selected_indices(rsp_results)

    # Tied W_real among {0,2,3} should be ordered by ascending index.
    assert ordered.tolist() == [0, 2, 3]


def test_ordered_knockoff_selected_indices_raises_for_descending_violation():
    rsp_results = {
        "feature_index_map": {
            0: (0.9, True),
            1: (1.2, True),
            2: (0.1, False),
        },
    }

    with pytest.raises(ValueError, match="sorted by descending W_real"):
        _ordered_knockoff_selected_indices(rsp_results)


def test_ordered_knockoff_selected_indices_raises_for_tie_break_violation():
    rsp_results = {
        "feature_index_map": {
            2: (1.0, True),
            1: (1.0, True),
            0: (0.5, False),
        },
    }

    with pytest.raises(ValueError, match="tie-break must use ascending feature index"):
        _ordered_knockoff_selected_indices(rsp_results)