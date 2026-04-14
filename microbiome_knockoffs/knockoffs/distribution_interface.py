from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DistributionLearner(ABC):
    """Interface for feature-wise knockoff distribution learning.

    Input structure:
    - X_j_raw: feature vector shape (n_samples,).
    - S_matrix_raw: neighbor matrix shape (n_samples, k_neighbors).

    Output structure:
    - predict_support: tuple(mask_or_none, status)
      mask_or_none shape (n_samples,) bool or None when feature is all zeros.
    - predict_values: tuple(values, status)
      values shape (n_sim_nonzero,) for True positions of simulated support.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable learner name for metadata."""

    @abstractmethod
    def predict_support(
        self,
        X_j_raw: np.ndarray,
        S_matrix_raw: np.ndarray,
        classifier_params: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, str]:
        """Predict and sample support mask for a single feature."""

    @abstractmethod
    def predict_values(
        self,
        X_j_raw: np.ndarray,
        S_matrix_raw: np.ndarray,
        is_nonzero_sim: np.ndarray,
        regressor_params: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        """Predict and sample values conditioned on simulated support."""
