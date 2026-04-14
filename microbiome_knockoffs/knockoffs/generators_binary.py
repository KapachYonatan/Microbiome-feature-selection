from __future__ import annotations

import numpy as np

from .generators_base import BaseKnockoffGenerator


class BinaryKnockoffGenerator(BaseKnockoffGenerator):
    """Knockoff generator with binary transform for high-sparsity features.

    Input:
    - X: ndarray shape (n_samples, n_features).

    Output via generate():
    - KnockoffOutputs where X_transformed stores the binary-transformed matrix.
    """

    def __init__(
        self,
        X: np.ndarray,
        sparsity_threshold: float = 0.8,
        **kwargs,
    ) -> None:
        self.sparsity_threshold = sparsity_threshold
        self.binary_mask: np.ndarray | None = None
        super().__init__(X=X, **kwargs)

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        print("Initializing Binary Knockoff Generator...")
        sparsity = (X == 0).mean(axis=0)
        self.binary_mask = sparsity > self.sparsity_threshold

        print(
            f"  > Transforming {int(np.sum(self.binary_mask))} sparse features to binary "
            f"(threshold={self.sparsity_threshold})"
        )

        X_transformed = X.copy().astype(np.float32)
        X_transformed[:, self.binary_mask] = (X[:, self.binary_mask] != 0).astype(np.float32)
        return X_transformed

    def _sample_feature(self, X_j_raw: np.ndarray, S_matrix_raw: np.ndarray, feature_idx: int) -> np.ndarray:
        unique_vals = np.unique(X_j_raw)
        is_binary = np.all(np.isin(unique_vals, [0.0, 1.0])) and len(unique_vals) <= 2

        if not is_binary:
            return super()._sample_feature(X_j_raw, S_matrix_raw, feature_idx)

        X_tilde = np.zeros(len(X_j_raw), dtype=np.float32)
        is_nonzero_sim, support_status = self.distribution_learner.predict_support(
            X_j_raw,
            S_matrix_raw,
            classifier_params=self.clf_params,
            rng=self.rng,
        )
        self._log_status(feature_idx, "Support", support_status, int(np.sum(np.abs(X_j_raw) > 0)))

        if is_nonzero_sim is None:
            return X_tilde

        X_tilde[is_nonzero_sim] = 1.0
        return X_tilde
