from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from ..contracts import CalibrationBatch, KnockoffOutputs
from .distribution_hurdle_lgbm import HurdleLGBMDistribution
from .distribution_interface import DistributionLearner
from .neighbor_index_faiss import FaissHNSWIndex
from .neighbor_index_interface import NeighborIndex
from .tuning_interface import HyperparameterTuner
from .tuning_optuna_lgbm import OptunaLGBMTuner


class BaseKnockoffGenerator:
    """Base modular knockoff generator with swappable learner, tuner, and index backends.

    Input:
    - X: ndarray shape (n_samples, n_features).

    Output via generate():
    - KnockoffOutputs with transformed original matrix and knockoff matrix,
      both shape (n_samples, n_features).
    """

    def __init__(
        self,
        X: np.ndarray,
        k_neighbors: int = 50,
        distribution_learner: DistributionLearner | None = None,
        tuner: HyperparameterTuner | None = None,
        neighbor_index: NeighborIndex | None = None,
        random_seed: int = 42,
    ) -> None:
        self.k = k_neighbors
        self.logs: list[dict[str, Any]] = []
        self.rng = np.random.default_rng(random_seed)

        X_processed = self._preprocess_data(X)
        self.X_processed = X_processed.astype(np.float32, copy=False)
        self.n_samples, self.n_features = self.X_processed.shape

        print(f"Initializing Generator for {self.X_processed.shape} matrix...")

        raw_features = self.X_processed.T.astype(np.float32, copy=False)
        self.raw_vectors = np.ascontiguousarray(raw_features)

        print("  > Normalizing vectors for cosine search...")
        self.search_vectors = normalize(raw_features, axis=1, norm="l2")
        self.search_vectors = np.ascontiguousarray(self.search_vectors.astype(np.float32, copy=False))

        self.neighbor_index = neighbor_index or FaissHNSWIndex(vector_dim=self.n_samples)
        self.neighbor_index.fit(self.search_vectors)

        self.distribution_learner = distribution_learner or HurdleLGBMDistribution()
        self.tuner = tuner or OptunaLGBMTuner()

        self.clf_params = self.tuner.default_classifier_params.copy() if hasattr(self.tuner, "default_classifier_params") else {}
        self.reg_params = self.tuner.default_regressor_params.copy() if hasattr(self.tuner, "default_regressor_params") else {}

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Hook for subclasses to transform the input matrix before initialization."""

        return X

    def _prepare_calibration_data(self, n_calibration: int) -> CalibrationBatch:
        """Build grouped calibration arrays from sampled features."""

        n_calibration = min(n_calibration, self.n_features)
        if n_calibration <= 0:
            raise ValueError("n_calibration must be >= 1")

        indices = self.rng.choice(self.n_features, n_calibration, replace=False)

        X_stack_clf: list[np.ndarray] = []
        y_stack_clf: list[np.ndarray] = []
        groups_clf: list[np.ndarray] = []
        X_stack_reg: list[np.ndarray] = []
        y_stack_reg: list[np.ndarray] = []
        groups_reg: list[np.ndarray] = []

        for idx in indices:
            q_vec = self.search_vectors[idx].reshape(1, -1)
            _, neighbor_ids = self.neighbor_index.search(q_vec, self.k + 1)
            neighbor_indices = [n for n in neighbor_ids[0] if n != idx][: self.k]

            if len(neighbor_indices) < self.k:
                continue

            X_j = self.raw_vectors[idx]
            S_matrix = self.raw_vectors[neighbor_indices].T

            is_nonzero = np.abs(X_j) > 0
            y_binary = is_nonzero.astype(int)

            X_stack_clf.append(S_matrix)
            y_stack_clf.append(y_binary)
            groups_clf.append(np.full(S_matrix.shape[0], idx, dtype=np.int32))

            if np.sum(is_nonzero) > 10:
                X_stack_reg.append(S_matrix[is_nonzero])
                y_stack_reg.append(X_j[is_nonzero])
                groups_reg.append(np.full(np.sum(is_nonzero), idx, dtype=np.int32))

        if not X_stack_clf:
            raise RuntimeError("Calibration failed: no valid classifier calibration batches were built.")

        X_clf_all = np.vstack(X_stack_clf)
        y_clf_all = np.concatenate(y_stack_clf)
        groups_clf_all = np.concatenate(groups_clf)

        if X_stack_reg:
            X_reg_all = np.vstack(X_stack_reg)
            y_reg_all = np.concatenate(y_stack_reg)
            groups_reg_all = np.concatenate(groups_reg)
        else:
            X_reg_all, y_reg_all, groups_reg_all = None, None, None

        return CalibrationBatch(
            X_clf=X_clf_all,
            y_clf=y_clf_all,
            groups_clf=groups_clf_all,
            X_reg=X_reg_all,
            y_reg=y_reg_all,
            groups_reg=groups_reg_all,
        )

    def calibrate(self, n_calibration: int = 50, n_trials: int = 20) -> None:
        """Calibrate support and value models using the configured tuner."""

        print(f"\n--- Starting calibration ({n_calibration} sampled features) ---")
        batch = self._prepare_calibration_data(n_calibration)
        result = self.tuner.tune(batch=batch, n_trials=n_trials, seed=int(self.rng.integers(1, 10_000)))
        self.clf_params = result.classifier_params
        self.reg_params = result.regressor_params

        print("\n--- Calibration Results ---")
        print(f"  > Best classifier score: {result.classifier_score:.4f}")
        print(f"  > Best regressor score:  {result.regressor_score:.4f}")

    def _log_status(self, feature_idx: int, step: str, status: str, n_nonzero: int) -> None:
        if status == "OK":
            return
        self.logs.append(
            {
                "feature_idx": feature_idx,
                "step": step,
                "status": status,
                "n_nonzero": n_nonzero,
            }
        )

    def _sample_feature(self, X_j_raw: np.ndarray, S_matrix_raw: np.ndarray, feature_idx: int) -> np.ndarray:
        """Generate a knockoff vector for a single feature."""

        n = len(X_j_raw)
        X_tilde = np.zeros(n, dtype=np.float32)

        is_nonzero_sim, support_status = self.distribution_learner.predict_support(
            X_j_raw,
            S_matrix_raw,
            classifier_params=self.clf_params,
            rng=self.rng,
        )
        self._log_status(feature_idx, "Support", support_status, int(np.sum(np.abs(X_j_raw) > 0)))

        if is_nonzero_sim is None:
            return X_tilde

        if np.any(is_nonzero_sim):
            filled, value_status = self.distribution_learner.predict_values(
                X_j_raw,
                S_matrix_raw,
                is_nonzero_sim,
                regressor_params=self.reg_params,
                rng=self.rng,
            )
            self._log_status(feature_idx, "Values", value_status, int(np.sum(np.abs(X_j_raw) > 0)))
            X_tilde[is_nonzero_sim] = filled.astype(np.float32, copy=False)

        return X_tilde

    def _prepare_result(self, raw_knockoffs: list[np.ndarray]) -> KnockoffOutputs:
        """Build final output object. Subclasses can override output shaping."""

        return KnockoffOutputs(
            X_transformed=self.X_processed,
            X_knockoff=np.array(raw_knockoffs, dtype=np.float32).T,
            logs=self.logs.copy(),
        )

    def generate(
        self,
        n_calibration: int = 500,
        n_trials: int = 50,
        tune: bool = True,
    ) -> KnockoffOutputs:
        """Generate knockoff matrix.

        Input:
        - n_calibration: number of sampled features for calibration.
        - n_trials: tuner trial budget.
        - tune: if False, uses current parameter dictionaries without tuning.

        Output:
        - KnockoffOutputs with transformed input and generated knockoffs.
        """

        if tune:
            self.calibrate(n_calibration=n_calibration, n_trials=n_trials)

        self.logs = []
        raw_knockoffs: list[np.ndarray] = []

        print(f"\nStarting knockoff generation for {self.n_features} features...")

        for j in range(self.n_features):
            q_vec = self.search_vectors[j].reshape(1, -1)
            X_j_raw = self.raw_vectors[j]
            _, neighbors = self.neighbor_index.search(q_vec, self.k + 1)
            neighbor_indices = [idx for idx in neighbors[0] if idx != j][: self.k]

            S_matrix_raw = np.empty((self.n_samples, self.k), dtype=np.float32)
            for col_i, idx in enumerate(neighbor_indices):
                if idx < self.n_features:
                    S_matrix_raw[:, col_i] = self.raw_vectors[idx]
                else:
                    S_matrix_raw[:, col_i] = raw_knockoffs[idx - self.n_features]

            X_tilde = self._sample_feature(X_j_raw, S_matrix_raw, feature_idx=j)
            raw_knockoffs.append(X_tilde)

            X_tilde_norm = normalize(X_tilde.reshape(1, -1), axis=1, norm="l2").astype(np.float32)
            self.neighbor_index.add(X_tilde_norm)

            if j % 5000 == 0 and j > 0:
                print(f"  Processed {j} features...")

        return self._prepare_result(raw_knockoffs)
