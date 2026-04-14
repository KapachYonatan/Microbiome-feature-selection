from __future__ import annotations

import numpy as np

from microbiome_knockoffs.contracts import CalibrationBatch, TuningResult
from microbiome_knockoffs.knockoffs.distribution_interface import DistributionLearner
from microbiome_knockoffs.knockoffs.generators_base import BaseKnockoffGenerator
from microbiome_knockoffs.knockoffs.neighbor_index_interface import NeighborIndex
from microbiome_knockoffs.knockoffs.tuning_interface import HyperparameterTuner


class DummyNeighborIndex(NeighborIndex):
    def __init__(self):
        self.vectors = None

    @property
    def name(self) -> str:
        return "DummyNeighborIndex"

    def fit(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.copy()

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self.vectors.T
        order = np.argsort(scores[0])[::-1][:k]
        return scores[:, order], np.expand_dims(order, axis=0)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors])


class DummyDistribution(DistributionLearner):
    @property
    def name(self) -> str:
        return "DummyDistribution"

    def predict_support(self, X_j_raw, S_matrix_raw, classifier_params, rng):
        return (np.abs(X_j_raw) > 0), "OK"

    def predict_values(self, X_j_raw, S_matrix_raw, is_nonzero_sim, regressor_params, rng):
        return np.ones(int(np.sum(is_nonzero_sim)), dtype=np.float32), "OK"


class DummyTuner(HyperparameterTuner):
    @property
    def name(self) -> str:
        return "DummyTuner"

    def tune(self, batch: CalibrationBatch, n_trials: int, seed: int = 42) -> TuningResult:
        return TuningResult(
            classifier_params={"dummy": True},
            regressor_params={"dummy": True},
            classifier_score=0.5,
            regressor_score=-0.5,
        )


def test_base_generator_dependency_injection_and_shapes():
    X = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.0, 0.2, 1.2, 0.0],
            [1.0, 0.0, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    generator = BaseKnockoffGenerator(
        X=X,
        k_neighbors=2,
        distribution_learner=DummyDistribution(),
        tuner=DummyTuner(),
        neighbor_index=DummyNeighborIndex(),
        random_seed=7,
    )

    out = generator.generate(tune=False)

    assert out.X_transformed.shape == X.shape
    assert out.X_knockoff.shape == X.shape
    assert np.all(out.X_knockoff >= 0)
