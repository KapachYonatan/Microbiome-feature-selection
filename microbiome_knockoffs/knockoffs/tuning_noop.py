from __future__ import annotations

from ..contracts import CalibrationBatch, TuningResult
from .tuning_interface import HyperparameterTuner


class NoOpTuner(HyperparameterTuner):
    """Return fixed/default parameters without optimization."""

    def __init__(
        self,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
    ) -> None:
        self.default_classifier_params = classifier_params or {
            "objective": "binary",
            "max_depth": 5,
            "num_leaves": 15,
            "is_unbalance": True,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
        }
        self.default_regressor_params = regressor_params or {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 10,
            "min_child_samples": 5,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
        }

    @property
    def name(self) -> str:
        return "NoOpTuner"

    def tune(self, batch: CalibrationBatch, n_trials: int, seed: int = 42) -> TuningResult:
        _ = batch, n_trials, seed
        return TuningResult(
            classifier_params=self.default_classifier_params.copy(),
            regressor_params=self.default_regressor_params.copy(),
            classifier_score=float("nan"),
            regressor_score=float("nan"),
        )
