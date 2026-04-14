from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import CalibrationBatch, TuningResult


class HyperparameterTuner(ABC):
    """Interface for hyperparameter tuning backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tuning backend name for metadata."""

    @abstractmethod
    def tune(self, batch: CalibrationBatch, n_trials: int, seed: int = 42) -> TuningResult:
        """Tune support and value model parameters.

        Input:
        - batch: grouped calibration arrays.
        - n_trials: optimization trial budget.

        Output:
        - TuningResult with classifier/regressor params and summary scores.
        """
