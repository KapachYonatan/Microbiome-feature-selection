from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microbiome_knockoffs.contracts import CalibrationBatch, RunConfig, validate_run_config
from microbiome_knockoffs.knockoffs.tuning_noop import NoOpTuner


def test_validate_run_config_accepts_flat_no_optuna():
    config = RunConfig(
        study_name="demo",
        base_dir=Path("/tmp"),
        deterministic_mode=True,
        faiss_mode="flat",
        faiss_threads=1,
        use_optuna_tuning=False,
    )
    validate_run_config(config)


def test_validate_run_config_rejects_invalid_faiss_mode():
    config = RunConfig(
        study_name="demo",
        base_dir=Path("/tmp"),
        faiss_mode="invalid",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="Invalid faiss_mode"):
        validate_run_config(config)


def test_noop_tuner_returns_provided_params():
    clf = {"n_estimators": 17, "max_depth": 3, "objective": "binary", "n_jobs": 1}
    reg = {"n_estimators": 21, "max_depth": 4, "objective": "regression", "n_jobs": 1}

    tuner = NoOpTuner(classifier_params=clf, regressor_params=reg)
    batch = CalibrationBatch(
        X_clf=np.zeros((1, 1), dtype=float),
        y_clf=np.zeros((1,), dtype=int),
        groups_clf=np.zeros((1,), dtype=int),
        X_reg=np.zeros((1, 1), dtype=float),
        y_reg=np.zeros((1,), dtype=float),
        groups_reg=np.zeros((1,), dtype=int),
    )

    result = tuner.tune(batch=batch, n_trials=0, seed=42)
    assert result.classifier_params["n_estimators"] == 17
    assert result.regressor_params["n_estimators"] == 21


def test_noop_tuner_default_params_match_expected_configuration():
    tuner = NoOpTuner()
    batch = CalibrationBatch(
        X_clf=np.zeros((1, 1), dtype=float),
        y_clf=np.zeros((1,), dtype=int),
        groups_clf=np.zeros((1,), dtype=int),
        X_reg=np.zeros((1, 1), dtype=float),
        y_reg=np.zeros((1,), dtype=float),
        groups_reg=np.zeros((1,), dtype=int),
    )

    result = tuner.tune(batch=batch, n_trials=0, seed=42)

    assert result.classifier_params == {
        "objective": "binary",
        "max_depth": 5,
        "num_leaves": 15,
        "is_unbalance": True,
        "learning_rate": 0.05,
        "verbose": -1,
        "n_jobs": 1,
    }
    assert result.regressor_params == {
        "objective": "regression",
        "max_depth": 4,
        "num_leaves": 10,
        "min_child_samples": 5,
        "learning_rate": 0.05,
        "verbose": -1,
        "n_jobs": 1,
    }
