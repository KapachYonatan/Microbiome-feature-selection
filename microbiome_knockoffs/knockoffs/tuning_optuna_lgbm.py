from __future__ import annotations

import warnings

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from ..contracts import CalibrationBatch, TuningResult
from .tuning_interface import HyperparameterTuner

optuna.logging.set_verbosity(optuna.logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


class OptunaLGBMTuner(HyperparameterTuner):
    """Optuna + GroupKFold tuner for support and value models."""

    def __init__(
        self,
        default_classifier_params: dict | None = None,
        default_regressor_params: dict | None = None,
    ) -> None:
        self.default_classifier_params = default_classifier_params or {
            "n_estimators": 30,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbose": -1,
            "objective": "binary",
            "n_jobs": 1,
        }
        self.default_regressor_params = default_regressor_params or {
            "n_estimators": 30,
            "max_depth": 2,
            "learning_rate": 0.1,
            "verbose": -1,
            "objective": "regression",
            "n_jobs": 1,
        }

    @property
    def name(self) -> str:
        return "OptunaLGBMTuner"

    def tune(self, batch: CalibrationBatch, n_trials: int, seed: int = 42) -> TuningResult:
        clf_params, clf_score = self._tune_classifier(
            batch.X_clf,
            batch.y_clf,
            batch.groups_clf,
            n_trials,
            seed,
        )
        reg_params, reg_score = self._tune_regressor(
            batch.X_reg,
            batch.y_reg,
            batch.groups_reg,
            n_trials,
            seed,
        )

        return TuningResult(
            classifier_params=clf_params,
            regressor_params=reg_params,
            classifier_score=clf_score,
            regressor_score=reg_score,
        )

    def _tune_classifier(
        self,
        X_clf_all: np.ndarray,
        y_clf_all: np.ndarray,
        groups_clf_all: np.ndarray,
        n_trials: int,
        seed: int,
    ) -> tuple[dict, float]:
        n_unique_groups = len(np.unique(groups_clf_all))
        n_splits = min(5, n_unique_groups)

        if n_splits < 2 or len(np.unique(y_clf_all)) < 2:
            print("[WARNING] Insufficient grouped variability for classifier tuning.")
            return self.default_classifier_params.copy(), float("nan")

        pos_count = int(np.sum(y_clf_all == 1))
        neg_count = int(np.sum(y_clf_all == 0))
        scale_pos_weight = max(1.0, neg_count / max(1, pos_count))

        sample_size = X_clf_all.shape[0]
        if sample_size < 60000:
            n_estimators_low, n_estimators_high = 40, 180
            min_child_high = 80
        else:
            n_estimators_low, n_estimators_high = 80, 300
            min_child_high = 150

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, min_child_high),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
                "n_jobs": 1,
                "verbose": -1,
                "objective": "binary",
                "scale_pos_weight": scale_pos_weight,
            }

            fold_scores: list[float] = []
            cv = GroupKFold(n_splits=n_splits)

            for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_clf_all, y_clf_all, groups=groups_clf_all)):
                X_tr, X_va = X_clf_all[tr_idx], X_clf_all[va_idx]
                y_tr, y_va = y_clf_all[tr_idx], y_clf_all[va_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    continue

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                preds = model.predict_proba(X_va)[:, 1]

                pr_auc = average_precision_score(y_va, preds)
                roc_auc = roc_auc_score(y_va, preds)
                fold_scores.append(0.7 * pr_auc + 0.3 * roc_auc)

                trial.report(float(np.median(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not fold_scores:
                return -1.0

            return float(np.median(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials)

        tuned = study.best_params
        tuned.update(
            {
                "n_jobs": 1,
                "verbose": -1,
                "objective": "binary",
                "scale_pos_weight": scale_pos_weight,
            }
        )
        return tuned, float(study.best_value)

    def _tune_regressor(
        self,
        X_reg_all: np.ndarray | None,
        y_reg_all: np.ndarray | None,
        groups_reg_all: np.ndarray | None,
        n_trials: int,
        seed: int,
    ) -> tuple[dict, float]:
        if X_reg_all is None or y_reg_all is None or groups_reg_all is None:
            print("[WARNING] Not enough non-zero data for regression tuning.")
            return self.default_regressor_params.copy(), float("nan")

        n_unique_groups = len(np.unique(groups_reg_all))
        n_splits = min(5, n_unique_groups)
        if n_splits < 2:
            print("[WARNING] Insufficient grouped variability for regression tuning.")
            return self.default_regressor_params.copy(), float("nan")

        sample_size = X_reg_all.shape[0]
        if sample_size < 30000:
            n_estimators_low, n_estimators_high = 50, 200
            min_child_high = 60
        else:
            n_estimators_low, n_estimators_high = 100, 350
            min_child_high = 120

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, min_child_high),
                "n_estimators": trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
                "n_jobs": 1,
                "verbose": -1,
                "objective": "regression",
            }

            fold_scores: list[float] = []
            cv = GroupKFold(n_splits=n_splits)

            for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_reg_all, y_reg_all, groups=groups_reg_all)):
                X_tr, X_va = X_reg_all[tr_idx], X_reg_all[va_idx]
                y_tr, y_va = y_reg_all[tr_idx], y_reg_all[va_idx]

                if len(y_tr) < 10 or len(y_va) < 10:
                    continue

                model = lgb.LGBMRegressor(**params)
                y_tr_log = np.log1p(y_tr)
                y_va_log = np.log1p(y_va)

                model.fit(
                    X_tr,
                    y_tr_log,
                    eval_set=[(X_va, y_va_log)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )

                preds_log = model.predict(X_va)
                fold_scores.append(-mean_absolute_error(y_va_log, preds_log))

                trial.report(float(np.median(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not fold_scores:
                return -1e9

            return float(np.median(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=n_trials)

        tuned = study.best_params
        tuned.update({"n_jobs": 1, "verbose": -1, "objective": "regression"})
        return tuned, float(study.best_value)
