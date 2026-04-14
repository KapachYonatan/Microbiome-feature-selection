from __future__ import annotations

import warnings

import numpy as np
import lightgbm as lgb

from .distribution_interface import DistributionLearner


# LightGBM sklearn wrappers auto-attach synthetic feature names on fit with ndarray.
# Predicting again with ndarray then emits noisy sklearn validation warnings.
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


class HurdleLGBMDistribution(DistributionLearner):
    """Default hurdle distribution learner using LGBM support/value models."""

    @property
    def name(self) -> str:
        return "HurdleLGBMDistribution"

    def predict_support(
        self,
        X_j_raw: np.ndarray,
        S_matrix_raw: np.ndarray,
        classifier_params: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray | None, str]:
        n = len(X_j_raw)
        is_nonzero_true = np.abs(X_j_raw) > 0
        y_binary = is_nonzero_true.astype(int)
        n_nonzero = int(np.sum(y_binary))
        status = "OK"

        if n_nonzero == 0:
            return None, status
        if n_nonzero == n:
            return np.ones(n, dtype=bool), status

        clf = lgb.LGBMClassifier(**classifier_params)
        try:
            clf.fit(S_matrix_raw, y_binary)
            probs = clf.predict_proba(S_matrix_raw)[:, 1]
            is_nonzero_sim = rng.random(n) < probs
        except Exception as exc:
            status = f"Fallback: Classifier Exception ({exc})"
            is_nonzero_sim = rng.random(n) < (n_nonzero / n)

        return is_nonzero_sim, status

    def predict_values(
        self,
        X_j_raw: np.ndarray,
        S_matrix_raw: np.ndarray,
        is_nonzero_sim: np.ndarray,
        regressor_params: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        is_nonzero_true = np.abs(X_j_raw) > 0
        n_nonzero = int(np.sum(is_nonzero_true))
        n_sim_nonzero = int(np.sum(is_nonzero_sim))
        values = np.zeros(n_sim_nonzero, dtype=np.float32)
        status = "OK"

        if n_sim_nonzero == 0:
            return values, status

        min_samples = int(regressor_params.get("min_child_samples", 10))

        if n_nonzero <= min_samples:
            status = f"Fallback: Too few samples for regression ({n_nonzero} <= {min_samples})"
            empirical = X_j_raw[is_nonzero_true]
            empirical = empirical[empirical > 0]

            if len(empirical) > 0:
                log_vals = np.log(empirical)
                mu = float(np.mean(log_vals))
                sigma = float(np.std(log_vals))
                noise = rng.normal(mu, sigma, size=n_sim_nonzero)
                values = np.exp(noise).astype(np.float32)

            return values, status

        local_params = regressor_params.copy()
        if n_nonzero < 2 * min_samples:
            new_min_child = max(1, int(n_nonzero / 2))
            local_params["min_child_samples"] = new_min_child
            status = (
                "Adjusted: LGBM min_child_samples lowered "
                f"to {new_min_child} (n_nonzero={n_nonzero})"
            )

        try:
            reg = lgb.LGBMRegressor(**local_params)
            X_train = S_matrix_raw[is_nonzero_true]
            y_train = X_j_raw[is_nonzero_true]
            y_train = y_train[y_train > 0]

            if len(y_train) == 0:
                return values, "Fallback: No positive non-zero values available"

            reg.fit(X_train[: len(y_train)], np.log(y_train))

            X_test = S_matrix_raw[is_nonzero_sim]
            mu_hat_log = reg.predict(X_test)

            preds_train_log = reg.predict(X_train[: len(y_train)])
            residuals_log = np.log(y_train) - preds_train_log
            std_dev_log = float(np.std(residuals_log))

            noise = rng.normal(0, std_dev_log, size=len(mu_hat_log))
            values = np.exp(mu_hat_log + noise).astype(np.float32)

        except Exception as exc:
            status = f"Fallback: Regressor Exception ({exc})"
            empirical = X_j_raw[is_nonzero_true]
            empirical = empirical[empirical > 0]
            if len(empirical) > 0:
                log_vals = np.log(empirical)
                mu = float(np.mean(log_vals))
                sigma = float(np.std(log_vals))
                noise = rng.normal(mu, sigma, size=n_sim_nonzero)
                values = np.exp(noise).astype(np.float32)

        return values, status
