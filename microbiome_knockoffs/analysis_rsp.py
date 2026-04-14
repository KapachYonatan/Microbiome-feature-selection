from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from .contracts import RSPResult


def calculate_threshold(W: np.ndarray, fdr: float = 0.1, offset: int = 1) -> float:
    """Compute knockoff threshold for target FDR.

    Input:
    - W: ndarray shape (n_features,), knockoff statistics.

    Output:
    - scalar threshold. Returns inf when no valid threshold exists.
    """

    t_values = np.sort(np.unique(np.abs(W[W != 0])))

    for t in t_values:
        fp = int(np.sum(W <= -t))
        selected = int(np.sum(W >= t))
        estimated_fdr = float("inf") if selected == 0 else (offset + fp) / selected
        if estimated_fdr <= fdr:
            return float(t)

    return float("inf")


def select_features(W: np.ndarray, threshold: float) -> np.ndarray:
    """Return selected feature indices sorted by descending W score."""

    if threshold == float("inf"):
        return np.array([], dtype=np.int32)

    indices = np.where(W >= threshold)[0]
    # Stable ordering: descending W, then ascending feature index for ties.
    order = np.lexsort((indices, -W[indices]))
    sorted_indices = indices[order]
    return sorted_indices.astype(np.int32)


def _validate_selected_indices(W: np.ndarray, selected_indices: np.ndarray) -> None:
    """Validate index bounds and descending W order for selected features."""

    if selected_indices.size == 0:
        return

    n_features = W.shape[0]
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= n_features:
        raise ValueError("selected_indices contains out-of-range feature indices")

    W_selected = W[selected_indices]
    if np.any(W_selected[:-1] < W_selected[1:]):
        raise ValueError("selected_indices must be ordered by descending W")


def compute_knockoffs_statistic(
    X_std: np.ndarray,
    X_tilde_std: np.ndarray,
    y_vec: np.ndarray,
) -> np.ndarray:
    """Compute W = |T_j| - |T~_j| for binary labels."""

    mask_1 = y_vec == 1
    mask_0 = y_vec == 0

    Z = np.abs(X_std[mask_1].mean(axis=0) - X_std[mask_0].mean(axis=0))
    Z_tilde = np.abs(X_tilde_std[mask_1].mean(axis=0) - X_tilde_std[mask_0].mean(axis=0))
    return Z - Z_tilde


def calculate_real_rsp_statistics(
    X: np.ndarray,
    X_tilde: np.ndarray,
    y: np.ndarray,
    target_fdr: float = 0.1,
) -> dict:
    """Compute real-data knockoff selection statistics.

    Output dict keys:
    - W_real, selected_indices, RP, threshold
    """

    X_scaled = StandardScaler().fit_transform(X)
    X_tilde_scaled = StandardScaler().fit_transform(X_tilde)

    W_real = compute_knockoffs_statistic(X_scaled, X_tilde_scaled, y)
    threshold = calculate_threshold(W_real, fdr=target_fdr, offset=1)
    selected_indices = select_features(W_real, threshold)
    _validate_selected_indices(W_real, selected_indices)

    return {
        "W_real": W_real,
        "selected_indices": selected_indices,
        "RP": int(len(selected_indices)),
        "threshold": float(threshold),
    }


def calculate_shuffled_rsp_statistics(
    X: np.ndarray,
    X_tilde: np.ndarray,
    y: np.ndarray,
    RP: int,
    target_fdr: float = 0.1,
    num_shuffles: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    """Estimate shuffled positives and build the RSP curve.

    Output dict keys:
    - SP, rsp, beta_values
    """

    X_scaled = StandardScaler().fit_transform(X)
    X_tilde_scaled = StandardScaler().fit_transform(X_tilde)
    rng = rng or np.random.default_rng(0)

    sp_counts: list[int] = []

    for _ in range(num_shuffles):
        y_perm = rng.permutation(y)
        W_shuf = compute_knockoffs_statistic(X_scaled, X_tilde_scaled, y_perm)
        threshold = calculate_threshold(W_shuf, fdr=target_fdr, offset=1)
        sp_counts.append(int(len(select_features(W_shuf, threshold))))

    SP = float(np.mean(sp_counts))

    beta_values = np.linspace(0, 1, 100)
    numerator = beta_values * RP - SP
    denominator = beta_values * RP + SP

    rsp = np.zeros_like(beta_values)
    valid = denominator > 0
    rsp[valid] = numerator[valid] / denominator[valid]

    return {
        "SP": SP,
        "rsp": rsp,
        "beta_values": beta_values,
    }


def calculate_and_plot_rsp(
    X: np.ndarray,
    X_tilde: np.ndarray,
    y: np.ndarray,
    target_fdr: float = 0.05,
    num_shuffles: int = 10,
    save_path: str | None = None,
    rng: np.random.Generator | None = None,
) -> RSPResult:
    """Compute real and shuffled knockoff statistics and save RSP plot.

    Input:
    - X and X_tilde arrays of shape (n_samples, n_features).
    - y label vector of shape (n_samples,).

    Output:
    - RSPResult dataclass containing vectors and summary metrics.
    """

    real_stats = calculate_real_rsp_statistics(X, X_tilde, y, target_fdr=target_fdr)
    shuffled_stats = calculate_shuffled_rsp_statistics(
        X,
        X_tilde,
        y,
        RP=real_stats["RP"],
        target_fdr=target_fdr,
        num_shuffles=num_shuffles,
        rng=rng,
    )

    plt.figure(figsize=(9, 6))
    plt.plot(shuffled_stats["beta_values"], shuffled_stats["rsp"], linewidth=2, color="tab:green")
    plt.title(f"RSP: knockoff W (FDR={target_fdr})")
    plt.xlabel("$\\beta$ weighting factor")
    plt.ylabel("RSP Score")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.ylim(-1.1, 1.1)
    plt.text(
        0.05,
        0.9,
        f"RP={real_stats['RP']}\\nSP={shuffled_stats['SP']:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    return RSPResult(
        W_real=real_stats["W_real"],
        selected_indices=real_stats["selected_indices"],
        rsp=shuffled_stats["rsp"],
        beta_values=shuffled_stats["beta_values"],
        RP=real_stats["RP"],
        SP=shuffled_stats["SP"],
        threshold=real_stats["threshold"],
    )
