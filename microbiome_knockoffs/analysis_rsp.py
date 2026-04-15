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


def build_feature_index_map(
    W: np.ndarray,
    is_significant: np.ndarray,
) -> dict[int, tuple[float, bool]]:
    """Build sorted feature map: feature_index -> (W_real, is_significant).

    The mapping iteration order is deterministic and sorted by:
    1) descending W_real, 2) ascending feature index for ties.
    """

    if W.ndim != 1:
        raise ValueError("W must be a 1D array")
    if is_significant.ndim != 1:
        raise ValueError("is_significant must be a 1D array")
    if W.shape[0] != is_significant.shape[0]:
        raise ValueError("W and is_significant must have the same length")

    n_features = int(W.shape[0])
    all_indices = np.arange(n_features, dtype=np.int32)
    order = np.lexsort((all_indices, -W))
    ordered_indices = all_indices[order]

    return {
        int(idx): (float(W[idx]), bool(is_significant[idx]))
        for idx in ordered_indices
    }


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
    - RP, threshold, feature_index_map
    """

    X_scaled = StandardScaler().fit_transform(X)
    X_tilde_scaled = StandardScaler().fit_transform(X_tilde)

    W_real = compute_knockoffs_statistic(X_scaled, X_tilde_scaled, y)
    threshold = calculate_threshold(W_real, fdr=target_fdr, offset=1)
    is_significant = W_real >= threshold
    feature_index_map = build_feature_index_map(W_real, is_significant)

    return {
        "RP": int(np.sum(is_significant)),
        "threshold": float(threshold),
        "feature_index_map": feature_index_map,
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
        sp_counts.append(int(np.sum(W_shuf >= threshold)))

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
        feature_index_map=real_stats["feature_index_map"],
        rsp=shuffled_stats["rsp"],
        beta_values=shuffled_stats["beta_values"],
        RP=real_stats["RP"],
        SP=shuffled_stats["SP"],
        threshold=real_stats["threshold"],
    )
