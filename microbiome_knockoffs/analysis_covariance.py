from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def exact_covariance_comparison(
    X_orig: np.ndarray,
    X_knockoff: np.ndarray,
    n_anchors: int = 100,
    k_pairs: int = 100,
) -> tuple[list[float], list[float]]:
    """Sample feature pairs and compute exact covariance in original and knockoff matrices.

    Input:
    - X_orig: ndarray shape (n_samples, n_features).
    - X_knockoff: ndarray shape (n_samples, n_features).

    Output:
    - cov_original: list[float] covariances from original data.
    - cov_knockoff: list[float] covariances from knockoff data.
    """

    n_samples, n_features = X_orig.shape
    X_std = StandardScaler().fit_transform(X_orig)

    cov_original: list[float] = []
    cov_knockoff: list[float] = []

    anchor_indices = np.random.choice(n_features, n_anchors, replace=False)

    for idx_anchor in anchor_indices:
        vec_anchor = X_std[:, idx_anchor]

        batch_size = min(5000, n_features)
        search_batch_idx = np.random.choice(n_features, batch_size, replace=False)
        batch_vectors = X_std[:, search_batch_idx]

        corrs = np.dot(batch_vectors.T, vec_anchor) / n_samples
        top_k = min(k_pairs + 1, len(corrs))
        top_indices_local = np.argsort(np.abs(corrs))[-top_k:]
        targets = search_batch_idx[top_indices_local]

        for idx_target in targets:
            if idx_target == idx_anchor:
                continue

            c_orig = np.cov(X_orig[:, idx_anchor], X_orig[:, idx_target])[0, 1]
            c_knock = np.cov(X_knockoff[:, idx_anchor], X_knockoff[:, idx_target])[0, 1]
            cov_original.append(float(c_orig))
            cov_knockoff.append(float(c_knock))

    return cov_original, cov_knockoff


def plot_cov_preservation(
    X_orig: np.ndarray,
    X_knockoff: np.ndarray,
    title_suffix: str = "",
    save_path: str | None = None,
) -> float:
    """Generate covariance preservation scatter plot and return preservation correlation.

    Output:
    - Pearson correlation between sampled original and knockoff covariance values.
    """

    orig_covs, knock_covs = exact_covariance_comparison(
        X_orig,
        X_knockoff,
        n_anchors=100,
        k_pairs=100,
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=orig_covs, y=knock_covs, alpha=0.5, s=20, color="teal")

    min_val = min(min(orig_covs), min(knock_covs))
    max_val = max(max(orig_covs), max(knock_covs))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Match (y=x)")

    plt.xlabel("Covariance (Original Data)")
    plt.ylabel("Covariance (Knockoff Data)")
    plt.title(f"Covariance Preservation {title_suffix}".strip())
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    r_preservation = float(np.corrcoef(orig_covs, knock_covs)[0, 1])
    print(f"Overall Preservation Score: {r_preservation:.4f}")
    return r_preservation
