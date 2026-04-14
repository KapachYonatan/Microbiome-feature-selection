from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


def merge_images_2x2(image_paths: list[Path], output_path: Path) -> Path:
    """Merge exactly four images into a 2x2 grid and save output image.

    Input:
    - image_paths: list of exactly 4 existing image paths.

    Output:
    - output_path where merged image is written.
    """

    if len(image_paths) != 4:
        raise ValueError("merge_images_2x2 requires exactly four image paths")

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes_flat = axes.flatten()

    for idx, image_path in enumerate(image_paths):
        image = mpimg.imread(image_path)
        axes_flat[idx].imshow(image)
        axes_flat[idx].axis("off")

    plt.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return output_path


def _latest_run_with_file(study_dir: Path, filename: str) -> Path | None:
    runs_dir = study_dir / "runs"
    if not runs_dir.is_dir():
        return None

    candidates = sorted(
        [run for run in runs_dir.iterdir() if run.is_dir() and (run / filename).exists()],
        key=lambda run: run.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_run_file(
    base_dir: Path,
    study: str,
    filename: str,
    run_folder: str | None,
) -> Path:
    """Resolve artifact path from explicit run folder, study root, or latest matching run."""

    study_dir = base_dir / study

    if run_folder:
        candidate = study_dir / "runs" / run_folder / filename
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Expected file in run folder not found: {candidate}")

    study_level = study_dir / filename
    if study_level.exists():
        return study_level

    latest_run = _latest_run_with_file(study_dir, filename)
    if latest_run is None:
        raise FileNotFoundError(f"Could not resolve {filename} for study={study}")

    return latest_run / filename


def load_rsp_results(
    base_dir: Path,
    studies: list[str],
    run_map: dict[str, str],
) -> dict[str, dict]:
    """Load RSP result dictionaries per study."""

    results: dict[str, dict] = {}
    for study in studies:
        rsp_path = resolve_run_file(base_dir, study, "rsp_results.npy", run_map.get(study))
        results[study] = np.load(rsp_path, allow_pickle=True).item()
    return results


def compute_bh_rsp_curve(
    X: np.ndarray,
    y: np.ndarray,
    betas: np.ndarray,
    target_fdr: float,
    n_shuffles: int,
    rng: np.random.Generator,
) -> list[float]:
    """Compute naive BH RSP curve with fixed FDR.

    Input:
    - X: ndarray shape (n_samples, n_features).
    - y: ndarray shape (n_samples,), binary labels.
    - betas: ndarray shape (n_points,), beta axis for RSP.

    Output:
    - list[float] RSP values aligned to betas.
    """

    _, pvals_real = mannwhitneyu(X[y == 1], X[y == 0], axis=0)
    pvals_real = np.nan_to_num(pvals_real, nan=1.0)
    rejected, _ = fdrcorrection(pvals_real, alpha=target_fdr)
    rp = int(np.sum(rejected))

    shuffled_discoveries = []
    for _ in range(n_shuffles):
        y_shuff = rng.permutation(y)
        _, pvals_shuff = mannwhitneyu(X[y_shuff == 1], X[y_shuff == 0], axis=0)
        pvals_shuff = np.nan_to_num(pvals_shuff, nan=1.0)
        rejected_shuff, _ = fdrcorrection(pvals_shuff, alpha=target_fdr)
        shuffled_discoveries.append(int(np.sum(rejected_shuff)))

    sp = float(np.mean(shuffled_discoveries))

    bh_rsp_vals: list[float] = []
    for beta in betas:
        denominator = (beta * rp) + sp
        if denominator == 0:
            bh_rsp_vals.append(0.0)
        else:
            bh_rsp_vals.append(max(0.0, ((beta * rp) - sp) / denominator))

    return bh_rsp_vals


def _load_cache(cache_file: Path) -> dict[str, list[float]]:
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _save_cache(cache_file: Path, cache: dict[str, list[float]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle)


def plot_rsp_grid(
    base_dir: Path,
    studies: list[str],
    run_map: dict[str, str],
    cache_file: Path,
    save_path: Path,
    target_fdr: float = 0.05,
    n_shuffles: int = 20,
    seed: int = 42,
) -> Path:
    """Generate study-wise RSP comparison grid (Knockoff vs Naive BH).

    Output:
    - save_path where the grid image is written.
    """

    rsp_results = load_rsp_results(base_dir, studies, run_map)
    bh_cache = _load_cache(cache_file)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for idx, study in enumerate(studies[:4]):
        results = rsp_results[study]
        ax = axes_flat[idx]

        betas = np.asarray(results["beta_values"])[1:]
        knockoff_rsp = np.asarray(results["rsp"])[1:]

        ax.plot(betas, knockoff_rsp, label="Model-X Knockoff", color="#1f77b4", linestyle="-", linewidth=2.5)

        if study in bh_cache:
            bh_rsp_vals = bh_cache[study]
        else:
            x_binary_path = resolve_run_file(base_dir, study, "X_binary.npy", run_map.get(study))
            y_path = base_dir / study / "y_clean.npy"
            if not y_path.exists():
                raise FileNotFoundError(f"Missing labels file: {y_path}")

            X = np.load(x_binary_path)
            y = np.load(y_path)

            bh_rsp_vals = compute_bh_rsp_curve(
                X=X,
                y=y,
                betas=betas,
                target_fdr=target_fdr,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            bh_cache[study] = bh_rsp_vals
            _save_cache(cache_file, bh_cache)

        ax.plot(
            betas,
            bh_rsp_vals,
            label=f"Naive BH (FDR={target_fdr})",
            color="#d62728",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
        )

        ax.set_title(study, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if idx >= 2:
            ax.set_xlabel("Beta Value", fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel("RSP", fontsize=12)

    for idx in range(len(studies), 4):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path
