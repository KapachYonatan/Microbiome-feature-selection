#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from microbiome_knockoffs.evaluation_classifier_comparison import (  # noqa: E402
    _ordered_knockoff_selected_indices,
    build_k_grid,
    classifier_registry,
)


def _predict_scores(model: object, X: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.asarray(probs[:, 1], dtype=float)
        return np.asarray(probs.ravel(), dtype=float)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)

    return np.asarray(model.predict(X), dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose knockoff top-K ordering and plot ROC AUC vs K using only knockoff features."
        )
    )
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument("--study", required=True, help="Study name")
    parser.add_argument("--run-folder", required=True, help="Run folder under study/runs")
    parser.add_argument("--classifier", default="lgbm", choices=["lgbm", "logreg_l2"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-grid-points", type=int, default=20)
    parser.add_argument("--k-start", type=int, default=1)
    parser.add_argument("--k-end", type=int, default=5000)
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional output path for ROC AUC vs K plot",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional CSV output with columns K and roc_auc",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    study_dir = Path(args.base_dir) / args.study
    run_dir = study_dir / "runs" / args.run_folder

    x_filtered_path = run_dir / "X_filtered.npz"
    y_path = study_dir / "y_clean.npy"
    rsp_path = run_dir / "rsp_results.npy"

    required = [x_filtered_path, y_path, rsp_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    X_filtered = sparse.load_npz(x_filtered_path).tocsr().astype(np.float32, copy=False)
    y = np.load(y_path)
    rsp_results = np.load(rsp_path, allow_pickle=True).item()

    W_real = np.asarray(rsp_results["W_real"], dtype=float)
    saved_selected = np.asarray(rsp_results.get("selected_indices", []), dtype=int)
    if saved_selected.size == 0:
        raise ValueError("rsp_results.npy has no selected_indices")

    expected_desc = saved_selected[np.argsort(W_real[saved_selected])[::-1]]
    saved_desc = bool(np.array_equal(saved_selected, expected_desc))

    comparison_selected = _ordered_knockoff_selected_indices(rsp_results)
    comparison_desc = bool(np.array_equal(comparison_selected, expected_desc))

    if X_filtered.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X_filtered rows={X_filtered.shape[0]} y={y.shape[0]}")

    idx_all = np.arange(y.shape[0])
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    y_train = y[idx_train]
    y_test = y[idx_test]

    k_capacity = int(comparison_selected.shape[0])
    k_values = build_k_grid(args.k_start, args.k_end, k_capacity, args.k_grid_points)

    classifiers = classifier_registry()
    classifier_factory = classifiers[args.classifier]

    def _auc_curve(ordered_indices: np.ndarray) -> list[float]:
        curve: list[float] = []
        for K in k_values:
            feature_indices = ordered_indices[:K]
            X_train = X_filtered[idx_train][:, feature_indices]
            X_test = X_filtered[idx_test][:, feature_indices]

            model = classifier_factory(args.random_state)
            model.fit(X_train, y_train)
            y_score = _predict_scores(model, X_test)
            curve.append(float(roc_auc_score(y_test, y_score)))

            del model, X_train, X_test, y_score, feature_indices
        return curve

    desc_scores = _auc_curve(comparison_selected)
    asc_scores = _auc_curve(comparison_selected[::-1])

    plot_path = Path(args.plot_path) if args.plot_path else (run_dir / "knockoff_topk_auc_vs_k_diagnosis.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.plot(
        k_values,
        desc_scores,
        marker="o",
        linewidth=2,
        color="tab:green",
        label="Knockoff Top-K (descending)",
    )
    plt.plot(
        k_values,
        asc_scores,
        marker="o",
        linewidth=2,
        color="tab:red",
        linestyle="--",
        label="Knockoff Top-K (ascending)",
    )
    plt.title(f"Knockoff Top-K ROC AUC vs K Order Diagnosis ({args.study})")
    plt.xlabel("K")
    plt.ylabel("ROC AUC")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.text(
        0.02,
        0.05,
        (
            f"saved_desc={saved_desc}\n"
            f"comparison_desc={comparison_desc}\n"
            f"selected={comparison_selected.size}"
        ),
        transform=plt.gca().transAxes,
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("K,roc_auc_descending,roc_auc_ascending\n")
            for K, score_desc, score_asc in zip(k_values, desc_scores, asc_scores):
                handle.write(f"{K},{score_desc:.8f},{score_asc:.8f}\n")

    print(f"saved_selected_descending={saved_desc}")
    print(f"comparison_selected_descending={comparison_desc}")
    print(f"k_values={k_values}")
    print(f"descending_final_auc={desc_scores[-1]:.6f}")
    print(f"ascending_final_auc={asc_scores[-1]:.6f}")
    print(f"plot_path={plot_path}")


if __name__ == "__main__":
    main()