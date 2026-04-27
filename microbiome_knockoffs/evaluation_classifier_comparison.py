from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import fdrcorrection

from .preprocessing_gene_abundance import load_matrix_and_genes


ClassifierFactory = Callable[[int], Any]


@dataclass(frozen=True)
class ClassifierComparisonConfig:
    """Configuration for knockoff-vs-baseline classifier comparison."""

    base_dir: Path
    study_name: str
    run_folder: str
    random_state: int = 42
    test_size: float = 0.2
    k_min: int = 40
    k_max: int = 1000
    k_grid_points: int = 10
    random_trials: int = 10

    @property
    def study_dir(self) -> Path:
        return self.base_dir / self.study_name

    @property
    def run_dir(self) -> Path:
        return self.study_dir / "runs" / self.run_folder


@dataclass(frozen=True)
class ComparisonData:
    """Container with all matrices needed for classifier comparison."""

    X_raw: sparse.csr_matrix
    X_clean: sparse.csr_matrix
    X_filtered: sparse.csr_matrix
    y: np.ndarray
    genes_raw: np.ndarray
    genes_clean: np.ndarray | None
    genes_filtered: np.ndarray
    X_bacteria_source: sparse.csr_matrix
    genes_bacteria_source: np.ndarray
    bacteria_source_tag: str
    knockoff_ranked_indices: np.ndarray


@dataclass(frozen=True)
class SelectionMethod:
    """Feature selection method descriptor.

    kind: "bh" (BH-ranked) or "knockoff" (ranked by knockoff W-statistic).
    source: one of raw, clean, filtered, bacteria, gene_clustered.
    """

    key: str
    label: str
    kind: str
    source: str


def classifier_registry() -> dict[str, ClassifierFactory]:
    return {
        "lgbm": lambda rs: lgb.LGBMClassifier(
            random_state=rs, verbose=-1, n_estimators=300, learning_rate=0.05, min_child_samples=5
        ),
        "logreg_l2": lambda rs: LogisticRegression(random_state=rs, solver="liblinear", max_iter=2000),
    }


def method_registry() -> dict[str, SelectionMethod]:
    return {
        "bh_raw": SelectionMethod(key="bh_raw", label="BH Top-K (Raw MTX)", kind="bh", source="raw"),
        "bh_clean": SelectionMethod(key="bh_clean", label="BH Top-K (X_clean)", kind="bh", source="clean"),
        "bh_filtered": SelectionMethod(key="bh_filtered", label="BH Top-K (X_filtered)", kind="bh", source="filtered"),
        "knockoff_topk": SelectionMethod(key="knockoff_topk", label="Knockoff Top-K", kind="knockoff", source="filtered"),
        "bacteria_bh": SelectionMethod(key="bacteria_bh", label="BH Top-K (Bacteria)", kind="bh", source="bacteria"),
        "gene_clustered_bh": SelectionMethod(key="gene_clustered_bh", label="BH Top-K (Gene-Clustered)", kind="bh", source="gene_clustered"),
    }


def _normalize_1d_string_array(values: np.ndarray) -> np.ndarray:
    if np.ndim(values) == 0:
        return np.array([str(values)])
    return values.astype(str)


def _to_csr_float32(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    return matrix.tocsr().astype(np.float32, copy=False)


def _align_raw_sample_rows(X_raw: sparse.csr_matrix, y_sample_count: int) -> sparse.csr_matrix:
    raw_rows = int(X_raw.shape[0])
    if raw_rows == y_sample_count:
        return X_raw
    if raw_rows > y_sample_count:
        print(f"Warning: raw MTX has {raw_rows - y_sample_count} extra rows vs y_clean. Trimming.")
        return X_raw[:y_sample_count, :].tocsr()
    raise ValueError(f"raw MTX rows ({raw_rows}) < y_clean rows ({y_sample_count})")


def _ordered_knockoff_ranked_indices(rsp_results: dict[str, Any]) -> np.ndarray:
    """Extract all feature indices from feature_index_map sorted by descending W_real."""

    feature_index_map = rsp_results.get("feature_index_map")
    if not isinstance(feature_index_map, dict) or len(feature_index_map) == 0:
        raise ValueError("rsp_results missing non-empty feature_index_map")

    ranked: list[int] = []
    prev_w: float | None = None
    prev_idx: int | None = None

    for raw_idx, raw_values in feature_index_map.items():
        idx = int(raw_idx)
        if idx < 0:
            raise ValueError("feature_index_map contains negative feature index")

        if not isinstance(raw_values, (tuple, list)) or len(raw_values) != 2:
            raise ValueError("feature_index_map values must be (W_real, is_significant) tuples")

        w_val = float(raw_values[0])

        if prev_w is not None:
            if prev_w < w_val:
                raise ValueError("feature_index_map must be sorted by descending W_real")
            if prev_w == w_val and prev_idx is not None and prev_idx > idx:
                raise ValueError("feature_index_map tie-break must use ascending feature index")

        prev_w = w_val
        prev_idx = idx
        ranked.append(idx)

    return np.asarray(ranked, dtype=np.int32)


def load_classifier_inputs(config: ClassifierComparisonConfig) -> ComparisonData:
    """Load matrices and labels for all configured selection methods.

    Required files:
    - study: X_clean.npz, y_clean.npy, {study}_gene_families.mtx, {study}_genes.txt
    - run: X_filtered.npz, genes_filtered.txt, rsp_results.npy
    """

    paths = {
        "raw_mtx": config.study_dir / f"{config.study_name}_gene_families.mtx",
        "raw_genes": config.study_dir / f"{config.study_name}_genes.txt",
        "X_clean": config.study_dir / "X_clean.npz",
        "genes_clean": config.study_dir / "genes_clean.txt",
        "X_filtered": config.run_dir / "X_filtered.npz",
        "genes_filtered": config.run_dir / "genes_filtered.txt",
        "rsp_results": config.run_dir / "rsp_results.npy",
        "y_clean": config.study_dir / "y_clean.npy",
    }

    missing = [k for k in ["raw_mtx", "raw_genes", "X_clean", "X_filtered", "genes_filtered", "rsp_results", "y_clean"] if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    X_raw, genes_raw_series = load_matrix_and_genes(paths["raw_mtx"], paths["raw_genes"])
    genes_raw = genes_raw_series.astype(str).to_numpy()
    X_raw = _to_csr_float32(X_raw)

    X_clean = _to_csr_float32(sparse.load_npz(paths["X_clean"]))
    X_filtered = _to_csr_float32(sparse.load_npz(paths["X_filtered"]))
    genes_filtered = _normalize_1d_string_array(np.loadtxt(paths["genes_filtered"], dtype=str))
    genes_clean: np.ndarray | None = (
        _normalize_1d_string_array(np.loadtxt(paths["genes_clean"], dtype=str))
        if paths["genes_clean"].exists() else None
    )

    y = np.load(paths["y_clean"])
    ranked_feature_indices = _ordered_knockoff_ranked_indices(
        np.load(paths["rsp_results"], allow_pickle=True).item()
    )
    X_raw = _align_raw_sample_rows(X_raw, y.shape[0])

    if X_clean.shape[0] != y.shape[0] or X_filtered.shape[0] != y.shape[0]:
        raise ValueError("Sample mismatch among X_clean/X_filtered and y_clean")
    if X_raw.shape[1] != genes_raw.shape[0]:
        raise ValueError("Feature mismatch between raw matrix and raw genes list")
    if genes_clean is not None and X_clean.shape[1] != genes_clean.shape[0]:
        raise ValueError("Feature mismatch between X_clean and genes_clean")
    if X_filtered.shape[1] != genes_filtered.shape[0]:
        raise ValueError("Feature mismatch between X_filtered and genes_filtered")
    if ranked_feature_indices.size == 0:
        raise ValueError("No ranked knockoff features found in rsp_results")

    if genes_clean is not None:
        X_bacteria_source = X_clean
        genes_bacteria_source = genes_clean
        bacteria_source_tag = "clean"
    else:
        X_bacteria_source = X_filtered
        genes_bacteria_source = genes_filtered
        bacteria_source_tag = "filtered"

    return ComparisonData(
        X_raw=X_raw,
        X_clean=X_clean,
        X_filtered=X_filtered,
        y=y,
        genes_raw=genes_raw,
        genes_clean=genes_clean,
        genes_filtered=genes_filtered,
        X_bacteria_source=X_bacteria_source,
        genes_bacteria_source=genes_bacteria_source,
        bacteria_source_tag=bacteria_source_tag,
        knockoff_ranked_indices=ranked_feature_indices,
    )


def build_bacteria_feature_matrix(
    X_clean: sparse.csr_matrix,
    genes_clean: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Aggregate gene-level abundance to bacteria-level mean abundance matrix (gene|bacteria format)."""

    pair_split = pd.Series(genes_clean).str.rsplit("|", n=1, expand=True)
    if pair_split.shape[1] < 2:
        raise ValueError("Expected genes_clean entries in gene|bacteria format")

    bacteria_tokens = pair_split[1].astype(str).values
    _, n_features = X_clean.shape
    rows = np.arange(n_features)
    unique_bacteria, inverse_bacteria = np.unique(bacteria_tokens, return_inverse=True)
    group_matrix = sparse.csr_matrix(
        (np.ones(n_features, dtype=np.float32), (rows, inverse_bacteria)),
        shape=(n_features, len(unique_bacteria)),
    )
    X_bacteria = X_clean.dot(group_matrix).multiply(1.0 / np.bincount(inverse_bacteria).astype(np.float32)).tocsr()
    return X_bacteria, unique_bacteria.astype(str)


def build_gene_feature_matrix(
    X_source: sparse.csr_matrix,
    feature_names_source: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Aggregate abundance matrix to gene-level mean features (gene|bacteria format)."""

    pair_split = pd.Series(feature_names_source).str.rsplit("|", n=1, expand=True)
    if pair_split.shape[1] < 2:
        raise ValueError("Expected feature names in gene|bacteria format for gene aggregation")

    gene_tokens = pair_split[0].astype(str).values
    _, n_features = X_source.shape
    rows = np.arange(n_features)
    unique_genes, inverse_genes = np.unique(gene_tokens, return_inverse=True)
    group_matrix = sparse.csr_matrix(
        (np.ones(n_features, dtype=np.float32), (rows, inverse_genes)),
        shape=(n_features, len(unique_genes)),
    )
    gene_matrix = X_source.dot(group_matrix).multiply(1.0 / np.bincount(inverse_genes).astype(np.float32)).tocsr()
    return gene_matrix, unique_genes.astype(str)


def build_kmeans_clustered_feature_matrix(
    X_gene: sparse.csr_matrix,
    target_k: int,
    random_state: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Cluster gene-level features with MiniBatchKMeans and aggregate by cluster mean."""

    if target_k < 1:
        raise ValueError("target_k must be >= 1")
    _, n_gene_features = X_gene.shape
    if n_gene_features < 1:
        raise ValueError("Cannot cluster an empty gene feature matrix")

    n_clusters = min(int(target_k), int(n_gene_features))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=5, batch_size=4096)
    cluster_labels = kmeans.fit_predict(X_gene.T.tocsr())

    # Compact labels in case MiniBatchKMeans left empty clusters.
    unique_labels, compact_labels = np.unique(cluster_labels, return_inverse=True)
    if unique_labels.shape[0] < n_clusters:
        n_clusters = int(unique_labels.shape[0])
        cluster_labels = compact_labels

    rows = np.arange(n_gene_features)
    cluster_map = sparse.csr_matrix(
        (np.ones(n_gene_features, dtype=np.float32), (rows, cluster_labels)),
        shape=(n_gene_features, n_clusters),
    )
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters).astype(np.float32)
    if np.any(cluster_sizes == 0):
        raise ValueError("Empty cluster remained after label remapping")
    clustered_matrix = X_gene.dot(cluster_map).multiply(1.0 / cluster_sizes).tocsr()
    cluster_names = np.array([f"gene_cluster_{i}" for i in range(n_clusters)], dtype=str)
    return clustered_matrix, cluster_names


def rank_features_bh(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    test_name: str = "f_classif",
    mannwhitney_chunk_size: int = 2048,
) -> np.ndarray:
    """Rank features by ascending BH-adjusted p-value (then raw p-value as tie-break)."""

    if test_name == "f_classif":
        _, pvals = f_classif(X_train, y_train)
    elif test_name == "mannwhitney":
        if sparse.issparse(X_train):
            mask_1 = y_train == 1
            mask_0 = y_train == 0
            n_features = X_train.shape[1]
            pvals = np.empty(n_features, dtype=float)
            for start in range(0, n_features, mannwhitney_chunk_size):
                end = min(start + mannwhitney_chunk_size, n_features)
                block_1 = X_train[mask_1, start:end].toarray()
                block_0 = X_train[mask_0, start:end].toarray()
                _, pvals[start:end] = mannwhitneyu(block_1, block_0, axis=0)
                del block_1, block_0
        else:
            _, pvals = mannwhitneyu(X_train[y_train == 1], X_train[y_train == 0], axis=0)
    else:
        raise ValueError(f"Unsupported BH test '{test_name}'. Use 'f_classif' or 'mannwhitney'.")

    pvals = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    _, pvals_bh = fdrcorrection(pvals, alpha=0.05)
    return np.lexsort((pvals, pvals_bh)).astype(np.int32)


def build_k_grid(k_min: int, k_max: int, k_grid_points: int = 10) -> list[int]:
    """Build a K grid from k_min to k_max with k_grid_points evenly-spaced values."""

    if k_max < k_min:
        raise ValueError(f"k_max={k_max} < k_min={k_min}")
    if k_grid_points < 2:
        raise ValueError("k_grid_points must be at least 2")
    return sorted(set(int(v) for v in np.linspace(k_min, k_max, num=k_grid_points)))


def _predict_scores(model: Any, X: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return np.asarray(probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel(), dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _fit_and_score_all(
    X_train: sparse.csr_matrix | np.ndarray,
    X_test: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier_factory: ClassifierFactory,
    random_state: int,
) -> dict[str, float]:
    """Fit one model and return ROC AUC, PR AUC, and F1 scores."""
    model = classifier_factory(random_state)
    model.fit(X_train, y_train)
    y_score = _predict_scores(model, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def run_classifier_comparison(
    config: ClassifierComparisonConfig,
    classifier_name: str = "lgbm",
    enabled_methods: list[str] | tuple[str, ...] | None = None,
    method_labels: dict[str, str] | None = None,
    bh_test: str = "mannwhitney",
) -> pd.DataFrame:
    """Run classifier comparison using BH-ranked and knockoff-ranked feature selection.

    Output:
    - Long-format DataFrame with columns: K, method_key, method_label, score, metric_name, classifier_name
    """

    all_methods = method_registry()
    classifiers = classifier_registry()

    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {sorted(classifiers)}")

    if enabled_methods is None:
        enabled_methods = list(all_methods)
    unknown = [m for m in enabled_methods if m not in all_methods]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(all_methods)}")

    methods = [
        SelectionMethod(
            key=m.key,
            label=method_labels.get(m.key, m.label) if method_labels else m.label,
            kind=m.kind,
            source=m.source,
        )
        for m in (all_methods[k] for k in enabled_methods)
    ]

    data = load_classifier_inputs(config)
    idx_all = np.arange(data.y.shape[0])
    idx_train, idx_test = train_test_split(
        idx_all, test_size=config.test_size, random_state=config.random_state, stratify=data.y,
    )
    y_train, y_test = data.y[idx_train], data.y[idx_test]
    classifier_factory = classifiers[classifier_name]

    # Build source matrix map
    matrix_map: dict[str, sparse.csr_matrix] = {
        "raw": data.X_raw,
        "clean": data.X_clean,
        "filtered": data.X_filtered,
    }
    required_sources = {m.source for m in methods}

    if "bacteria" in required_sources:
        bacteria_matrix, _ = build_bacteria_feature_matrix(data.X_bacteria_source, data.genes_bacteria_source)
        matrix_map["bacteria"] = bacteria_matrix
        print(f"Built bacteria matrix ({bacteria_matrix.shape[1]} features) from '{data.bacteria_source_tag}' source")

    if "gene_clustered" in required_sources:
        X_gene, gene_names = build_gene_feature_matrix(data.X_bacteria_source, data.genes_bacteria_source)
        target_k = min(config.k_max, int(gene_names.shape[0]))
        if target_k < 1:
            raise ValueError("Cannot build gene-clustered features: no gene features available")
        print(f"Clustering {X_gene.shape[1]} gene features → {target_k} clusters")
        gene_clustered_matrix, _ = build_kmeans_clustered_feature_matrix(X_gene, target_k, config.random_state)
        matrix_map["gene_clustered"] = gene_clustered_matrix
        del X_gene
        print(f"Gene-clustered matrix: {gene_clustered_matrix.shape}")

    # Compute BH rankings on training data for each BH source
    bh_rankings: dict[str, np.ndarray] = {}
    for source in {m.source for m in methods if m.kind == "bh"}:
        print(f"Computing BH ranking for source '{source}'...")
        bh_rankings[source] = rank_features_bh(matrix_map[source][idx_train], y_train, test_name=bh_test)

    k_values = build_k_grid(config.k_min, config.k_max, config.k_grid_points)
    rows: list[dict[str, object]] = []

    for method in methods:
        t0 = perf_counter()
        print(f"[{method.label}] Starting ({len(k_values)} K values)...")
        source_matrix = matrix_map[method.source]
        ranking = data.knockoff_ranked_indices if method.kind == "knockoff" else bh_rankings[method.source]
        method_capacity = min(int(ranking.shape[0]), int(source_matrix.shape[1]))

        if any(int(K) > method_capacity for K in k_values):
            print(f"  Info: caps at {method_capacity} features for larger K.")

        score_cache: dict[int, dict[str, float]] = {}
        for K in k_values:
            effective_k = min(int(K), method_capacity)
            if effective_k not in score_cache:
                feat_idx = ranking[:effective_k]
                X_tr = source_matrix[idx_train][:, feat_idx]
                X_te = source_matrix[idx_test][:, feat_idx]
                score_cache[effective_k] = _fit_and_score_all(X_tr, X_te, y_train, y_test, classifier_factory, config.random_state)
                del X_tr, X_te
            for metric_name, score in score_cache[effective_k].items():
                rows.append({
                    "K": int(K),
                    "method_key": method.key,
                    "method_label": method.label,
                    "score": float(score),
                    "metric_name": metric_name,
                    "classifier_name": classifier_name,
                })
        print(f"[{method.label}] Done in {perf_counter() - t0:.1f}s.")

    return pd.DataFrame(rows)


def plot_classifier_comparison(
    results: pd.DataFrame,
    study_name: str,
    save_path: Path,
) -> None:
    """Save a 3-panel comparison figure (ROC AUC, PR AUC, F1 Score) to save_path."""

    metrics = [
        ("roc_auc", "ROC AUC"),
        ("average_precision", "PR AUC"),
        ("f1", "F1 Score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (metric_name, metric_label) in zip(axes, metrics):
        subset = results[results["metric_name"] == metric_name]
        for method_label, group in subset.groupby("method_label", sort=False):
            group = group.sort_values("K")
            ax.plot(group["K"], group["score"], marker="o", linewidth=2, label=method_label)
        ax.set_xlabel("K")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} — {study_name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

