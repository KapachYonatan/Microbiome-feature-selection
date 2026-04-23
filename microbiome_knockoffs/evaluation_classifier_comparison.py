from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import fdrcorrection

from .preprocessing_gene_abundance import load_matrix_and_genes


ClassifierFactory = Callable[[int], Any]
MetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ClassifierComparisonConfig:
    """Configuration for knockoff-vs-baseline classifier comparison.

    Input structure:
    - base_dir/study_name identifies study input files.
    - run_folder points to one knockoff run directory under study/runs.

    Output usage:
    - Controls K-grid construction and random-baseline trial count.
    """

    base_dir: Path
    study_name: str
    run_folder: str
    random_state: int = 42
    test_size: float = 0.2
    k_grid_points: int = 10
    random_trials: int = 10
    k_end: int = 5000

    @property
    def study_dir(self) -> Path:
        return self.base_dir / self.study_name

    @property
    def run_dir(self) -> Path:
        return self.study_dir / "runs" / self.run_folder


@dataclass(frozen=True)
class ComparisonData:
    """Container with all matrices needed for classifier comparison.

    Output structure:
    - X_raw: csr_matrix (n_samples, n_raw_features)
    - X_clean: csr_matrix (n_samples, n_clean_features)
    - X_filtered: csr_matrix (n_samples, n_filtered_features)
    - y: ndarray[int] (n_samples,)
    - genes_* arrays aligned to their corresponding matrix columns
    - knockoff_selected_indices: ndarray[int], ranked descending by importance
    - X_bacteria_source/genes_bacteria_source: matrix/name pair used for bacteria baseline.
    """

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
    knockoff_selected_indices: np.ndarray


@dataclass(frozen=True)
class SelectionMethod:
    """Feature selection method descriptor used by the evaluation loop.

    Fields:
    - key: stable internal identifier.
    - label: plot legend label.
    - kind: one of ranked, random, constant.
    - source: one of raw, clean, filtered, bacteria, gene_clustered.
    """

    key: str
    label: str
    kind: str
    source: str


def classifier_registry() -> dict[str, ClassifierFactory]:
    """Create classifier factory registry.

    Output:
    - dict[name -> factory(random_state) -> fitted estimator contract].
    """

    return {
        "lgbm": lambda random_state: lgb.LGBMClassifier(random_state=random_state, verbose=-1, n_estimators=300,
                                                         learning_rate=0.05, min_child_samples=5),
        "logreg_l2": lambda random_state: LogisticRegression(
            random_state=random_state,
            solver="liblinear",
            max_iter=2000,
        ),
    }


def metric_registry() -> dict[str, MetricFunction]:
    """Create metric scorer registry.

    Output:
    - dict[name -> scorer(y_true, y_score)].
    """

    return {
        "roc_auc": lambda y_true, y_score: float(roc_auc_score(y_true, y_score)),
        "average_precision": lambda y_true, y_score: float(average_precision_score(y_true, y_score)),
    }


def method_registry() -> dict[str, SelectionMethod]:
    """Create feature-selection method registry."""

    return {
        "random_raw": SelectionMethod(
            key="random_raw",
            label="Random K (Raw MTX)",
            kind="random",
            source="raw",
        ),
        "bh_raw": SelectionMethod(
            key="bh_raw",
            label="BH Top-K (Raw MTX)",
            kind="ranked",
            source="raw",
        ),
        "random_clean": SelectionMethod(
            key="random_clean",
            label="Random K (X_clean)",
            kind="random",
            source="clean",
        ),
        "random_filtered": SelectionMethod(
            key="random_filtered",
            label="Random K (X_filtered)",
            kind="random",
            source="filtered",
        ),
        "knockoff_topk": SelectionMethod(
            key="knockoff_topk",
            label="Knockoff Top-K",
            kind="ranked",
            source="filtered",
        ),
        "bacteria_constant": SelectionMethod(
            key="bacteria_constant",
            label="Bacteria Features (Constant)",
            kind="constant",
            source="bacteria",
        ),
        "gene_clustered_random": SelectionMethod(
            key="gene_clustered_random",
            label="Gene-Clustered Random K",
            kind="random",
            source="gene_clustered",
        ),
    }


def _normalize_1d_string_array(values: np.ndarray) -> np.ndarray:
    if np.ndim(values) == 0:
        return np.array([str(values)])
    return values.astype(str)


def _to_csr_float32(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Convert sparse matrix to CSR float32 without unnecessary copies."""

    return matrix.tocsr().astype(np.float32, copy=False)


def _align_raw_sample_rows(
    X_raw: sparse.csr_matrix,
    y_sample_count: int,
) -> sparse.csr_matrix:
    """Align raw matrix sample rows to y_clean count when raw has extra rows.

    Some studies have one or more trailing raw rows that are not represented
    in y_clean after preprocessing. We trim trailing rows to preserve the
    sample alignment contract used by classifier comparison.
    """

    raw_rows = int(X_raw.shape[0])
    if raw_rows == y_sample_count:
        return X_raw

    if raw_rows > y_sample_count:
        extra_rows = raw_rows - y_sample_count
        print(
            "Warning: raw MTX has extra rows compared with y_clean "
            f"(raw={raw_rows}, y={y_sample_count}). Trimming last {extra_rows} row(s)."
        )
        return X_raw[:y_sample_count, :].tocsr()

    raise ValueError(
        "Sample mismatch between raw MTX and y_clean. "
        f"raw={raw_rows} y={y_sample_count}. "
        "raw MTX has fewer rows than y_clean, cannot auto-align."
    )


def _ordered_knockoff_selected_indices(rsp_results: dict[str, Any]) -> np.ndarray:
    """Extract significant feature indices from canonical feature_index_map.

    The map must already be sorted by descending W_real with ascending feature
    index as the tie-break. This function validates ordering and returns the
    significant indices in that saved order.
    """

    feature_index_map = rsp_results.get("feature_index_map")
    if not isinstance(feature_index_map, dict) or len(feature_index_map) == 0:
        raise ValueError("rsp_results missing non-empty feature_index_map")

    selected_feature_indices: list[int] = []
    prev_w: float | None = None
    prev_idx: int | None = None

    for raw_idx, raw_values in feature_index_map.items():
        idx = int(raw_idx)
        if idx < 0:
            raise ValueError("feature_index_map contains negative feature index")

        if not isinstance(raw_values, (tuple, list)) or len(raw_values) != 2:
            raise ValueError("feature_index_map values must be (W_real, is_significant) tuples")

        w_val = float(raw_values[0])
        is_significant = bool(raw_values[1])

        if prev_w is not None:
            if prev_w < w_val:
                raise ValueError("feature_index_map must be sorted by descending W_real")
            if prev_w == w_val and prev_idx is not None and prev_idx > idx:
                raise ValueError(
                    "feature_index_map tie-break must use ascending feature index"
                )

        prev_w = w_val
        prev_idx = idx

        if is_significant:
            selected_feature_indices.append(idx)

    return np.asarray(selected_feature_indices, dtype=np.int32)


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

    required_keys = [
        "raw_mtx",
        "raw_genes",
        "X_clean",
        "X_filtered",
        "genes_filtered",
        "rsp_results",
        "y_clean",
    ]
    missing = [name for name in required_keys if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    X_raw, genes_raw_series = load_matrix_and_genes(paths["raw_mtx"], paths["raw_genes"])
    genes_raw = genes_raw_series.astype(str).to_numpy()
    X_raw = _to_csr_float32(X_raw)

    X_clean = _to_csr_float32(sparse.load_npz(paths["X_clean"]))
    X_filtered = _to_csr_float32(sparse.load_npz(paths["X_filtered"]))
    genes_filtered = _normalize_1d_string_array(np.loadtxt(paths["genes_filtered"], dtype=str))

    genes_clean: np.ndarray | None
    if paths["genes_clean"].exists():
        genes_clean = _normalize_1d_string_array(np.loadtxt(paths["genes_clean"], dtype=str))
    else:
        genes_clean = None

    y = np.load(paths["y_clean"])
    rsp_results = np.load(paths["rsp_results"], allow_pickle=True).item()
    selected_feature_indices = _ordered_knockoff_selected_indices(rsp_results)

    sample_count = y.shape[0]
    X_raw = _align_raw_sample_rows(X_raw, sample_count)

    if X_clean.shape[0] != y.shape[0] or X_filtered.shape[0] != y.shape[0]:
        raise ValueError("Sample mismatch among X_clean/X_filtered and y_clean")
    if X_raw.shape[1] != genes_raw.shape[0]:
        raise ValueError("Feature mismatch between raw matrix and raw genes list")
    if genes_clean is not None and X_clean.shape[1] != genes_clean.shape[0]:
        raise ValueError("Feature mismatch between X_clean and genes_clean")
    if X_filtered.shape[1] != genes_filtered.shape[0]:
        raise ValueError("Feature mismatch between X_filtered and genes_filtered")
    if selected_feature_indices.size == 0:
        raise ValueError("No knockoff selected features found in rsp_results")

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
        knockoff_selected_indices=selected_feature_indices,
    )


def build_bacteria_feature_matrix(
    X_clean: sparse.csr_matrix,
    genes_clean: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Aggregate gene-level abundance to bacteria-level mean abundance matrix.

    Input:
    - X_clean: csr_matrix (n_samples, n_genes)
    - genes_clean: ndarray[str] (n_genes,), expected in "gene|bacteria" format.

    Output:
    - bacteria_matrix: csr_matrix shape (n_samples, n_unique_bacteria)
    - bacteria_names: ndarray[str] shape (n_unique_bacteria,)
    """

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
    """Aggregate abundance matrix to gene-level features.

    Input:
    - X_source: csr_matrix (n_samples, n_features), where feature names follow gene|bacteria.
    - feature_names_source: ndarray[str] aligned to X_source columns.

    Output:
    - gene_matrix: csr_matrix shape (n_samples, n_unique_genes)
    - gene_names: ndarray[str] shape (n_unique_genes,)
    """

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
    """Cluster gene-level features with MiniBatchKMeans and aggregate by cluster.

    Input:
    - X_gene: csr_matrix (n_samples, n_gene_features)
    - target_k: requested number of clustered output features
    - random_state: reproducibility seed for MiniBatchKMeans

    Output:
    - clustered_matrix: csr_matrix (n_samples, n_clusters)
    - cluster_names: ndarray[str] (n_clusters,)
    """

    if target_k < 1:
        raise ValueError("target_k must be >= 1 for MiniBatchKMeans clustering")

    _, n_gene_features = X_gene.shape
    if n_gene_features < 1:
        raise ValueError("Cannot cluster an empty gene feature matrix")

    n_clusters = min(int(target_k), int(n_gene_features))
    gene_vectors = X_gene.T.tocsr()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=5,
        batch_size=4096,
    )
    cluster_labels = kmeans.fit_predict(gene_vectors)

    # MiniBatchKMeans can leave some clusters empty, especially at large K.
    # Compact labels to the set of observed clusters so aggregation can proceed.
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
        raise ValueError("Cluster compaction failed: empty cluster remained after label remapping")

    clustered_matrix = X_gene.dot(cluster_map).multiply(1.0 / cluster_sizes).tocsr()
    cluster_names = np.array([f"gene_cluster_{idx}" for idx in range(n_clusters)], dtype=str)
    return clustered_matrix, cluster_names


def rank_features_bh(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    test_name: str = "f_classif",
    mannwhitney_chunk_size: int = 2048,
) -> np.ndarray:
    """Rank raw features using BH-corrected p-values.

    Input:
    - X_train: csr_matrix (n_train_samples, n_features)
    - y_train: ndarray[int] (n_train_samples,)
    - test_name: "f_classif" or "mannwhitney"

    Output:
    - ranked feature indices sorted by ascending BH-adjusted p-value, then raw p-value.
    """

    if test_name == "f_classif":
        _, pvals = f_classif(X_train, y_train)
    elif test_name == "mannwhitney":
        if sparse.issparse(X_train):
            if mannwhitney_chunk_size < 1:
                raise ValueError("mannwhitney_chunk_size must be >= 1")

            mask_1 = y_train == 1
            mask_0 = y_train == 0
            n_features = X_train.shape[1]
            pvals = np.empty(n_features, dtype=float)

            for start in range(0, n_features, mannwhitney_chunk_size):
                end = min(start + mannwhitney_chunk_size, n_features)
                block_1 = X_train[mask_1, start:end].toarray()
                block_0 = X_train[mask_0, start:end].toarray()
                _, pvals_block = mannwhitneyu(block_1, block_0, axis=0)
                pvals[start:end] = pvals_block
                del block_1, block_0, pvals_block
        else:
            _, pvals = mannwhitneyu(X_train[y_train == 1], X_train[y_train == 0], axis=0)
    else:
        raise ValueError(f"Unsupported BH test '{test_name}'. Use 'f_classif' or 'mannwhitney'.")

    pvals = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    _, pvals_bh = fdrcorrection(pvals, alpha=0.05)

    # Primary sort by BH-adjusted p-value, secondary by raw p-value.
    ranking = np.lexsort((pvals, pvals_bh))
    return ranking.astype(np.int32)


def build_k_grid(
    k_start: int,
    k_end_requested: int,
    k_grid_points: int,
) -> list[int]:
    """Build monotonic K grid from computed K_start to user-requested K_end."""

    if k_end_requested < k_start:
        raise ValueError(f"Requested K_end={k_end_requested} is smaller than K_start={k_start}")
    if k_grid_points < 2:
        raise ValueError("k_grid_points must be at least 2")

    k_values = np.linspace(k_start, k_end_requested, num=k_grid_points, dtype=int)
    k_values = np.unique(k_values).tolist()
    if k_values[0] != k_start:
        k_values.insert(0, k_start)
    if k_values[-1] != k_end_requested:
        k_values.append(k_end_requested)
    return k_values


def _predict_scores(model: Any, X: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    """Return 1D decision scores usable by metric functions."""

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.asarray(probs[:, 1], dtype=float)
        return np.asarray(probs.ravel(), dtype=float)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)

    return np.asarray(model.predict(X), dtype=float)


def _fit_and_score(
    X_train: sparse.csr_matrix | np.ndarray,
    X_test: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classifier_factory: ClassifierFactory,
    metric_fn: MetricFunction,
    random_state: int,
) -> float:
    """Fit one model and evaluate one metric score."""

    model = classifier_factory(random_state)
    model.fit(X_train, y_train)
    y_score = _predict_scores(model, X_test)
    return float(metric_fn(y_test, y_score))


def _matrix_sources(
    data: ComparisonData,
    bacteria_matrix: sparse.csr_matrix | None,
    gene_clustered_matrix: sparse.csr_matrix | None,
    required_sources: set[str] | None = None,
) -> dict[str, sparse.csr_matrix]:
    full_map = {
        "raw": data.X_raw,
        "clean": data.X_clean,
        "filtered": data.X_filtered,
    }
    if bacteria_matrix is not None:
        full_map["bacteria"] = bacteria_matrix
    if gene_clustered_matrix is not None:
        full_map["gene_clustered"] = gene_clustered_matrix

    if required_sources is None:
        return full_map

    matrix_map: dict[str, sparse.csr_matrix] = {}
    for source in required_sources:
        if source in full_map:
            matrix_map[source] = full_map[source]
    return matrix_map


def _capacity_for_method(
    method: SelectionMethod,
    ranking_map: dict[str, np.ndarray],
    matrix_map: dict[str, sparse.csr_matrix],
) -> int | None:
    if method.kind == "constant":
        return None

    if method.kind == "ranked":
        if method.key not in ranking_map:
            raise ValueError(f"Missing ranking for method '{method.key}'")
        return int(ranking_map[method.key].shape[0])

    if method.kind == "random":
        if method.source not in matrix_map:
            raise ValueError(f"Missing matrix source '{method.source}' for random method '{method.key}'")
        return int(matrix_map[method.source].shape[1])

    raise ValueError(f"Unsupported method kind '{method.kind}'")


def run_classifier_comparison(
    config: ClassifierComparisonConfig,
    classifier_name: str = "lgbm",
    metric_name: str = "roc_auc",
    enabled_methods: list[str] | tuple[str, ...] | None = None,
    method_labels: dict[str, str] | None = None,
    bh_test: str = "f_classif",
) -> pd.DataFrame:
    """Run modular classifier comparison across configurable feature-selection methods.

    Output:
    - Long-format DataFrame with columns:
      K, method_key, method_label, score, classifier_name, metric_name
    """

    all_methods = method_registry()
    classifiers = classifier_registry()
    metrics = metric_registry()

    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {sorted(classifiers)}")
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric '{metric_name}'. Available: {sorted(metrics)}")

    if enabled_methods is None:
        enabled_methods = [
            "random_raw",
            "bh_raw",
            "random_clean",
            "random_filtered",
            "knockoff_topk",
            "bacteria_constant",
            "gene_clustered_random",
        ]

    unknown_methods = [method for method in enabled_methods if method not in all_methods]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Available: {sorted(all_methods)}")

    methods = [all_methods[method] for method in enabled_methods]
    if method_labels:
        methods = [
            SelectionMethod(
                key=method.key,
                label=method_labels.get(method.key, method.label),
                kind=method.kind,
                source=method.source,
            )
            for method in methods
        ]

    required_sources = {method.source for method in methods}

    data = load_classifier_inputs(config)

    idx_all = np.arange(data.y.shape[0])
    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data.y,
    )
    y_train, y_test = data.y[idx_train], data.y[idx_test]

    needs_bacteria = any(method.key == "bacteria_constant" for method in methods)
    needs_gene_clustered = any(method.key == "gene_clustered_random" for method in methods)
    bacteria_matrix: sparse.csr_matrix | None
    if needs_bacteria:
        bacteria_matrix, _ = build_bacteria_feature_matrix(
            data.X_bacteria_source,
            data.genes_bacteria_source,
        )
        print(f"Using '{data.bacteria_source_tag}' matrix for bacteria baseline features")
    else:
        bacteria_matrix = None

    ranking_map: dict[str, np.ndarray] = {}
    if any(method.key == "knockoff_topk" for method in methods):
        ranking_map["knockoff_topk"] = data.knockoff_selected_indices
    if any(method.key == "bh_raw" for method in methods):
        ranking_map["bh_raw"] = rank_features_bh(data.X_raw[idx_train], y_train, test_name=bh_test)

    matrix_map = _matrix_sources(
        data,
        bacteria_matrix,
        gene_clustered_matrix=None,
        required_sources=required_sources,
    )

    if needs_gene_clustered:
        candidate_methods = [
            method
            for method in methods
            if method.kind != "constant" and method.key != "gene_clustered_random"
        ]

        if candidate_methods:
            candidate_capacities = [
                _capacity_for_method(method, ranking_map, matrix_map)
                for method in candidate_methods
            ]
            max_k_needed = int(min(capacity for capacity in candidate_capacities if capacity is not None))
        else:
            max_k_needed = int(config.k_end)

        X_gene, gene_names = build_gene_feature_matrix(
            data.X_bacteria_source,
            data.genes_bacteria_source,
        )
        target_k = min(int(config.k_end), max(1, max_k_needed), int(gene_names.shape[0]))
        if target_k < 1:
            raise ValueError("Cannot build gene-clustered features: no gene features available")

        print(
            f"Using '{data.bacteria_source_tag}' matrix for gene aggregation source "
            f"({X_gene.shape[1]} unique genes)"
        )
        print(f"Clustering gene features with MiniBatchKMeans to target K={target_k}")

        gene_clustered_matrix, _ = build_kmeans_clustered_feature_matrix(
            X_gene,
            target_k=target_k,
            random_state=config.random_state,
        )
        del X_gene
        print(f"Clustered gene matrix shape: {gene_clustered_matrix.shape}")

        matrix_map = _matrix_sources(
            data,
            bacteria_matrix,
            gene_clustered_matrix=gene_clustered_matrix,
            required_sources=required_sources,
        )

    method_k_capacity: dict[str, int] = {
        method.key: int(_capacity_for_method(method, ranking_map, matrix_map))
        for method in methods
        if method.kind != "constant"
    }
    if not method_k_capacity:
        raise ValueError("At least one K-dependent method must be enabled.")
    if any(capacity < 1 for capacity in method_k_capacity.values()):
        raise ValueError("All K-dependent methods must have at least one available feature")

    # K_start should respect all enabled methods, including constant baselines
    # such as bacteria features.
    all_capacities = list(method_k_capacity.values())
    for method in methods:
        if method.kind == "constant" and method.source in matrix_map:
            all_capacities.append(int(matrix_map[method.source].shape[1]))

    k_start = int(min(all_capacities))
    k_values = build_k_grid(k_start, int(config.k_end), config.k_grid_points)

    classifier_factory = classifiers[classifier_name]
    metric_fn = metrics[metric_name]

    rng = np.random.default_rng(config.random_state)
    rows: list[dict[str, object]] = []

    for method in methods:
        if method.kind == "constant":
            X_train_const = matrix_map[method.source][idx_train]
            X_test_const = matrix_map[method.source][idx_test]
            score = _fit_and_score(
                X_train_const,
                X_test_const,
                y_train,
                y_test,
                classifier_factory,
                metric_fn,
                config.random_state,
            )
            del X_train_const, X_test_const

            for K in k_values:
                rows.append(
                    {
                        "K": int(K),
                        "method_key": method.key,
                        "method_label": method.label,
                        "score": float(score),
                        "classifier_name": classifier_name,
                        "metric_name": metric_name,
                    }
                )
            continue

        source_matrix = matrix_map[method.source]
        method_capacity = method_k_capacity[method.key]
        if any(int(K) > method_capacity for K in k_values):
            print(
                f"Info: method '{method.key}' caps at {method_capacity} features; "
                "scores for larger K reuse the capped model score."
            )

        score_cache: dict[int, float] = {}

        for K in k_values:
            effective_k = min(int(K), method_capacity)

            if effective_k not in score_cache:
                if method.kind == "ranked":
                    feature_indices = ranking_map[method.key][:effective_k]

                    X_train_sub = source_matrix[idx_train][:, feature_indices]
                    X_test_sub = source_matrix[idx_test][:, feature_indices]
                    score_cache[effective_k] = _fit_and_score(
                        X_train_sub,
                        X_test_sub,
                        y_train,
                        y_test,
                        classifier_factory,
                        metric_fn,
                        config.random_state,
                    )
                    del X_train_sub, X_test_sub, feature_indices
                elif method.kind == "random":
                    trial_scores: list[float] = []
                    n_features = source_matrix.shape[1]
                    for trial_idx in range(config.random_trials):
                        trial_seed = int(rng.integers(1, 1_000_000_000))
                        feature_indices = rng.choice(n_features, size=effective_k, replace=False)

                        X_train_sub = source_matrix[idx_train][:, feature_indices]
                        X_test_sub = source_matrix[idx_test][:, feature_indices]
                        trial_scores.append(
                            _fit_and_score(
                                X_train_sub,
                                X_test_sub,
                                y_train,
                                y_test,
                                classifier_factory,
                                metric_fn,
                                trial_seed + trial_idx,
                            )
                        )
                        del X_train_sub, X_test_sub, feature_indices
                    score_cache[effective_k] = float(np.mean(trial_scores))
                    del trial_scores
                else:
                    raise ValueError(f"Unsupported method kind '{method.kind}'")

            score = float(score_cache[effective_k])

            rows.append(
                {
                    "K": int(K),
                    "method_key": method.key,
                    "method_label": method.label,
                    "score": float(score),
                    "classifier_name": classifier_name,
                    "metric_name": metric_name,
                }
            )

    return pd.DataFrame(rows)


def plot_classifier_comparison(
    results: pd.DataFrame,
    study_name: str,
    save_path: Path,
    metric_name: str = "roc_auc",
) -> None:
    """Plot and save method-wise classifier comparison curves.

    Input:
    - results: long-format DataFrame produced by run_classifier_comparison.
    - save_path: output PNG path.

    Output:
    - Saved figure showing metric-vs-K lines for enabled methods.
    """

    metric_label = {
        "roc_auc": "ROC AUC",
        "average_precision": "Average Precision",
    }.get(metric_name, metric_name)

    plt.figure(figsize=(10, 6))
    for method_label, group in results.groupby("method_label", sort=False):
        method_data = group.sort_values("K")
        plt.plot(
            method_data["K"],
            method_data["score"],
            marker="o",
            linewidth=2,
            label=method_label,
        )

    plt.xlabel("K")
    plt.ylabel(metric_label)
    plt.title(f"Classifier Comparison Across K ({study_name})")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
