from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse

try:
    from ete3 import NCBITaxa
except ImportError:  # pragma: no cover
    NCBITaxa = None


@dataclass(frozen=True)
class PreprocessConfig:
    """Gene abundance preprocessing configuration.

    Input structure:
    - base_dir/study_name identifies source files:
      {study}_gene_families.mtx, {study}_genes.txt, {study}_metadata.tsv

    Output usage:
    - Controls label mapping and prevalence filtering threshold for saved clean artifacts.
    """

    base_dir: Path
    study_name: str
    condition_col: str
    healthy_label: str
    condition_label: str
    rare_threshold: float = 0.05

    @property
    def study_dir(self) -> Path:
        return self.base_dir / self.study_name

    @property
    def mtx_path(self) -> Path:
        return self.study_dir / f"{self.study_name}_gene_families.mtx"

    @property
    def genes_path(self) -> Path:
        return self.study_dir / f"{self.study_name}_genes.txt"

    @property
    def meta_path(self) -> Path:
        return self.study_dir / f"{self.study_name}_metadata.tsv"


@dataclass(frozen=True)
class PreprocessArtifacts:
    """Preprocessing outputs.

    Output structure:
    - X_clean: csr_matrix shape (n_samples_kept, n_features_kept).
    - y_clean: ndarray[int] shape (n_samples_kept,).
    - genes_clean: ndarray[str] shape (n_features_kept,).
    """

    X_clean: sparse.csr_matrix
    y_clean: np.ndarray
    genes_clean: np.ndarray


def load_matrix_and_genes(mtx_path: Path, genes_path: Path) -> tuple[sparse.csr_matrix, pd.Series]:
    """Load gene names and matrix, then enforce matrix orientation as samples x features."""

    genes_df = pd.read_csv(genes_path, header=None, names=["Gene_Family"])
    n_expected_features = len(genes_df)

    raw_matrix = io.mmread(mtx_path)

    if raw_matrix.shape[1] == n_expected_features:
        X = raw_matrix.tocsr()
    elif raw_matrix.shape[0] == n_expected_features:
        X = raw_matrix.T.tocsr()
    else:
        raise ValueError(
            "CRITICAL DIMENSION MISMATCH: matrix shape "
            f"{raw_matrix.shape} does not align with genes count {n_expected_features}."
        )

    return X, genes_df["Gene_Family"]


def get_robust_bacteria_mask(genes_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build bacteria inclusion mask and novel-feature mask from gene taxonomy strings.

    Input:
    - genes_list: list[str] of feature identifiers, expected to include g__ or s__ tokens.

    Output:
    - bacteria_mask: bool ndarray shape (n_features,), True if feature is kept.
    - novel_mask: bool ndarray shape (n_features,), True for taxa not resolved in NCBI.
    """

    if NCBITaxa is None:
        raise ImportError("ete3 is required for taxonomy filtering. Install ete3 to run preprocessing.")

    ncbi = NCBITaxa()
    BACTERIA_TAXID = 2

    genes_series = pd.Series(genes_list)
    taxa_raw = genes_series.str.extract(r"g__([^\.\|]+)")[0]
    taxa_raw = taxa_raw.fillna(genes_series.str.extract(r"s__([^\.\|]+)")[0])

    regex_suffixes = (
        r"_(CAG|[ku]?SGB|[A-Z]?FGB|GGB|MAG|KLE|unclassified|Incertae_Sedis|"
        r"Family_[IVX]+|group|genosp|sp|bacterium|oral_taxon).*"
    )

    clean_taxa = taxa_raw.str.replace(regex_suffixes, "", regex=True)
    clean_taxa = clean_taxa.str.replace(r"_\d+$", "", regex=True)
    clean_taxa = clean_taxa.str.replace("_", " ")

    unique_clean_taxa = clean_taxa.dropna().unique()
    bacteria_dict: dict[str, bool] = {}
    novel_dict: dict[str, bool] = {}

    for taxa in unique_clean_taxa:
        try:
            name2taxid = ncbi.get_name_translator([taxa])
            if name2taxid:
                taxid = name2taxid[taxa][0]
                lineage = ncbi.get_lineage(taxid)
                bacteria_dict[taxa] = BACTERIA_TAXID in lineage
                novel_dict[taxa] = False
            else:
                bacteria_dict[taxa] = True
                novel_dict[taxa] = True
        except Exception:
            # Keep unresolved taxa to avoid over-pruning potentially novel biomarkers.
            bacteria_dict[taxa] = True
            novel_dict[taxa] = True

    bacteria_mask = clean_taxa.map(bacteria_dict).fillna(False).values.astype(bool)
    novel_mask = clean_taxa.map(novel_dict).fillna(False).values.astype(bool)
    return bacteria_mask, novel_mask


def apply_group_prevalence_filter(
    X: sparse.csr_matrix,
    y: np.ndarray,
    rare_threshold: float,
) -> np.ndarray:
    """Build a feature-keep mask using prevalence thresholds in each class group.

    Input:
    - X: csr_matrix (n_samples, n_features).
    - y: binary labels (n_samples,), values 0/1.

    Output:
    - keep_mask: bool ndarray (n_features,), True for retained features.
    """

    X_bool = X.astype(bool)
    X_h = X_bool[y == 0, :]
    X_c = X_bool[y == 1, :]

    prevalence_h = np.asarray(X_h.mean(axis=0)).ravel()
    prevalence_c = np.asarray(X_c.mean(axis=0)).ravel()

    rare_in_h = prevalence_h < rare_threshold
    rare_in_c = prevalence_c < rare_threshold
    rare_in_both = rare_in_h & rare_in_c

    return ~rare_in_both


def preprocess_study(config: PreprocessConfig) -> PreprocessArtifacts:
    """Run end-to-end preprocessing for one study and return in-memory artifacts."""

    X, genes = load_matrix_and_genes(config.mtx_path, config.genes_path)

    bacteria_mask, _ = get_robust_bacteria_mask(genes.tolist())
    keep_idx = np.where(bacteria_mask)[0]

    X = X.tocsc()[:, keep_idx].tocsr()
    genes = genes.iloc[keep_idx].reset_index(drop=True)

    metadata = pd.read_csv(config.meta_path, sep="\t", low_memory=False)
    y_raw = metadata[config.condition_col].map({config.healthy_label: 0, config.condition_label: 1}).to_numpy()

    valid_samples_mask = ~np.isnan(y_raw)
    X = X[valid_samples_mask, :]
    y = y_raw[valid_samples_mask].astype(np.int32)

    prevalence_keep = apply_group_prevalence_filter(X, y, config.rare_threshold)
    X_clean = X[:, prevalence_keep].tocsr()
    genes_clean = genes[prevalence_keep].reset_index(drop=True).astype(str).to_numpy()

    return PreprocessArtifacts(X_clean=X_clean, y_clean=y, genes_clean=genes_clean)


def save_preprocess_outputs(study_dir: Path, artifacts: PreprocessArtifacts) -> tuple[Path, Path, Path]:
    """Save preprocessing outputs to standard study paths.

    Output files:
    - X_clean.npz
    - y_clean.npy
    - genes_clean.txt
    """

    x_path = study_dir / "X_clean.npz"
    y_path = study_dir / "y_clean.npy"
    genes_path = study_dir / "genes_clean.txt"

    sparse.save_npz(x_path, artifacts.X_clean)
    np.save(y_path, artifacts.y_clean)
    np.savetxt(genes_path, artifacts.genes_clean, fmt="%s")

    return x_path, y_path, genes_path
