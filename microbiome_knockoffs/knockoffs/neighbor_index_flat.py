from __future__ import annotations

import numpy as np

from .neighbor_index_interface import NeighborIndex

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required for FaissFlatIPIndex") from exc


class FaissFlatIPIndex(NeighborIndex):
    """Deterministic exact FAISS inner-product neighbor index."""

    def __init__(self, vector_dim: int) -> None:
        self.vector_dim = vector_dim
        self._index: faiss.IndexFlatIP | None = None

    @property
    def name(self) -> str:
        return "FaissFlatIPIndex"

    def fit(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        index = faiss.IndexFlatIP(self.vector_dim)
        index.add(vectors)
        self._index = index

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not fitted. Call fit() before search().")
        query = np.ascontiguousarray(query.astype(np.float32, copy=False))
        distances, indices = self._index.search(query, k)
        return distances, indices

    def add(self, vectors: np.ndarray) -> None:
        if self._index is None:
            raise RuntimeError("Index not fitted. Call fit() before add().")
        vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        self._index.add(vectors)
