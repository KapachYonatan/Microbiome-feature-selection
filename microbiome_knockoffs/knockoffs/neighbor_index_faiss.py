from __future__ import annotations

import numpy as np

from .neighbor_index_interface import NeighborIndex

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required for FaissHNSWIndex") from exc


class FaissHNSWIndex(NeighborIndex):
    """FAISS HNSW inner-product neighbor index."""

    def __init__(self, vector_dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 100) -> None:
        self.vector_dim = vector_dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None

    @property
    def name(self) -> str:
        return "FaissHNSWIndex"

    def fit(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        index = faiss.IndexHNSWFlat(self.vector_dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
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
