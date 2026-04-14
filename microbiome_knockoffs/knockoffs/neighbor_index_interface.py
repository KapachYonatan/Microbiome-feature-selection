from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NeighborIndex(ABC):
    """Interface for neighbor search backends used by knockoff generation.

    Input structure:
    - vectors and query arrays are float32 ndarrays with shape (n_vectors, vector_dim)
      or (1, vector_dim) for queries.

    Output structure:
    - search returns (distances, indices), each ndarray shape (n_queries, k).
    """

    @abstractmethod
    def fit(self, vectors: np.ndarray) -> None:
        """Build index from vectors with shape (n_vectors, vector_dim)."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search nearest neighbors for query vectors."""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Append vectors to the existing index."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for metadata."""
