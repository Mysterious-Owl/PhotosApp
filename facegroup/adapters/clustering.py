"""Clustering for suggestions. HDBSCAN needs no group count and leaves outliers as noise (unsorted)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA


class HdbscanClusterer:
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 2, pca_above: int = 5000,
                 pca_dims: int = 64) -> None:
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._pca_above = pca_above
        self._pca_dims = pca_dims

    def cluster(self, vectors: np.ndarray) -> np.ndarray:
        n = len(vectors)
        if n < self._min_cluster_size:
            return np.full(n, -1)
        x = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        if n > self._pca_above:  # HDBSCAN slows sharply in 512 dimensions; 64 keeps identity structure
            x = PCA(n_components=self._pca_dims, random_state=0).fit_transform(x)
        # On unit vectors, euclidean distance is a monotone function of cosine similarity.
        # "leaf" selection prefers many tight groups over a few large mixed ones: purer suggestions.
        return HDBSCAN(min_cluster_size=self._min_cluster_size, min_samples=self._min_samples,
                       cluster_selection_method="leaf", copy=True).fit_predict(x)
