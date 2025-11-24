from typing import Tuple
import numpy as np

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    """
    Wrapper FAISS / sklearn NearestNeighbors avec la même interface.
    """

    def __init__(self, vectors: np.ndarray, metric: str = "cosine"):
        """
        vectors: (n_samples, dim) float32
        metric: "cosine" ou "ip"
        """
        self.metric = metric
        self.vectors = vectors.astype("float32")
        self.faiss_index = None
        self.nn = None

        if FAISS_AVAILABLE and metric in ("cosine", "ip"):
            self._build_faiss_index()
        else:
            self._build_sklearn_index()

    def _build_faiss_index(self):
        dim = self.vectors.shape[1]
        if self.metric == "cosine":
            # cosine via inner product sur vecteurs normalisés
            faiss.normalize_L2(self.vectors)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(self.vectors)
        self.faiss_index = index

    def _build_sklearn_index(self):
        self.nn = NearestNeighbors(metric=self.metric, algorithm="brute")
        self.nn.fit(self.vectors)

    def search(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vec: (dim,) ou (1, dim)
        Retourne (indices, scores_similarité)
        """
        q = query_vec.astype("float32").reshape(1, -1)

        if self.faiss_index is not None:
            if self.metric == "cosine":
                import faiss

                faiss.normalize_L2(q)
            sims, idx = self.faiss_index.search(q, k)
            return idx[0], sims[0]

        # fallback sklearn
        distances, indices = self.nn.kneighbors(q, n_neighbors=k)
        if self.metric == "cosine":
            sims = 1.0 - distances[0]
            return indices[0], sims
        return indices[0], distances[0]
