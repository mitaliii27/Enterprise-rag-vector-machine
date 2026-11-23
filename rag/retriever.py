"""Sparse, dense and hybrid (reciprocal-rank-fusion) retrieval over chunks."""
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, SparseIndex
from .vectorstore import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float          # score used for ranking (cosine, or RRF for hybrid)
    sparse_score: float   # lexical cosine - also used as the abstention confidence
    dense_score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], sparse: SparseIndex, embedder: Embedder, store: VectorStore,
                 mode: str = "hybrid", rrf_k: int = 60):
        assert mode in ("sparse", "dense", "hybrid")
        self.chunks, self.sparse, self.embedder, self.store, self.mode, self.rrf_k = chunks, sparse, embedder, store, mode, rrf_k
        self._index = {c.chunk_id: i for i, c in enumerate(chunks)}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        n = len(self.chunks)
        s_scores = self.sparse.scores(query)
        q_vec = self.embedder.embed([query])[0]
        d_scores = np.zeros(n)
        for cid, sc in self.store.search(q_vec, n):
            d_scores[self._index[cid]] = sc
        if self.mode == "sparse":
            final = s_scores
        elif self.mode == "dense":
            final = d_scores
        else:  # reciprocal rank fusion
            rs = np.empty(n, dtype=int); rs[np.argsort(-s_scores)] = np.arange(n)
            rd = np.empty(n, dtype=int); rd[np.argsort(-d_scores)] = np.arange(n)
            final = 1.0 / (self.rrf_k + rs + 1) + 1.0 / (self.rrf_k + rd + 1)
        top = np.argsort(-final)[:k]
        return [Hit(self.chunks[i], float(final[i]), float(s_scores[i]), float(d_scores[i])) for i in top]
