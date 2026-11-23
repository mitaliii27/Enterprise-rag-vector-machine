"""Vector store abstraction with three backends.

NumpyVectorStore - default, exact cosine, zero dependencies beyond numpy.
FaissVectorStore - exact inner-product index; tested for parity with the numpy store.
HanaVectorStore  - SAP HANA Cloud vector engine via SQL (REAL_VECTOR, COSINE_SIMILARITY).
                   !! NOT TESTED AGAINST A REAL HANA INSTANCE !! Only the generated SQL and
                   parameter handling are unit-tested with a fake DB-API connection. Verify
                   function names/syntax against the current SAP HANA Cloud vector-engine docs.
"""
import json
import re
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class NumpyVectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None

    def add(self, ids, vectors):
        self.ids += list(ids)
        self.mat = vectors.copy() if self.mat is None else np.vstack([self.mat, vectors])

    def search(self, query, k):
        sims = self.mat @ query
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


class FaissVectorStore:
    def __init__(self, dim: int):
        import faiss  # lazy: optional dependency
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, ids, vectors):
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.ids += list(ids)

    def search(self, query, k):
        scores, idx = self.index.search(np.ascontiguousarray(query.reshape(1, -1), dtype="float32"), min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class HanaVectorStore:
    """DB-API (hdbcli-style) adapter. UNTESTED against real HANA - see module docstring."""

    def __init__(self, conn, table: str, dim: int):
        if not _IDENT.match(table):
            raise ValueError("table must be a plain SQL identifier")
        self.conn, self.table, self.dim = conn, table, dim

    def create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(f'CREATE TABLE "{self.table}" (ID NVARCHAR(200) PRIMARY KEY, VEC REAL_VECTOR({int(self.dim)}))')

    def add(self, ids, vectors):
        cur = self.conn.cursor()
        for i, v in zip(ids, vectors):
            cur.execute(f'INSERT INTO "{self.table}" (ID, VEC) VALUES (?, TO_REAL_VECTOR(?))', (i, json.dumps([float(x) for x in v])))

    def search(self, query, k):
        cur = self.conn.cursor()
        cur.execute(f'SELECT TOP {int(k)} ID, COSINE_SIMILARITY(VEC, TO_REAL_VECTOR(?)) AS SCORE FROM "{self.table}" ORDER BY SCORE DESC',
                    (json.dumps([float(x) for x in query]),))
        return [(row[0], float(row[1])) for row in cur.fetchall()]
