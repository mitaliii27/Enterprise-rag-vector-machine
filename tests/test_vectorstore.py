import json

import numpy as np
import pytest

from rag.vectorstore import FaissVectorStore, HanaVectorStore, NumpyVectorStore


def _data(n=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [f"c{i}" for i in range(n)], v


def test_numpy_store_returns_nearest_first():
    ids, v = _data()
    s = NumpyVectorStore(); s.add(ids, v)
    res = s.search(v[3], 3)
    assert res[0][0] == "c3" and res[0][1] == pytest.approx(1.0) and res[0][1] >= res[1][1] >= res[2][1]


def test_faiss_matches_numpy():
    pytest.importorskip("faiss")
    ids, v = _data()
    a, b = NumpyVectorStore(), FaissVectorStore(v.shape[1])
    a.add(ids, v); b.add(ids, v)
    for i in range(0, 20, 3):
        assert [x[0] for x in a.search(v[i], 5)] == [x[0] for x in b.search(v[i], 5)]


class FakeCursor:
    def __init__(self, log, rows): self.log, self.rows = log, rows
    def execute(self, sql, params=None): self.log.append((" ".join(sql.split()), params))
    def fetchall(self): return self.rows


class FakeConn:
    def __init__(self, rows=()): self.log, self.rows = [], list(rows)
    def cursor(self): return FakeCursor(self.log, self.rows)


def test_hana_adapter_generates_expected_sql_and_params():
    conn = FakeConn(rows=[("c1", 0.9), ("c2", 0.5)])
    s = HanaVectorStore(conn, "POLICY_CHUNKS", dim=4)
    s.create_table()
    s.add(["c1"], np.array([[0.1, 0.2, 0.3, 0.4]]))
    res = s.search(np.array([0.1, 0.2, 0.3, 0.4]), 2)
    create, insert, select = (c[0] for c in conn.log)
    assert 'REAL_VECTOR(4)' in create and 'TO_REAL_VECTOR(?)' in insert
    assert 'COSINE_SIMILARITY(VEC, TO_REAL_VECTOR(?))' in select and 'SELECT TOP 2' in select and 'ORDER BY SCORE DESC' in select
    assert json.loads(conn.log[1][1][1]) == [0.1, 0.2, 0.3, 0.4]     # vector passed as a bound JSON array, not interpolated
    assert res == [("c1", 0.9), ("c2", 0.5)]


@pytest.mark.parametrize("bad", ['x"; DROP TABLE y; --', "a b", "1abc", ""])
def test_hana_adapter_rejects_unsafe_table_names(bad):
    with pytest.raises(ValueError):
        HanaVectorStore(FakeConn(), bad, 4)
