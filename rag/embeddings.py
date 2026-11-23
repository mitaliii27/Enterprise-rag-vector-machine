"""Embedders and the lexical index.

LsaEmbedder (TF-IDF -> truncated SVD) is a LOCAL STAND-IN for a hosted embedding model:
no network or API key needed, fully reproducible. It must be fit on the corpus, which a
real embedding model does not need. In a BTP deployment replace it with a model served
through SAP AI Core (anything implementing `embed(texts) -> L2-normalised ndarray`).
"""
import re
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ENGLISH_STOP = frozenset(TfidfVectorizer(stop_words="english").get_stop_words())


def stem_token(t: str) -> str:
    """Tiny deterministic suffix stripper (plurals only) - enough to unify receipt/receipts, accrual/accruals."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t.endswith("us"):
        return t[:-1]
    return t


def make_tokenizer(stem: bool):
    def tokenize(text: str) -> list[str]:
        toks = [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower()) if t not in ENGLISH_STOP]
        return [stem_token(t) for t in toks] if stem else toks
    return tokenize


def make_vectorizer(stem: bool) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, tokenizer=make_tokenizer(stem), token_pattern=None,
                           lowercase=False)


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LsaEmbedder:
    def __init__(self, n_components: int = 48, stem: bool = True):
        self.n_components = n_components
        self.vec = make_vectorizer(stem)
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "LsaEmbedder":
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        assert self.svd is not None, "fit() first"
        return normalize(self.svd.transform(self.vec.transform(texts)))

    @property
    def dim(self) -> int:
        assert self.svd is not None
        return self.svd.n_components


class SparseIndex:
    """TF-IDF cosine over all chunks (a lexical retriever; corpus is small enough to score exhaustively)."""

    def __init__(self, stem: bool = True):
        self.vec = make_vectorizer(stem)
        self.matrix = None

    def fit(self, texts: list[str]) -> "SparseIndex":
        self.matrix = self.vec.fit_transform(texts)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()
