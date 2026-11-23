"""End-to-end pipeline: ingest corpus -> retrieve -> abstain-or-answer."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, load_corpus
from .embeddings import LsaEmbedder, SparseIndex
from .generator import ExtractiveGenerator
from .retriever import Hit, Retriever
from .vectorstore import FaissVectorStore, NumpyVectorStore

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "rag" / "calibration.json"
FALLBACK_MIN_CONFIDENCE = 0.15
ABSTAIN_TEXT = "I don't know based on the provided documents."


def default_min_confidence() -> float:
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text())["min_confidence"]
    return FALLBACK_MIN_CONFIDENCE


@dataclass
class Answer:
    text: str
    abstained: bool
    confidence: float
    hits: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def citations(self) -> list[str]:
        return [] if self.abstained else [h.chunk.citation for h in self.hits]


class RagPipeline:
    def __init__(self, chunks: list[Chunk], retriever: Retriever, generator, min_confidence: float):
        self.chunks, self.retriever, self.generator, self.min_confidence = chunks, retriever, generator, min_confidence

    @classmethod
    def build(cls, corpus_dir=ROOT / "corpus", mode: str = "hybrid", store: str = "numpy", max_chars: int = 300,
              generator=None, min_confidence: float | None = None, stem: bool = True) -> "RagPipeline":
        chunks = load_corpus(corpus_dir, max_chars=max_chars)
        texts = [c.text for c in chunks]
        sparse = SparseIndex(stem=stem).fit(texts)
        emb = LsaEmbedder(stem=stem).fit(texts)
        vs = FaissVectorStore(emb.dim) if store == "faiss" else NumpyVectorStore()
        vs.add([c.chunk_id for c in chunks], emb.embed(texts))
        retriever = Retriever(chunks, sparse, emb, vs, mode=mode)
        return cls(chunks, retriever, generator or ExtractiveGenerator(),
                   default_min_confidence() if min_confidence is None else min_confidence)

    def ask(self, question: str, k: int = 3) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=k)
        confidence = hits[0].sparse_score if hits else 0.0
        if confidence < self.min_confidence:
            return Answer(ABSTAIN_TEXT, True, confidence, hits, (time.perf_counter() - t0) * 1000)
        return Answer(self.generator.generate(question, hits), False, confidence, hits, (time.perf_counter() - t0) * 1000)
