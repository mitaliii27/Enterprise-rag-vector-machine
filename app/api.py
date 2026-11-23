"""HTTP wrapper for the RAG pipeline (deployable as a Cloud Foundry app; not deployed here)."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.pipeline import RagPipeline


class AskIn(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)


def create_api(pipeline: RagPipeline | None = None) -> FastAPI:
    api = FastAPI(title="Enterprise Knowledge RAG")
    pipe = pipeline or RagPipeline.build()

    @api.get("/health")
    def health():
        return {"status": "ok", "chunks": len(pipe.chunks), "min_confidence": pipe.min_confidence}

    @api.post("/ask")
    def ask(body: AskIn):
        a = pipe.ask(body.question, k=body.k)
        return {"answer": a.text, "abstained": a.abstained, "confidence": round(a.confidence, 3),
                "citations": a.citations, "latency_ms": round(a.latency_ms, 2)}

    return api


api = create_api()
