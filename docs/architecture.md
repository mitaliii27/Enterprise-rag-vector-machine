# Architecture

```mermaid
flowchart LR
    D[Markdown policy docs] --> C[Heading-aware chunker<br/>contextual header + sentence overlap]
    C --> S[Sparse index<br/>TF-IDF cosine]
    C --> E[Embedder<br/>LSA stand-in | AI Core model]
    E --> V[(Vector store<br/>NumPy | FAISS | HANA Cloud)]
    Q[Question] --> R{Retriever<br/>sparse / dense / hybrid RRF}
    S --> R
    V --> R
    R --> T{top-hit confidence<br/>>= calibrated threshold?}
    T -- no --> X["I don't know based on the provided documents."]
    T -- yes --> G[Generator<br/>extractive | LLM]
    G --> A[Answer + citations]
```

## Design decisions
| Decision | Why |
|---|---|
| Contextual header on every chunk (`Doc title - Section:`) | A retrieved chunk should make sense alone; helps both lexical and dense matching. |
| Hybrid retrieval via reciprocal-rank fusion | Robust to score-scale differences; no tuning of mixing weights. (At this corpus size it ties sparse/dense - see report.) |
| Abstain on low lexical confidence | Wrong-but-confident answers are the costly failure in an enterprise setting. Threshold calibrated on a dev split, reported on a test split. |
| Stores behind a `VectorStore` protocol | Same pipeline runs on NumPy, FAISS and (adapter) HANA Cloud - swap without touching retrieval code. |
| Extractive generator by default | Zero hallucination by construction and no API needed; the `LLMGenerator` prompt enforces grounding + citations + an exact refusal sentence. |
| Gold-evidence integrity test | A typo in the QA set must not silently change metrics. |

## Mapping to the SAP stack in the JD
| Here | On SAP BTP | Status |
|---|---|---|
| `NumpyVectorStore` / `FaissVectorStore` | SAP HANA Cloud vector engine | `HanaVectorStore` adapter written; **SQL-generation tested with a fake connection only** |
| `LsaEmbedder` | Embedding model served via SAP AI Core | Not wired |
| `ExtractiveGenerator` / `LLMGenerator(complete=...)` | LLM via AI Core / generative AI hub | Prompt builder tested; no live model |
| `app/api.py` | Cloud Foundry app | Not deployed |
