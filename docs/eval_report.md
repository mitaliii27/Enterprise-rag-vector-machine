# Evaluation report

Toy-scale benchmark: **12 documents -> 35 chunks**, **52 questions** (40 answerable with gold evidence, 12 unanswerable). Results show the *method* works end to end; they say little about performance on a real, much larger corpus.

## Retrieval (answerable questions; hit = chunk from the gold document containing the gold evidence phrase)

| Retriever | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| sparse | 82.5% | 100.0% | 100.0% | 0.908 |
| dense | 82.5% | 100.0% | 100.0% | 0.904 |
| hybrid | 82.5% | 100.0% | 100.0% | 0.904 |

## Stemming ablation (motivated by the failures it fixes - treat the gain as in-sample)

| Retriever | recall@3 before | recall@3 after | MRR before | MRR after |
|---|---|---|---|---|
| sparse | 95.0% | 100.0% | 0.883 | 0.908 |
| dense | 95.0% | 100.0% | 0.879 | 0.904 |
| hybrid | 95.0% | 100.0% | 0.879 | 0.904 |

Before stemming, two questions missed the top 10 entirely: *receipt* vs *receipts* (q13) and *accrual* vs *accruals* (q27).

## Abstention (threshold calibrated on the dev split, reported on the held-out test split)

Chosen min-confidence: **0.22** (confidence = top hit's lexical cosine).

| Split | n | Answerable answered correctly | False abstentions | Unanswerable correctly abstained | Balanced acc. |
|---|---|---|---|---|---|
| dev | 26 | 95.0% | 5.0% | 100.0% | 97.5% |
| test | 26 | 75.0% | 20.0% | 83.3% | 79.2% |
| all, no abstention | 52 | 97.5% | 0% | 0% | 48.8% |

The dev -> test drop is expected with ~26 questions per split (about 6 unanswerable each) - one question moves a rate by 5-17 points. Do not read the test figure as precise.

## Chunk-size ablation (hybrid)

| Max chars | Chunks | recall@3 | MRR |
|---|---|---|---|
| 120 | 74 | 95.0% | 0.874 |
| 300 | 35 | 100.0% | 0.904 |
| 600 | 35 | 100.0% | 0.904 |
| 1200 | 35 | 100.0% | 0.904 |

Sizes of 300+ give identical chunks because every section in this corpus is already short.

## Other checks

- FAISS vs NumPy store: top-1 identical on **52/52** questions.
- End-to-end `ask()` latency on this machine (NumPy store, exhaustive scoring of 35 chunks): mean **1.53 ms**, p95 1.92 ms.

## Top-1 retrieval misses (hybrid) - correct chunk was ranked 2nd or 3rd

| Q | Question | Gold rank |
|---|---|---|
| q04 | Which low-value catalogue orders skip approval? | 2 |
| q05 | How much can a Department Head sign off? | 2 |
| q06 | What is the CFO's approval limit? | 2 |
| q13 | Above what amount is a receipt mandatory? | 3 |
| q18 | How many days does it take to onboard a new vendor? | 3 |
| q32 | How fast is a departing employee's access removed? | 2 |
| q40 | What fee do we charge when a customer sends back a working product? | 2 |

## Honest limits

- Embeddings are a local LSA stand-in, not a hosted embedding model; dense/hybrid gains over sparse are not visible at this scale.
- The generator is extractive (no hallucination, no reasoning/paraphrase). No live LLM was run.
- The HANA Cloud adapter is unit-tested for SQL generation only; it has not been run against a real HANA instance.
