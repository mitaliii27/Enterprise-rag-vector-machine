# Functional specification - Policy Q&A Assistant (sample go-live deliverable)
*Illustrative; all thresholds below are from the synthetic corpus.*

## 1. Business need
Employees ask routine policy questions (limits, deadlines, retention periods) and get inconsistent answers. The assistant answers from the approved policy set, with citations, and says so when it does not know.

## 2. Acceptance criteria
| ID | Criterion | How verified |
|---|---|---|
| QA-1 | Answers are grounded: every answer sentence appears verbatim in a cited chunk | `test_extractive_answer_sentences_come_only_from_retrieved_chunks` |
| QA-2 | Every answer carries `[document > section]` citations | `test_answers_with_citation_when_supported` |
| QA-3 | Out-of-scope questions return the exact refusal text, no citations | `test_abstains_on_out_of_domain_question` |
| QA-4 | The right document is in the top 3 for known questions | eval: recall@3 (see report) |
| QA-5 | Abstention threshold is calibrated on held-out data, not hand-set | `evals/run_evals.py` -> `rag/calibration.json` |
| QA-6 | Vector backend can be swapped without changing retrieval code | `VectorStore` protocol; numpy/FAISS parity test |

## 3. Out of scope
Multi-turn conversation, access control per document, document updates/re-indexing workflow, multilingual content.

## 4. Open items for a real engagement
Source systems and refresh cadence; per-user document permissions; which embedding/LLM endpoints are approved; PII handling in logs; acceptance thresholds agreed with the business (what abstention/false-abstention trade-off is acceptable).
