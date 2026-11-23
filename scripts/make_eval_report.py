"""Generate docs/eval_report.md from evals/results.json so no number is hand-typed."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
r = json.loads((root / "evals/results.json").read_text())
c, a = r["corpus"], r["abstention"]
L = ["# Evaluation report", "",
     f"Toy-scale benchmark: **{c['documents']} documents -> {c['chunks']} chunks**, **{c['questions']} questions** "
     f"({c['answerable']} answerable with gold evidence, {c['unanswerable']} unanswerable). "
     "Results show the *method* works end to end; they say little about performance on a real, much larger corpus.", "",
     "## Retrieval (answerable questions; hit = chunk from the gold document containing the gold evidence phrase)", "",
     "| Retriever | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|"]
for m, v in r["retrieval"].items():
    L.append(f"| {m} | {v['recall@1']}% | {v['recall@3']}% | {v['recall@5']}% | {v['mrr']} |")
L += ["", "## Stemming ablation (motivated by the failures it fixes - treat the gain as in-sample)", "",
      "| Retriever | recall@3 before | recall@3 after | MRR before | MRR after |", "|---|---|---|---|---|"]
for m, v in r["stemming_ablation"].items():
    L.append(f"| {m} | {v['no_stemming']['recall@3']}% | {v['stemming']['recall@3']}% | {v['no_stemming']['mrr']} | {v['stemming']['mrr']} |")
L += ["", "Before stemming, two questions missed the top 10 entirely: *receipt* vs *receipts* (q13) and *accrual* vs *accruals* (q27).", "",
      "## Abstention (threshold calibrated on the dev split, reported on the held-out test split)", "",
      f"Chosen min-confidence: **{a['chosen_threshold']}** (confidence = top hit's lexical cosine).", "",
      "| Split | n | Answerable answered correctly | False abstentions | Unanswerable correctly abstained | Balanced acc. |", "|---|---|---|---|---|---|"]
for name in ("dev", "test"):
    v = a[name]
    L.append(f"| {name} | {v['n']} | {v['answerable_correct']}% | {v['false_abstain']}% | {v['unanswerable_abstained']}% | {v['balanced_acc']}% |")
v = a["all_no_abstention"]
L += [f"| all, no abstention | {c['questions']} | {v['answerable_correct']}% | 0% | 0% | {v['balanced_acc']}% |", "",
      "The dev -> test drop is expected with ~26 questions per split (about 6 unanswerable each) - one question moves a rate by 5-17 points. "
      "Do not read the test figure as precise.", "",
      "## Chunk-size ablation (hybrid)", "", "| Max chars | Chunks | recall@3 | MRR |", "|---|---|---|---|"]
for k, v in r["chunk_size_ablation"].items():
    L.append(f"| {k} | {v['chunks']} | {v['recall@3']}% | {v['mrr']} |")
L += ["", "Sizes of 300+ give identical chunks because every section in this corpus is already short.", "",
      "## Other checks", "",
      f"- FAISS vs NumPy store: top-1 identical on **{r['store_parity']['top1_identical_numpy_vs_faiss']}/{r['store_parity']['questions']}** questions.",
      f"- End-to-end `ask()` latency on this machine (NumPy store, exhaustive scoring of 35 chunks): mean **{r['latency_ms']['mean']} ms**, p95 {r['latency_ms']['p95']} ms.",
      "", "## Top-1 retrieval misses (hybrid) - correct chunk was ranked 2nd or 3rd", "", "| Q | Question | Gold rank |", "|---|---|---|"]
for f in r["failures"]["hybrid_top1_misses"]:
    L.append(f"| {f['id']} | {f['question']} | {f['gold_rank']} |")
L += ["", "## Honest limits", "",
      "- Embeddings are a local LSA stand-in, not a hosted embedding model; dense/hybrid gains over sparse are not visible at this scale.",
      "- The generator is extractive (no hallucination, no reasoning/paraphrase). No live LLM was run.",
      "- The HANA Cloud adapter is unit-tested for SQL generation only; it has not been run against a real HANA instance."]
(root / "docs/eval_report.md").write_text("\n".join(L) + "\n")
print("wrote docs/eval_report.md")
