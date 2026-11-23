"""RAG evaluation: retrieval quality, abstention calibration (dev/test split), chunk-size ablation,
vector-store parity and latency. Every number in the README/report comes from results.json.

    python -m evals.run_evals
"""
import json
import statistics as st
from pathlib import Path

import numpy as np

from rag.pipeline import CALIBRATION, ROOT, RagPipeline

EVAL_DIR = Path(__file__).resolve().parent


def load_qa():
    return [json.loads(l) for l in (EVAL_DIR / "qa_set.jsonl").read_text().splitlines() if l.strip()]


def gold_rank(pipe: RagPipeline, q: dict, k: int = 10):
    for rank, h in enumerate(pipe.retriever.search(q["question"], k=k), start=1):
        if h.chunk.doc == q["doc"] and q["evidence"].lower() in h.chunk.text.lower():
            return rank
    return None


def retrieval_metrics(pipe: RagPipeline, qa) -> dict:
    ans = [q for q in qa if q["answerable"]]
    ranks = [gold_rank(pipe, q) for q in ans]
    n = len(ans)
    return {"n": n,
            "recall@1": round(100 * sum(r == 1 for r in ranks) / n, 1),
            "recall@3": round(100 * sum(r is not None and r <= 3 for r in ranks) / n, 1),
            "recall@5": round(100 * sum(r is not None and r <= 5 for r in ranks) / n, 1),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}, ranks


def per_item_answers(pipe: RagPipeline, qa):
    """Confidence + whether the extractive answer contains the evidence, independent of any threshold."""
    rows = []
    for q in qa:
        a = pipe.ask(q["question"])  # min_confidence irrelevant here: we read confidence and re-threshold below
        a_full = pipe.generator.generate(q["question"], a.hits)
        rows.append({"id": q["id"], "answerable": q["answerable"], "confidence": a.confidence,
                     "has_evidence": bool(q["answerable"] and q["evidence"].lower() in a_full.lower()),
                     "latency_ms": a.latency_ms})
    return rows


def score_at(rows, thr):
    ans = [r for r in rows if r["answerable"]]; un = [r for r in rows if not r["answerable"]]
    ans_ok = sum(r["confidence"] >= thr and r["has_evidence"] for r in ans) / max(1, len(ans))
    false_abstain = sum(r["confidence"] < thr for r in ans) / max(1, len(ans))
    abst_ok = sum(r["confidence"] < thr for r in un) / max(1, len(un))
    return {"answerable_correct": round(100 * ans_ok, 1), "false_abstain": round(100 * false_abstain, 1),
            "unanswerable_abstained": round(100 * abst_ok, 1), "balanced_acc": round(50 * (ans_ok + abst_ok), 1)}


def main():
    qa = load_qa()
    out = {"corpus": {}, "retrieval": {}, "failures": {}}

    pipes = {m: RagPipeline.build(mode=m, min_confidence=0.0) for m in ("sparse", "dense", "hybrid")}
    out["corpus"] = {"documents": len(list((ROOT / "corpus").glob("*.md"))), "chunks": len(pipes["hybrid"].chunks),
                     "questions": len(qa), "answerable": sum(q["answerable"] for q in qa),
                     "unanswerable": sum(not q["answerable"] for q in qa)}
    for m, p in pipes.items():
        out["retrieval"][m], ranks = retrieval_metrics(p, qa)
        if m == "hybrid":
            ans = [q for q in qa if q["answerable"]]
            out["failures"]["hybrid_recall@3_misses"] = [{"id": q["id"], "question": q["question"], "gold_rank": r}
                                                          for q, r in zip(ans, ranks) if r is None or r > 3]
            out["failures"]["hybrid_top1_misses"] = [{"id": q["id"], "question": q["question"], "gold_rank": r}
                                                      for q, r in zip(ans, ranks) if r != 1]

    # --- analyzer ablation: the fix motivated by the plural-mismatch misses ---------------------
    out["stemming_ablation"] = {}
    for m in ("sparse", "dense", "hybrid"):
        base, _ = retrieval_metrics(RagPipeline.build(mode=m, stem=False, min_confidence=0.0), qa)
        out["stemming_ablation"][m] = {"no_stemming": base, "stemming": out["retrieval"][m]}

    # --- abstention: calibrate on DEV (even index), report on TEST (odd index) ---------------
    hybrid = pipes["hybrid"]
    rows = per_item_answers(hybrid, qa)
    dev = [r for i, r in enumerate(rows) if i % 2 == 0]; test = [r for i, r in enumerate(rows) if i % 2 == 1]
    thresholds = [round(x, 2) for x in np.arange(0.0, 0.61, 0.01)]
    dev_curve = [{"threshold": t, **score_at(dev, t)} for t in thresholds]
    best = max(c["balanced_acc"] for c in dev_curve)
    best_ts = [c["threshold"] for c in dev_curve if c["balanced_acc"] == best]
    chosen = float(np.median(best_ts))
    CALIBRATION.write_text(json.dumps({"min_confidence": chosen, "calibrated_on": "dev split (even-indexed questions)",
                                       "dev_balanced_accuracy": best}, indent=2))
    out["abstention"] = {"chosen_threshold": chosen, "tied_best_thresholds": [float(min(best_ts)), float(max(best_ts))],
                         "dev": {"n": len(dev), **score_at(dev, chosen)}, "test": {"n": len(test), **score_at(test, chosen)},
                         "all_no_abstention": score_at(rows, 0.0), "dev_curve": dev_curve,
                         "test_points": [{"threshold": t, **score_at(test, t)} for t in thresholds]}
    out["answer_quality_no_abstention"] = {"answerable_contains_evidence_pct": round(
        100 * sum(r["has_evidence"] for r in rows if r["answerable"]) / out["corpus"]["answerable"], 1)}

    # --- chunk-size ablation (hybrid retriever) ------------------------------------------------
    out["chunk_size_ablation"] = {}
    for mc in (120, 300, 600, 1200):
        p = RagPipeline.build(mode="hybrid", max_chars=mc, min_confidence=0.0)
        m, _ = retrieval_metrics(p, qa)
        out["chunk_size_ablation"][str(mc)] = {"chunks": len(p.chunks), **m}

    # --- vector store parity: FAISS vs NumPy --------------------------------------------------
    pf = RagPipeline.build(mode="dense", store="faiss", min_confidence=0.0)
    same = sum(pf.retriever.search(q["question"], 1)[0].chunk.chunk_id == pipes["dense"].retriever.search(q["question"], 1)[0].chunk.chunk_id
               for q in qa)
    out["store_parity"] = {"questions": len(qa), "top1_identical_numpy_vs_faiss": same}

    # --- latency (end-to-end ask, numpy store, this machine) -----------------------------------
    lat = [r["latency_ms"] for r in rows]
    out["latency_ms"] = {"mean": round(st.mean(lat), 2), "p95": round(float(np.percentile(lat, 95)), 2), "n": len(lat)}

    (EVAL_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print("corpus:", out["corpus"])
    for m, v in out["retrieval"].items(): print(f"  {m:7s}", v)
    a = out["abstention"]
    print(f"threshold={a['chosen_threshold']} (tied best {a['tied_best_thresholds']})\n  dev ={a['dev']}\n  test={a['test']}\n  no-abstention={a['all_no_abstention']}")
    print("stemming ablation (recall@3 / mrr):", {m: (v["no_stemming"]["recall@3"], v["no_stemming"]["mrr"], "->", v["stemming"]["recall@3"], v["stemming"]["mrr"]) for m, v in out["stemming_ablation"].items()})
    print("chunk ablation:", {k: (v['chunks'], v['recall@3'], v['mrr']) for k, v in out['chunk_size_ablation'].items()})
    print("store parity:", out["store_parity"], "| latency:", out["latency_ms"])
    print("hybrid recall@3 misses:", [(f["id"], f["gold_rank"]) for f in out["failures"]["hybrid_recall@3_misses"]])


if __name__ == "__main__":
    main()
