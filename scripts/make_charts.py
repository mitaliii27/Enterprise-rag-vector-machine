"""Render docs/img/eval_summary.png from evals/results.json (run `make eval` first)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[1]
r = json.loads((root / "evals/results.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

# 1 - stemming ablation, recall@3 and MRR per retriever
modes = ["sparse", "dense", "hybrid"]
x = range(len(modes)); w = 0.36
before = [r["stemming_ablation"][m]["no_stemming"]["recall@3"] for m in modes]
after = [r["stemming_ablation"][m]["stemming"]["recall@3"] for m in modes]
b1 = ax[0].bar([i - w / 2 for i in x], before, w, label="no stemming", color="#b9b9b9")
b2 = ax[0].bar([i + w / 2 for i in x], after, w, label="+ plural stemmer", color="#2f7d6d")
for bars in (b1, b2):
    for b in bars:
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", fontsize=9)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(modes); ax[0].set_ylim(0, 112); ax[0].set_ylabel("recall@3 (%)")
ax[0].set_title("Retrieval recall@3 - 40 answerable questions\n(35-chunk toy corpus; fix was motivated by this set)", fontsize=10)
ax[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

# 2 - abstention curve (dev) + chosen threshold + test points
a = r["abstention"]
ts = [c["threshold"] for c in a["dev_curve"]]
ax[1].plot(ts, [c["answerable_correct"] for c in a["dev_curve"]], color="#2f7d6d", label="answerable answered correctly (dev)")
ax[1].plot(ts, [c["unanswerable_abstained"] for c in a["dev_curve"]], color="#7a5fb5", label="unanswerable abstained (dev)")
ax[1].plot(ts, [c["answerable_correct"] for c in a["test_points"]], color="#2f7d6d", ls=":", label="answerable correct (test)")
ax[1].plot(ts, [c["unanswerable_abstained"] for c in a["test_points"]], color="#7a5fb5", ls=":", label="unanswerable abstained (test)")
ax[1].axvline(a["chosen_threshold"], color="#c98a2b", lw=1.5)
ax[1].text(a["chosen_threshold"] + 0.01, 8, f"chosen\n{a['chosen_threshold']}", color="#c98a2b", fontsize=9)
ax[1].set_xlabel("min confidence (top-hit lexical cosine)"); ax[1].set_ylabel("%"); ax[1].set_ylim(0, 105)
ax[1].set_title("Abstention: threshold picked on dev, checked on test\n(n=26 each - small, noisy)", fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5, loc="center right")

# 3 - chunk size ablation
sizes = list(r["chunk_size_ablation"])
rec = [r["chunk_size_ablation"][s]["recall@3"] for s in sizes]
mrr = [r["chunk_size_ablation"][s]["mrr"] for s in sizes]
nch = [r["chunk_size_ablation"][s]["chunks"] for s in sizes]
ax[2].bar([f"{s}\n({n} chunks)" for s, n in zip(sizes, nch)], mrr, color="#c98a2b")
for i, (m, rc) in enumerate(zip(mrr, rec)):
    ax[2].text(i, m + 0.005, f"MRR {m:.3f}\nR@3 {rc:.0f}%", ha="center", fontsize=8.5)
ax[2].set_ylim(0, 1.05); ax[2].set_xlabel("max chunk size (chars)"); ax[2].set_ylabel("MRR")
ax[2].set_title("Chunk-size ablation (hybrid)\nflat above 300 chars: sections are short", fontsize=10)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = root / "docs/img/eval_summary.png"; out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
