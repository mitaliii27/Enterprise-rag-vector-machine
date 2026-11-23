"""Answer generation.

ExtractiveGenerator picks the best-matching sentences from the retrieved chunks and cites
them. It cannot hallucinate (every word comes from a retrieved chunk) but also cannot
reason or paraphrase - it is the offline stand-in for an LLM answer step.
LLMGenerator wraps any text-in/text-out callable with a grounded-answer prompt; it is
unit-tested for prompt construction only - no live model was run.
"""
import math
import re
from collections import Counter
from typing import Callable

from .chunking import split_sentences
from .retriever import Hit

_STOP = set("a an the of for to in on at is are be do does did how what when which who where can i we my our it its by from with or and "
            "much many long soon fast often there any per".split())


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in _STOP]


class ExtractiveGenerator:
    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        q = set(_tokens(question))
        df = Counter()
        sents: list[tuple[str, Hit]] = []
        for h in hits:
            for s in split_sentences(h.chunk.body):
                sents.append((s, h)); df.update(set(_tokens(s)))
        n = len(sents) or 1
        scored = []
        for s, h in sents:
            toks = set(_tokens(s))
            overlap = sum(math.log(1 + n / df[t]) for t in toks & q)
            scored.append((overlap + 0.01 * h.sparse_score, s, h))
        scored.sort(key=lambda x: -x[0])
        picked = scored[: self.max_sentences]
        return " ".join(f"{s} [{h.chunk.citation}]" for _, s, h in picked)


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n".join(f"[{i + 1}] ({h.chunk.citation}) {h.chunk.body}" for i, h in enumerate(hits))
    return ("Answer the question using ONLY the numbered context. Cite sources like [1]. "
            "If the context does not contain the answer, reply exactly: I don't know based on the provided documents.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")


class LLMGenerator:
    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def generate(self, question: str, hits: list[Hit]) -> str:
        return self.complete(build_prompt(question, hits))
