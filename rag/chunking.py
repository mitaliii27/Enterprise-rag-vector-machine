"""Heading-aware chunking: split on markdown headings, then pack sentences up to max_chars.

Each chunk text is prefixed with "<doc title> - <section>:" (contextual header) so a chunk
is self-describing when retrieved on its own. Consecutive chunks inside one section share
`overlap_sentences` sentences so facts on a boundary are not lost.
"""
import re
from dataclasses import dataclass
from pathlib import Path

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    title: str
    section: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title} - {self.section}: {self.body}"

    @property
    def citation(self) -> str:
        return f"{self.doc} > {self.section}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(" ".join(text.split())) if s.strip()]


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 300, overlap_sentences: int = 1) -> list[Chunk]:
    title, section, buf, sections = doc_id, "Overview", [], []
    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if buf:
                sections.append((section, " ".join(buf)))
            section, buf = line[3:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((section, " ".join(buf)))

    chunks: list[Chunk] = []
    for sec, text in sections:
        sents, cur, n = split_sentences(text), [], 0
        pieces: list[list[str]] = []
        for s in sents:
            if cur and sum(len(x) + 1 for x in cur) + len(s) > max_chars:
                pieces.append(cur)
                cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur.append(s)
        if cur:
            pieces.append(cur)
        for i, p in enumerate(pieces):
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, sec, " ".join(p)))
    return chunks


def load_corpus(corpus_dir: str | Path, max_chars: int = 300, overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        chunks += chunk_markdown(path.stem, path.read_text(), max_chars, overlap_sentences)
    return chunks
