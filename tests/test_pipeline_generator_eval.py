import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_api
from rag.chunking import split_sentences
from rag.generator import LLMGenerator, build_prompt
from rag.pipeline import ABSTAIN_TEXT, ROOT, RagPipeline


@pytest.fixture(scope="module")
def pipe():
    return RagPipeline.build(mode="hybrid", min_confidence=0.22)


@pytest.mark.parametrize("question,doc", [
    ("At what value do I need three competitive quotes?", "procurement_policy"),
    ("How many vacation days do full-timers get each year?", "hr_leave_policy"),
    ("What is the minimum purchase price for capitalising an item?", "asset_capitalization"),
    ("Above what amount is a receipt mandatory?", "travel_expense_policy"),   # needs the plural-unifying stemmer
])
def test_right_document_in_top3(pipe, question, doc):
    assert doc in [h.chunk.doc for h in pipe.retriever.search(question, k=3)]


def test_abstains_on_out_of_domain_question(pipe):
    a = pipe.ask("What is the current share price?")
    assert a.abstained and a.text == ABSTAIN_TEXT and a.citations == []


def test_answers_with_citation_when_supported(pipe):
    a = pipe.ask("How long are blanket purchase orders valid?")
    assert not a.abstained and "12 months" in a.text and "procurement_policy" in a.text


def test_extractive_answer_sentences_come_only_from_retrieved_chunks(pipe):
    """Faithfulness property of the extractive generator: strip citation tags, every remaining sentence
    must appear verbatim in a retrieved chunk."""
    import re
    for question in ("What is the daily meal allowance?", "How long are blanket purchase orders valid?",
                     "How fast is a departing employee's access removed?"):
        a = pipe.ask(question)
        assert not a.abstained
        pool = {s for h in a.hits for s in split_sentences(h.chunk.body)}
        sentences = [x.strip() for x in re.sub(r"\s*\[[^\]]+ > [^\]]+\]", "|", a.text).split("|") if x.strip()]
        assert sentences and all(s in pool for s in sentences), (question, sentences)


def test_min_confidence_controls_abstention():
    strict = RagPipeline.build(min_confidence=0.99).ask("How long are blanket purchase orders valid?")
    lax = RagPipeline.build(min_confidence=0.0).ask("What is the current share price?")
    assert strict.abstained and not lax.abstained


def test_prompt_is_grounded_and_numbered(pipe):
    hits = pipe.retriever.search("hotel cap", k=2)
    p = build_prompt("What is the hotel cap?", hits)
    assert "ONLY the numbered context" in p and "[1]" in p and "[2]" in p and "I don't know based on the provided documents." in p
    seen = {}
    LLMGenerator(lambda prompt: seen.setdefault("p", prompt) and "ok").generate("q", hits)
    assert seen["p"].endswith("Answer:")


def test_gold_evidence_exists_in_corpus():
    """Protects eval integrity: a typo in qa_set.jsonl must not silently depress or inflate metrics."""
    for line in (ROOT / "evals" / "qa_set.jsonl").read_text().splitlines():
        q = json.loads(line)
        if q["answerable"]:
            assert q["evidence"].lower() in (ROOT / "corpus" / f"{q['doc']}.md").read_text().lower(), q["id"]


def test_api_ask_and_health(pipe):
    c = TestClient(create_api(pipe))
    assert c.get("/health").json()["chunks"] == len(pipe.chunks)
    body = c.post("/ask", json={"question": "What is the CFO's approval limit?"}).json()
    assert set(body) >= {"answer", "abstained", "confidence", "citations"}
    assert c.post("/ask", json={"question": "x"}).status_code == 422
