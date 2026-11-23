from rag.chunking import chunk_markdown, load_corpus, split_sentences
from rag.embeddings import LsaEmbedder, make_tokenizer, stem_token
from rag.pipeline import ROOT

MD = "# Travel Policy\n\n## Hotels\nSentence one is here. Sentence two is here. Sentence three is here. Sentence four is here.\n\n## Meals\nOnly one sentence."


def test_sentence_split_keeps_decimals_and_amounts():
    assert split_sentences("Rate is EUR 0.30 per km. Receipts above EUR 25 are required.") == \
        ["Rate is EUR 0.30 per km.", "Receipts above EUR 25 are required."]


def test_sections_become_chunks_with_contextual_header():
    chunks = chunk_markdown("travel", MD, max_chars=1000)
    assert [c.section for c in chunks] == ["Hotels", "Meals"]
    assert chunks[0].text.startswith("Travel Policy - Hotels: ")


def test_max_chars_respected_with_sentence_overlap():
    chunks = chunk_markdown("travel", MD, max_chars=60, overlap_sentences=1)
    hotels = [c for c in chunks if c.section == "Hotels"]
    assert len(hotels) >= 2
    assert all(len(c.body) <= 60 + 40 for c in hotels)           # packing budget + one sentence of overlap
    assert hotels[0].body.split(". ")[-1].strip(". ") in hotels[1].body  # last sentence repeated -> overlap


def test_corpus_loads_every_document_with_unique_ids():
    chunks = load_corpus(ROOT / "corpus")
    assert len({c.doc for c in chunks}) == 12
    assert len({c.chunk_id for c in chunks}) == len(chunks) and all(c.body.strip() for c in chunks)


def test_stemmer_unifies_plurals_without_mangling():
    assert stem_token("receipts") == stem_token("receipt")
    assert stem_token("accruals") == "accrual"
    assert stem_token("policies") == "policy"
    assert stem_token("business") == "business" and stem_token("status") == "status"


def test_tokenizer_drops_stopwords_and_keeps_numbers():
    toks = make_tokenizer(stem=True)("What is the limit above EUR 25,000?")
    assert "the" not in toks and "25,000" in toks


def test_lsa_embeddings_are_unit_norm():
    import numpy as np
    chunks = load_corpus(ROOT / "corpus")
    emb = LsaEmbedder().fit([c.text for c in chunks])
    v = emb.embed(["hotel cap per night", "password length"])
    assert v.shape == (2, emb.dim) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
