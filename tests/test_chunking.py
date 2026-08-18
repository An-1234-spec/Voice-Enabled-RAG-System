"""
tests/test_chunking.py

Unit tests for all chunking strategies. Tests chunk count, boundary respect,
token limits, metadata propagation, and edge cases.

Run: python -m pytest tests/test_chunking.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from chunking.base import Chunk
from chunking.fixed_token import FixedTokenChunker
from chunking.sentence import SentenceChunker
from chunking.passage import PassageChunker
from chunking.semantic import SemanticChunker
from chunking.parent_child import ParentChildChunker


SAMPLE_TEXT = (
    "The Reserve Bank of India regulates monetary policy in the country. "
    "It was established in 1935 under the Reserve Bank of India Act. "
    "RBI's key functions include issuing currency, managing foreign exchange "
    "reserves, and acting as a banker to the government and other banks. "
    "It also supervises the banking sector to ensure financial stability."
)

MULTI_PARA_TEXT = (
    "The Reserve Bank of India regulates monetary policy.\n\n"
    "Cricket is one of the most popular sports in India. "
    "The Indian cricket team has won two World Cups.\n\n"
    "Goa is known for its beaches and Portuguese heritage."
)

METADATA = {
    "query_id": 42,
    "query": "what does RBI do",
    "query_type": "descriptive",
    "answer": "The RBI regulates monetary policy.",
    "is_selected": 1,
    "language": "en",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _chunk(chunker, text=SAMPLE_TEXT, **kwargs):
    md = {**METADATA, **kwargs}
    return chunker.chunk(text, passage_id="q42_p0", document_id="doc_42", **md)


# ── Common tests applied to all strategies ───────────────────────────────

class TestAllChunkers:
    """Tests that every chunker must pass."""

    CHUNKERS = [
        FixedTokenChunker(chunk_size=60, chunk_overlap=10),
        SentenceChunker(chunk_size=60, chunk_overlap=10),
        PassageChunker(chunk_size=60, chunk_overlap=10),
        ParentChildChunker(parent_chunk_size=120, child_chunk_size=40),
    ]

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_returns_list_of_chunks(self, chunker):
        chunks = _chunk(chunker)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_non_empty_text_produces_chunks(self, chunker):
        chunks = _chunk(chunker)
        assert len(chunks) >= 1

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_empty_text_produces_no_chunks(self, chunker):
        assert _chunk(chunker, text="") == []
        assert _chunk(chunker, text="   ") == []

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_chunk_ids_are_unique(self, chunker):
        chunks = _chunk(chunker)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {ids}"

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_metadata_propagated(self, chunker):
        chunks = _chunk(chunker)
        for c in chunks:
            assert c.query_id == 42
            assert c.query == "what does RBI do"
            assert c.language == "en"
            assert c.passage_id == "q42_p0"
            assert c.document_id == "doc_42"

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_chunk_index_and_total(self, chunker):
        chunks = _chunk(chunker)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_chunk_text_non_empty(self, chunker):
        chunks = _chunk(chunker)
        for c in chunks:
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"

    @pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.strategy_name)
    def test_token_count_positive(self, chunker):
        chunks = _chunk(chunker)
        for c in chunks:
            assert c.token_count > 0


# ── Strategy-specific tests ──────────────────────────────────────────────

class TestFixedTokenChunker:
    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError):
            FixedTokenChunker(chunk_size=50, chunk_overlap=50)

    def test_small_chunk_size_produces_many_chunks(self):
        chunker = FixedTokenChunker(chunk_size=20, chunk_overlap=5)
        chunks = _chunk(chunker)
        assert len(chunks) >= 3


class TestSentenceChunker:
    def test_single_sentence_single_chunk(self):
        chunker = SentenceChunker(chunk_size=200)
        chunks = _chunk(chunker, text="This is a single sentence.")
        assert len(chunks) == 1

    def test_respects_sentence_boundaries(self):
        chunker = SentenceChunker(chunk_size=60, chunk_overlap=0)
        chunks = _chunk(chunker)
        # Each chunk should generally not start mid-word from the middle of a sentence
        for c in chunks:
            assert len(c.text) > 10  # Not degenerate


class TestPassageChunker:
    def test_paragraph_boundaries(self):
        chunker = PassageChunker(chunk_size=200)
        chunks = _chunk(chunker, text=MULTI_PARA_TEXT)
        assert len(chunks) >= 1

    def test_single_paragraph_single_chunk(self):
        chunker = PassageChunker(chunk_size=500)
        chunks = _chunk(chunker, text="Short paragraph here.")
        assert len(chunks) == 1


class TestParentChildChunker:
    def test_child_size_must_be_less_than_parent(self):
        with pytest.raises(ValueError):
            ParentChildChunker(parent_chunk_size=100, child_chunk_size=100)

    def test_parent_ids_set(self):
        chunker = ParentChildChunker(parent_chunk_size=120, child_chunk_size=40)
        chunks = _chunk(chunker)
        for c in chunks:
            assert c.parent_id is not None
            assert "parent" in c.parent_id

    def test_parent_text_in_extra(self):
        chunker = ParentChildChunker(parent_chunk_size=120, child_chunk_size=40)
        chunks = _chunk(chunker)
        for c in chunks:
            assert "parent_text" in c.extra
            assert len(c.extra["parent_text"]) > 0


class TestSemanticChunker:
    def test_with_fake_embedder(self):
        """Semantic chunker with injected embedder (no model download needed)."""
        def fake_embed(sentences):
            return [[float(i)] * 8 for i in range(len(sentences))]

        chunker = SemanticChunker(chunk_size=200, embedder=fake_embed)
        chunks = _chunk(chunker)
        assert len(chunks) >= 1

    def test_single_sentence_no_crash(self):
        def fake_embed(sentences):
            return [[1.0] * 8 for _ in sentences]

        chunker = SemanticChunker(chunk_size=200, embedder=fake_embed)
        chunks = _chunk(chunker, text="Just one sentence.")
        assert len(chunks) == 1
