"""
chunking/sentence.py

Sentence-aware chunking strategy.

Unlike passage.py (which respects paragraph boundaries and treats overlap as
a no-op), this strategy operates purely on sentence boundaries and DOES apply
token overlap between consecutive chunks — that's the standard "sentence
window" approach and it's the main practical difference between the two
strategies for the benchmark in evaluation/chunking_eval.py.

Strategy:
  - Split text into sentences.
  - Greedily pack sentences into a chunk until adding the next sentence
    would exceed chunk_size tokens.
  - Start the next chunk by carrying back the last ~chunk_overlap tokens'
    worth of sentences from the previous chunk, so retrieval doesn't lose
    context right at a chunk boundary.
  - A single sentence longer than chunk_size on its own (rare, but MSMARCO
    machine-translated text can produce run-ons) is hard-split on word
    boundaries as a last resort — there's no smaller natural unit to fall
    back to the way passage.py can fall back to sentences.
"""

from __future__ import annotations

import re
from typing import List

from chunking.base import BaseChunker, Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u2018\u201c])")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Normalize paragraph breaks to spaces first — sentence chunking doesn't
    # care about paragraph structure, only passage.py does.
    text = re.sub(r"\s*\n\s*", " ", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceChunker(BaseChunker):
    """
    Sentence-boundary chunker with token-based overlap between chunks.
    """

    def __init__(
        self,
        chunk_size: int = 384,
        chunk_overlap: int = 50,
        min_chunk_tokens: int = 20,
    ):
        super().__init__(strategy_name="sentence")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens

    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Expand any single sentence that alone exceeds chunk_size.
        units: List[str] = []
        for sent in sentences:
            if self._estimate_tokens(sent) <= self.chunk_size:
                units.append(sent)
            else:
                units.extend(self._hard_split_words(sent))

        raw_texts = self._pack_with_overlap(units)
        raw_texts = self._merge_small_trailing(raw_texts)

        total = len(raw_texts)
        return [
            self._build_chunk(
                text=chunk_text,
                passage_id=passage_id,
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total,
                **metadata,
            )
            for idx, chunk_text in enumerate(raw_texts)
        ]

    # -- internal helpers ----------------------------------------------

    def _hard_split_words(self, sentence: str) -> List[str]:
        """Last-resort word-level split for a single oversized sentence."""
        words = sentence.split()
        pieces: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self._estimate_tokens(word + " ")
            if current and current_tokens + word_tokens > self.chunk_size:
                pieces.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _pack_with_overlap(self, sentences: List[str]) -> List[str]:
        """
        Greedily pack sentences into chunks under chunk_size tokens. Each
        new chunk (after the first) is seeded with trailing sentences from
        the previous chunk totaling roughly chunk_overlap tokens.
        """
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            if current and current_tokens + sent_tokens > self.chunk_size:
                chunks.append(" ".join(current))
                seeded = self._overlap_seed(current) + [sent]
                seeded_tokens = sum(self._estimate_tokens(s) for s in seeded)
                # Overlap is best-effort, chunk_size is a hard ceiling: if
                # seeding with the previous chunk's tail would itself blow
                # the budget (can happen when a unit is large relative to
                # chunk_size), drop the seed rather than overflow.
                if seeded_tokens > self.chunk_size and len(seeded) > 1:
                    current = [sent]
                    current_tokens = sent_tokens
                else:
                    current = seeded
                    current_tokens = seeded_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _overlap_seed(self, previous_sentences: List[str]) -> List[str]:
        """Pick trailing sentences from the previous chunk worth ~chunk_overlap tokens."""
        if self.chunk_overlap <= 0:
            return []

        seed: List[str] = []
        seed_tokens = 0
        for sent in reversed(previous_sentences):
            sent_tokens = self._estimate_tokens(sent)
            if seed and seed_tokens + sent_tokens > self.chunk_overlap:
                break
            seed.insert(0, sent)
            seed_tokens += sent_tokens

        return seed

    def _merge_small_trailing(self, chunks: List[str]) -> List[str]:
        """Fold any undersized chunk into a neighbour (checks both directions)."""
        if len(chunks) <= 1:
            return chunks

        merged: List[str] = []
        for c in chunks:
            if merged and self._estimate_tokens(c) < self.min_chunk_tokens:
                combined = merged[-1] + " " + c
                if self._estimate_tokens(combined) <= self.chunk_size * 1.15:
                    merged[-1] = combined
                    continue
            merged.append(c)

        # The backward pass above can't fix an undersized FIRST chunk (there's
        # nothing before it to merge into) — handle that case separately.
        if len(merged) > 1 and self._estimate_tokens(merged[0]) < self.min_chunk_tokens:
            combined = merged[0] + " " + merged[1]
            if self._estimate_tokens(combined) <= self.chunk_size * 1.15:
                merged = [combined] + merged[2:]

        return merged


if __name__ == "__main__":
    sample = (
        "The Reserve Bank of India regulates monetary policy in the country. "
        "It was established in 1935 under the Reserve Bank of India Act. "
        "RBI's key functions include issuing currency, managing foreign exchange "
        "reserves, and acting as a banker to the government and other banks. "
        "It also supervises the banking sector to ensure financial stability. "
        "The central bank plays a crucial role in controlling inflation through "
        "repo rate adjustments and other monetary tools that influence liquidity "
        "across the Indian economy. Ok."
    )

    chunker = SentenceChunker(chunk_size=60, chunk_overlap=15, min_chunk_tokens=5)
    chunks = chunker.chunk(
        sample,
        passage_id="q1_p0",
        document_id="doc_1",
        query_id=1,
        query="what does RBI do",
        language="en",
    )

    for c in chunks:
        print(f"[{c.chunk_index}/{c.total_chunks}] id={c.chunk_id} tokens={c.token_count}")
        print(f"    {c.text}")
        print()

    print(f"Total chunks: {len(chunks)}")