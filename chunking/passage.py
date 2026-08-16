"""
chunking/passage.py

Passage/paragraph-aware chunking strategy.

MSMARCO-XI passages are usually already short, single-paragraph units, so this
chunker's real job is:
  1. Keep natural paragraph boundaries intact when a passage happens to be
     multi-paragraph.
  2. Merge consecutive short paragraphs so we don't emit tiny, low-signal
     chunks that hurt retrieval precision.
  3. Split any paragraph that overflows chunk_size into sentence-bounded
     sub-chunks, so nothing gets silently truncated at embedding time.

Note on chunk_overlap: unlike fixed_token.py, this strategy does not apply
token overlap between chunks — paragraph/sentence boundaries already provide
natural separation, and overlapping would re-merge content we deliberately
split apart. The constructor still accepts chunk_overlap for interface/config
consistency with the other strategies, but it's a no-op here.
"""

from __future__ import annotations

import re
from typing import List

from chunking.base import BaseChunker, Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u2018\u201c])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _PARAGRAPH_SPLIT_RE.split(text)
    # Passages with no blank-line breaks (the common MSMARCO case) come back
    # as a single "paragraph" — that's expected and correct.
    return [p.strip() for p in parts if p.strip()]


class PassageChunker(BaseChunker):
    """
    Paragraph/passage-aware chunker.

    Strategy:
      - Split text on blank-line paragraph boundaries.
      - Greedily merge consecutive paragraphs into a chunk while staying
        under `chunk_size` tokens (estimated via BaseChunker._estimate_tokens).
      - If a single paragraph alone exceeds `chunk_size`, fall back to
        sentence-level splitting for just that paragraph.
      - `min_chunk_tokens` folds any resulting chunk that's too small into
        its neighbour instead of emitting a near-empty chunk.
    """

    def __init__(
        self,
        chunk_size: int = 384,
        chunk_overlap: int = 50,
        min_chunk_tokens: int = 20,
    ):
        super().__init__(strategy_name="passage")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  # unused, kept for config consistency
        self.min_chunk_tokens = min_chunk_tokens

    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        units: List[str] = []
        for para in paragraphs:
            if self._estimate_tokens(para) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(self._split_oversized_paragraph(para))

        raw_texts = self._greedy_merge(units)
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

    def _split_oversized_paragraph(self, paragraph: str) -> List[str]:
        """Sentence-bound sub-split for a paragraph that alone exceeds chunk_size."""
        sentences = _split_sentences(paragraph)
        if not sentences:
            return [paragraph]

        pieces: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current and current_tokens + sent_tokens > self.chunk_size:
                pieces.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _greedy_merge(self, units: List[str]) -> List[str]:
        """Merge consecutive paragraph/sentence-group units under chunk_size."""
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._estimate_tokens(unit)
            if current and current_tokens + unit_tokens > self.chunk_size:
                chunks.append("\n\n".join(current))
                current = [unit]
                current_tokens = unit_tokens
            else:
                current.append(unit)
                current_tokens += unit_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _merge_small_trailing(self, chunks: List[str]) -> List[str]:
        """Fold any undersized chunk into a neighbour (checks both directions)."""
        if len(chunks) <= 1:
            return chunks

        merged: List[str] = []
        for c in chunks:
            if merged and self._estimate_tokens(c) < self.min_chunk_tokens:
                combined = merged[-1] + "\n\n" + c
                if self._estimate_tokens(combined) <= self.chunk_size * 1.15:
                    merged[-1] = combined
                    continue
            merged.append(c)

        # The backward pass above can't fix an undersized FIRST chunk (there's
        # nothing before it to merge into) — handle that case separately.
        if len(merged) > 1 and self._estimate_tokens(merged[0]) < self.min_chunk_tokens:
            combined = merged[0] + "\n\n" + merged[1]
            if self._estimate_tokens(combined) <= self.chunk_size * 1.15:
                merged = [combined] + merged[2:]

        return merged


if __name__ == "__main__":
    sample = (
        "The Reserve Bank of India regulates monetary policy in the country. "
        "It was established in 1935 under the Reserve Bank of India Act.\n\n"
        "RBI's key functions include issuing currency, managing foreign exchange "
        "reserves, and acting as a banker to the government and other banks. "
        "It also supervises the banking sector to ensure financial stability. "
        "The central bank plays a crucial role in controlling inflation through "
        "repo rate adjustments and other monetary tools that influence liquidity "
        "across the Indian economy.\n\n"
        "Ok."
    )

    chunker = PassageChunker(chunk_size=60, chunk_overlap=10, min_chunk_tokens=5)
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
        print(f"    {c.text[:80]}{'...' if len(c.text) > 80 else ''}")
        print()

    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk repr: {chunks[0]}")