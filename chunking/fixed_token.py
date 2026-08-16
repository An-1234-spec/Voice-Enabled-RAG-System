"""
chunking/fixed_token.py

Fixed-token-size chunking strategy — the baseline against which passage.py
and sentence.py earn their keep in evaluation/chunking_eval.py.

Strategy:
  - Pure word-level sliding window, no sentence or paragraph awareness at all.
  - Pack words into a chunk until adding the next word would exceed
    chunk_size tokens (via BaseChunker._estimate_tokens).
  - Slide forward by (chunk_size - chunk_overlap) tokens' worth of words, so
    consecutive chunks share a fixed token-count overlap — this is the
    "textbook" RAG chunking baseline everyone compares against.
  - This will happily cut a chunk mid-sentence. That's expected and is
    exactly the weakness passage.py / sentence.py are meant to demonstrate
    an improvement over in the benchmark.
"""

from __future__ import annotations

from typing import List

from chunking.base import BaseChunker, Chunk


class FixedTokenChunker(BaseChunker):
    """
    Naive fixed-size sliding-window chunker (word-level granularity).
    """

    def __init__(self, chunk_size: int = 384, chunk_overlap: int = 50):
        super().__init__(strategy_name="fixed_token")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size}) or the window never advances."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        raw_texts = self._sliding_window(words)

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

    def _sliding_window(self, words: List[str]) -> List[str]:
        chunks: List[str] = []
        n = len(words)
        start = 0

        while start < n:
            end = start
            current_tokens = 0

            # Grow the window word-by-word until we'd exceed chunk_size.
            while end < n:
                next_tokens = self._estimate_tokens(words[end] + " ")
                if end > start and current_tokens + next_tokens > self.chunk_size:
                    break
                current_tokens += next_tokens
                end += 1

            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= n:
                break

            # Advance the window, holding back roughly chunk_overlap tokens'
            # worth of trailing words for the next chunk to re-include.
            overlap_tokens = 0
            back = end
            while back > start:
                w_tokens = self._estimate_tokens(words[back - 1] + " ")
                if overlap_tokens + w_tokens > self.chunk_overlap:
                    break
                overlap_tokens += w_tokens
                back -= 1

            next_start = back
            # Guarantee forward progress even in pathological cases
            # (e.g. one very long "word" eating the whole overlap budget).
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks


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

    chunker = FixedTokenChunker(chunk_size=20, chunk_overlap=5)
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