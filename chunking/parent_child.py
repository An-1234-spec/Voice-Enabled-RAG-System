"""
chunking/parent_child.py

Parent-child chunking strategy.

Idea: index small, precise "child" chunks for retrieval (good for matching a
narrow query against a narrow piece of text), but when a child chunk is
retrieved, hand the LLM the larger "parent" chunk it came from for fuller
context — small chunks retrieve well, big chunks generate well, and this
strategy avoids picking just one size.

What gets returned from chunk(): ONLY child Chunk objects — those are the
retrievable units that should be embedded and put in FAISS/BM25. Each child
carries:
  - `parent_id`             — the base.Chunk field this strategy exists for
  - `extra["parent_text"]`  — the full parent text, so orchestrator.py /
                               generation/prompts.py can expand context
                               without needing a separate parent lookup store
  - `extra["parent_index"]` — which parent (within this passage) it belongs to

Two-level packing, no overlap at either level (overlap muddies which parent
a child "really" belongs to, and children within the same parent already
share full context via the parent expansion):
  1. Sentences are greedily packed into parent-sized groups
     (parent_chunk_size tokens each).
  2. Each parent is then greedily packed into child-sized groups
     (child_chunk_size tokens each), still sentence-bounded.
"""

from __future__ import annotations

from typing import List, Tuple

from chunking.base import BaseChunker, Chunk

try:
    from chunking.sentence import _split_sentences
except ImportError:  # pragma: no cover
    import re

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u2018\u201c])")

    def _split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]


class ParentChildChunker(BaseChunker):
    """
    Two-level chunker: index children, expand to parents at generation time.
    """

    def __init__(
        self,
        parent_chunk_size: int = 512,
        child_chunk_size: int = 128,
    ):
        super().__init__(strategy_name="parent_child")
        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                f"child_chunk_size ({child_chunk_size}) must be smaller than "
                f"parent_chunk_size ({parent_chunk_size})."
            )
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        parents = self._pack_sentences(sentences, self.parent_chunk_size)

        # (child_text, parent_index, parent_text) triples, flattened across
        # all parents in this passage.
        flattened: List[Tuple[str, int, str]] = []
        for p_idx, parent_text in enumerate(parents):
            parent_sentences = _split_sentences(parent_text)
            children = self._pack_sentences(parent_sentences, self.child_chunk_size)
            for child_text in children:
                flattened.append((child_text, p_idx, parent_text))

        total = len(flattened)
        result: List[Chunk] = []
        for c_idx, (child_text, p_idx, parent_text) in enumerate(flattened):
            parent_chunk_id = f"{passage_id}_{self.strategy_name}_parent_{p_idx}"

            child = self._build_chunk(
                text=child_text,
                passage_id=passage_id,
                document_id=document_id,
                chunk_index=c_idx,
                total_chunks=total,
                parent_id=parent_chunk_id,
                **metadata,
            )
            child.extra["parent_text"] = parent_text
            child.extra["parent_index"] = p_idx
            result.append(child)

        return result

    # -- internal helpers ----------------------------------------------

    def _pack_sentences(self, sentences: List[str], size_limit: int) -> List[str]:
        """Greedy sentence packing under size_limit tokens, no overlap."""
        if not sentences:
            return []

        groups: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)

            # A single sentence longer than size_limit gets its own group
            # rather than being split further — child-level granularity is
            # already small, and further splitting would fragment meaning.
            if sent_tokens > size_limit:
                if current:
                    groups.append(" ".join(current))
                    current, current_tokens = [], 0
                groups.append(sent)
                continue

            if current and current_tokens + sent_tokens > size_limit:
                groups.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            groups.append(" ".join(current))

        return groups


if __name__ == "__main__":
    sample = (
        "The Reserve Bank of India regulates monetary policy in the country. "
        "It was established in 1935 under the Reserve Bank of India Act. "
        "RBI's key functions include issuing currency, managing foreign exchange "
        "reserves, and acting as a banker to the government and other banks. "
        "It also supervises the banking sector to ensure financial stability. "
        "The central bank plays a crucial role in controlling inflation through "
        "repo rate adjustments and other monetary tools that influence liquidity "
        "across the Indian economy."
    )

    chunker = ParentChildChunker(parent_chunk_size=60, child_chunk_size=20)
    chunks = chunker.chunk(
        sample,
        passage_id="q1_p0",
        document_id="doc_1",
        query_id=1,
        query="what does RBI do",
        language="en",
    )

    for c in chunks:
        print(f"[{c.chunk_index}/{c.total_chunks}] id={c.chunk_id} parent_id={c.parent_id} tokens={c.token_count}")
        print(f"    child : {c.text}")
        print(f"    parent: {c.extra['parent_text'][:100]}...")
        print()

    print(f"Total child chunks: {len(chunks)}")
    print(f"Distinct parents: {len(set(c.parent_id for c in chunks))}")