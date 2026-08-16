"""
chunking/semantic.py

Semantic chunking strategy — detects topic shifts via sentence-embedding
similarity rather than fixed size or surface punctuation.

Approach (percentile-breakpoint method, the standard technique for this):
  1. Split text into sentences.
  2. Embed every sentence in one batch call (cheap: all-MiniLM-L6-v2 is small).
  3. Compute cosine "distance" (1 - similarity) between each consecutive pair.
  4. Any gap whose distance exceeds the Nth percentile of all gaps in this
     passage is treated as a topic-shift breakpoint.
  5. Group sentences between breakpoints into a raw semantic group.
  6. Because groups can still come out larger than chunk_size, each group is
     re-packed with the same sentence-level greedy splitting passage.py uses
     for oversized paragraphs — semantic boundaries win first, token budget
     is enforced second.

Embedding model is NOT hardcoded to a specific SDK. Pass any callable via
`embedder=` with signature `List[str] -> List[Sequence[float]]`. If you don't
pass one, this lazily imports sentence-transformers and loads
all-MiniLM-L6-v2 on first use — the import only happens inside chunk(), so
this module stays importable even before you've pip-installed
sentence-transformers (relevant since embeddings/embedder.py hasn't been
built yet in the pipeline order).
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, List, Optional, Sequence

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


EmbedFn = Callable[[List[str]], List[Sequence[float]]]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: List[float], pct: float) -> float:
    """Simple linear-interpolation percentile, no numpy dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] + (s[int(c)] - s[int(f)]) * (k - f)


class SemanticChunker(BaseChunker):
    """
    Embedding-similarity chunker with a percentile breakpoint threshold.
    """

    def __init__(
        self,
        chunk_size: int = 384,
        min_chunk_tokens: int = 20,
        breakpoint_percentile: float = 85.0,
        embedder: Optional[EmbedFn] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(strategy_name="semantic")
        self.chunk_size = chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.model_name = model_name
        self._embedder = embedder
        self._model = None  # lazily loaded sentence-transformers model

    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            raw_texts = sentences  # 0 or 1 sentence: nothing to segment
        else:
            embeddings = self._embed(sentences)
            groups = self._group_by_similarity(sentences, embeddings)
            raw_texts = self._enforce_size_budget(groups)
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

    # -- embedding -------------------------------------------------------

    def _embed(self, sentences: List[str]) -> List[Sequence[float]]:
        if self._embedder is not None:
            return self._embedder(sentences)

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "SemanticChunker needs either an injected `embedder=` callable "
                    "or the `sentence-transformers` package installed "
                    "(`pip install sentence-transformers`)."
                ) from e
            self._model = SentenceTransformer(self.model_name)

        return self._model.encode(sentences, convert_to_numpy=False)

    # -- segmentation ------------------------------------------------------

    def _group_by_similarity(
        self, sentences: List[str], embeddings: List[Sequence[float]]
    ) -> List[str]:
        distances = [
            1.0 - _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]
        threshold = _percentile(distances, self.breakpoint_percentile)

        groups: List[str] = []
        current: List[str] = [sentences[0]]

        for i, dist in enumerate(distances):
            sent = sentences[i + 1]
            # Only break where the gap is both above the percentile threshold
            # AND strictly greater than the median gap — avoids treating a
            # near-uniform passage (all gaps ~equal) as having "shifts".
            if dist >= threshold and dist > statistics.median(distances):
                groups.append(" ".join(current))
                current = [sent]
            else:
                current.append(sent)

        if current:
            groups.append(" ".join(current))

        return groups

    def _enforce_size_budget(self, groups: List[str]) -> List[str]:
        """Re-split any semantic group that exceeds chunk_size, sentence-bounded."""
        result: List[str] = []
        for group in groups:
            if self._estimate_tokens(group) <= self.chunk_size:
                result.append(group)
                continue

            sub_sentences = _split_sentences(group)
            current: List[str] = []
            current_tokens = 0
            for sent in sub_sentences:
                sent_tokens = self._estimate_tokens(sent)
                if current and current_tokens + sent_tokens > self.chunk_size:
                    result.append(" ".join(current))
                    current = [sent]
                    current_tokens = sent_tokens
                else:
                    current.append(sent)
                    current_tokens += sent_tokens
            if current:
                result.append(" ".join(current))
        return result

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
    # No internet in this sandbox to pull a real sentence-transformers model.
    # A raw bag-of-words embedder is too crude to demo this correctly: two
    # sentences on the same topic with zero shared words (e.g. "Kohli is a
    # great batsman" vs "India won the World Cup") would register as
    # maximally distant, drowning the real signal in lexical noise. Real
    # sentence embeddings don't have that problem. This synthetic embedder
    # instead assigns each sentence a topic-cluster vector (+ noise) purely
    # so the demo has a meaningful similarity structure to detect against.
    # ON YOUR MACHINE: drop the `embedder=` arg entirely to use the real
    # all-MiniLM-L6-v2 model via sentence-transformers.
    import random

    _TOPIC_KEYWORDS = {
        "rbi": {"rbi", "reserve", "bank", "monetary", "currency", "reserves"},
        "cricket": {"cricket", "world", "cups", "kohli", "batsman", "team"},
        "goa": {"goa", "beaches", "nightlife", "portuguese", "tourist", "coast"},
    }

    def fake_topic_embedder(sentences: List[str]) -> List[List[float]]:
        rng = random.Random(42)
        vecs = []
        for s in sentences:
            words = set(w.strip(".,'").lower() for w in s.split())
            scores = {
                topic: len(words & kws) for topic, kws in _TOPIC_KEYWORDS.items()
            }
            best_topic = max(scores, key=scores.get)
            base = {"rbi": [1, 0, 0], "cricket": [0, 1, 0], "goa": [0, 0, 1]}[best_topic]
            noisy = [x + rng.uniform(-0.15, 0.15) for x in base]
            vecs.append(noisy)
        return vecs

    sample = (
        "The Reserve Bank of India regulates monetary policy in the country. "
        "It was established in 1935 under the Reserve Bank of India Act. "
        "RBI's key functions include issuing currency and managing reserves. "
        "Cricket is one of the most popular sports in India. "
        "The Indian cricket team has won two World Cups. "
        "Virat Kohli is regarded as one of the greatest modern batsmen. "
        "Goa is a popular tourist destination on India's west coast. "
        "It is known for its beaches, nightlife, and Portuguese heritage."
    )

    chunker = SemanticChunker(
        chunk_size=200,
        breakpoint_percentile=70.0,
        embedder=fake_topic_embedder,
    )
    chunks = chunker.chunk(
        sample,
        passage_id="q1_p0",
        document_id="doc_1",
        query_id=1,
        query="tell me about India",
        language="en",
    )

    for c in chunks:
        print(f"[{c.chunk_index}/{c.total_chunks}] id={c.chunk_id} tokens={c.token_count}")
        print(f"    {c.text}")
        print()

    print(f"Total chunks: {len(chunks)}")