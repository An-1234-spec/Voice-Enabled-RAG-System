"""
embeddings/embedder.py

Batch embedding using sentence-transformers/all-MiniLM-L6-v2 (384d, ~80MB,
fast CPU inference — per the plan's rationale: strong MS MARCO retrieval
quality for its size, no GPU required).

Consumes the Chunk objects produced by any chunking/*.py strategy directly
(embed_chunks), or raw strings (embed_texts) for one-off query embedding at
query time.

Design notes:
  - Model loading is lazy (only happens on first .embed_texts()/.dim call),
    so importing this module doesn't require sentence-transformers to be
    installed or trigger a model download.
  - `encode_fn` can be injected to bypass SentenceTransformer entirely —
    useful for unit testing without a GPU/network, and is how this file's
    own __main__ demo runs in an offline sandbox.
  - CUDA is auto-detected and used if available, falling back to CPU
    (matches the plan's "fast enough on CPU" expectation either way — your
    RTX 4050's 6GB VRAM is complete overkill for a 22M-param model, so
    don't worry about a repeat of the CG-HTMN VRAM issues here).
  - normalize_embeddings=True by default: FAISS IndexFlatIP (planned next)
    computes inner product, so pre-normalized vectors turn that into
    cosine similarity for free.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

try:
    from chunking.base import Chunk
except ImportError:  # pragma: no cover
    Chunk = None  # allows standalone use without the chunking package present

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EncodeFn = Callable[[List[str]], "Sequence[Sequence[float]]"]


@dataclass
class EmbeddingResult:
    """Bundles vectors with the IDs/metadata needed to build a FAISS index."""

    ids: List[str]
    vectors: "object"  # np.ndarray, shape (n, dim) — typed loosely so numpy stays optional at import time
    dim: int
    model_name: str
    elapsed_sec: float
    metadata: List[dict] = field(default_factory=list)

    def save(self, path_prefix: Union[str, Path]) -> None:
        """
        Writes two files: `{path_prefix}.npy` (vectors) and
        `{path_prefix}.json` (ids + metadata + model_name), so
        indexing/faiss_index.py can load them independently of how they
        were produced.
        """
        import numpy as np

        path_prefix = Path(path_prefix)
        path_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{path_prefix}.npy", self.vectors)
        with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self.ids,
                    "dim": self.dim,
                    "model_name": self.model_name,
                    "metadata": self.metadata,
                },
                f,
            )

    @classmethod
    def load(cls, path_prefix: Union[str, Path]) -> "EmbeddingResult":
        import numpy as np

        path_prefix = Path(path_prefix)
        vectors = np.load(f"{path_prefix}.npy")
        with open(f"{path_prefix}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            ids=meta["ids"],
            vectors=vectors,
            dim=meta["dim"],
            model_name=meta["model_name"],
            elapsed_sec=0.0,
            metadata=meta.get("metadata", []),
        )


class Embedder:
    """Thin wrapper around sentence-transformers with lazy loading and DI."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        encode_fn: Optional[EncodeFn] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._device = device
        self._model = None
        self._encode_fn = encode_fn
        self._dim: Optional[int] = None

    def _load(self) -> None:
        if self._encode_fn is not None or self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Embedder needs the `sentence-transformers` package "
                "(`pip install sentence-transformers --break-system-packages`), "
                "or pass `encode_fn=` to inject your own embedding function."
            ) from e

        device = self._device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device

        self._model = SentenceTransformer(self.model_name, device=device)

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        if self._encode_fn is not None:
            # Determine dim from a throwaway probe encode.
            probe = self._encode_fn(["_dim_probe_"])
            self._dim = len(probe[0])
            return self._dim
        self._load()
        self._dim = self._model.get_sentence_embedding_dimension()
        return self._dim

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> "object":
        """Returns an (n, dim) float32 numpy array. Empty input -> shape (0, dim)."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self._encode_fn is not None:
            vectors = np.asarray(self._encode_fn(texts), dtype="float32")
            if self.normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms
            return vectors

        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def embed_query(self, query: str) -> "object":
        """Single-vector convenience wrapper for query-time embedding."""
        return self.embed_texts([query])[0]

    def embed_chunks(
        self,
        chunks: List["Chunk"],
        show_progress: bool = False,
        include_metadata: bool = True,
    ) -> EmbeddingResult:
        """
        Embeds a list of Chunk objects (from any chunking/*.py strategy).
        `chunk.chunk_id` becomes the ID FAISS/BM25 indices key against.
        """
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        metadata = []
        if include_metadata:
            for c in chunks:
                metadata.append(
                    {
                        "passage_id": c.passage_id,
                        "document_id": c.document_id,
                        "strategy": c.strategy,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        "parent_id": c.parent_id,
                        "language": c.language,
                        "query_id": c.query_id,
                        "is_selected": c.is_selected,
                    }
                )

        start = time.perf_counter()
        vectors = self.embed_texts(texts, show_progress=show_progress)
        elapsed = time.perf_counter() - start

        return EmbeddingResult(
            ids=ids,
            vectors=vectors,
            dim=vectors.shape[1] if len(vectors) else self.dim,
            model_name=self.model_name if self._encode_fn is None else "custom:encode_fn",
            elapsed_sec=elapsed,
            metadata=metadata,
        )


if __name__ == "__main__":
    # No internet in this sandbox to download all-MiniLM-L6-v2 from
    # Hugging Face, so the demo injects a deterministic fake encode_fn just
    # to prove the batching / save-load / Chunk-integration plumbing is
    # correct. ON YOUR MACHINE: drop `encode_fn=` entirely to use the real
    # model — everything else about the API is identical either way.
    import hashlib
    import struct

    def fake_encode_fn(texts: List[str]) -> List[List[float]]:
        """Deterministic 8-dim pseudo-embedding via hashing, for offline testing only."""
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            floats = [b / 255.0 for b in h[:8]]
            vecs.append(floats)
        return vecs

    embedder = Embedder(encode_fn=fake_encode_fn, normalize=True)

    print(f"Embedding dim: {embedder.dim}")

    texts = [
        "The Reserve Bank of India regulates monetary policy.",
        "Cricket is a popular sport in India.",
        "Goa is known for its beaches.",
    ]
    vectors = embedder.embed_texts(texts)
    print(f"embed_texts shape: {vectors.shape}")

    # Integration with the real Chunk dataclass, if chunking/ is importable.
    if Chunk is not None:
        chunks = [
            Chunk(
                chunk_id=f"c{i}",
                passage_id=f"p{i}",
                document_id=f"d{i}",
                text=t,
                token_count=len(t) // 4,
                strategy="demo",
                chunk_index=0,
                total_chunks=1,
                language="en",
            )
            for i, t in enumerate(texts)
        ]
        result = embedder.embed_chunks(chunks)
        print(
            f"embed_chunks: {len(result.ids)} ids, vectors shape "
            f"{result.vectors.shape}, elapsed {result.elapsed_sec:.4f}s"
        )

        # Save/load round-trip.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/test_embeddings"
            result.save(prefix)
            loaded = EmbeddingResult.load(prefix)
            assert loaded.ids == result.ids
            assert loaded.vectors.shape == result.vectors.shape
            assert (loaded.vectors == result.vectors).all()
            print("Save/load round-trip: OK")

    query_vec = embedder.embed_query("what does RBI do")
    print(f"embed_query shape: {query_vec.shape}")