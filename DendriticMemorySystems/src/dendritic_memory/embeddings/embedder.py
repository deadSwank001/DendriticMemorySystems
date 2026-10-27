"""Embedding layer — wraps sentence-transformers for vector generation."""

from __future__ import annotations

from typing import Optional

from dendritic_memory.storage.memory_record import MemoryRecord


class EmbeddingLayer:
    """
    Produces and caches sentence embeddings for memory records and queries.

    Uses sentence-transformers by default. Falls back to a simple bag-of-words
    hash if the library is not installed (useful for testing without GPU/network).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            self._model = None  # will use fallback

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return a float list."""
        self._load()
        if self._model is not None:
            vec = self._model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        return self._fallback_embed(text)

    def embed_record(self, record: MemoryRecord) -> MemoryRecord:
        """Populate record.embedding in place and return the record."""
        record.embedding = self.embed(record.content)
        return record

    def embed_batch(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Embed a list of records efficiently using batch inference."""
        self._load()
        texts = [r.content for r in records]
        if self._model is not None:
            vecs = self._model.encode(texts, convert_to_numpy=True, batch_size=64)
            for record, vec in zip(records, vecs):
                record.embedding = vec.tolist()
        else:
            for record in records:
                record.embedding = self._fallback_embed(record.content)
        return records

    @staticmethod
    def _fallback_embed(text: str, dim: int = 128) -> list[float]:
        """
        Deterministic bag-of-characters hash embedding.
        Used when sentence-transformers is unavailable (e.g., in CI without network).
        NOT suitable for production retrieval.
        """
        import hashlib

        vec = [0.0] * dim
        words = text.lower().split()
        for word in words:
            digest = hashlib.md5(word.encode()).digest()
            for i, byte in enumerate(digest[:dim]):
                vec[i % dim] += (byte - 128) / 128.0
        norm = sum(v**2 for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
