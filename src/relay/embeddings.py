"""Embedding manager — handles model loading, encoding, and hashing.

Provides both raw embedding functions (``embed``, ``sparse_embed``) and a
LlamaIndex-native ``RelayEmbedding`` wrapper so that relay's embedding
pipeline can be used directly within LlamaIndex.
"""

import functools
import hashlib
import struct
from typing import TYPE_CHECKING, Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer

from relay.config import CONFIG

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding

# Lazy-loaded singletons
_model: Optional[SentenceTransformer] = None
_sparse_model: Optional["SparseTextEmbedding"] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(CONFIG.model_name)
    return _model


def _get_sparse_model() -> "SparseTextEmbedding":
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        _sparse_model = SparseTextEmbedding(model_name=CONFIG.sparse_model_name)
    return _sparse_model


@functools.lru_cache(maxsize=1024)
def embed(text: str) -> list[float]:
    """Embed a text string into a dense vector."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


@functools.lru_cache(maxsize=1024)
def sparse_embed(text: str) -> tuple[list[int], list[float]]:
    """Compute a SPLADE sparse embedding — returns (indices, values)."""
    model = _get_sparse_model()
    results = list(model.embed([text]))
    sparse = results[0]
    return sparse.indices.tolist(), sparse.values.tolist()


def content_hash(text: str) -> str:
    """Compute SHA256 hash of document text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_hash(vector: list[float]) -> str:
    """Compute SHA256 hash of an embedding vector.

    Serializes floats as IEEE 754 double-precision bytes for deterministic hashing.
    """
    raw = b"".join(struct.pack("!d", v) for v in vector)
    return hashlib.sha256(raw).hexdigest()


class RelayEmbedding(BaseEmbedding):
    """LlamaIndex ``BaseEmbedding`` backed by relay's ``embed()``.

    Wraps relay's dense embedding pipeline (``sentence-transformers``) into a
    LlamaIndex-compatible embedding model, so relay can be used as the
    embedding backend in any LlamaIndex pipeline.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name=CONFIG.model_name, **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
