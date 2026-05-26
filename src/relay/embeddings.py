"""Embedding manager — handles model loading, encoding, and hashing."""

import functools
import hashlib
import struct
from typing import TYPE_CHECKING, Optional

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
