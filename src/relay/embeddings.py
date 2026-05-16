"""Embedding manager — handles model loading, encoding, and hashing."""

import hashlib
import struct
from typing import Optional

from sentence_transformers import SentenceTransformer

from relay.config import CONFIG

# Lazy-loaded singleton
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model lazily (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(CONFIG.model_name)
    return _model


def embed(text: str) -> list[float]:
    """Embed a text string into a dense vector."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def content_hash(text: str) -> str:
    """Compute SHA256 hash of document text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_hash(vector: list[float]) -> str:
    """Compute SHA256 hash of an embedding vector.

    Serializes floats as IEEE 754 double-precision bytes for deterministic hashing.
    """
    raw = b"".join(struct.pack("!d", v) for v in vector)
    return hashlib.sha256(raw).hexdigest()
