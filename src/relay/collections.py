"""Qdrant collection setup and client management.

Creates and ensures existence of:
  - relay_documents  (named vectors: 'semantic' @ 384d, 'sparse' SPLADE)
  - relay_epochs     (1d dummy vector, payload-only storage)
  - relay_retrieval_logs (1d dummy vector, payload-only storage)
"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from relay.config import CONFIG

# Module-level client singleton
_client: Optional[QdrantClient] = None

# Cache for collection_has_sparse() — collection schema never changes at runtime
_has_sparse_cache: dict[str, bool] = {}


def get_client() -> QdrantClient:
    """Get or create the Qdrant client singleton."""
    global _client
    if _client is None:
        _client = QdrantClient(host=CONFIG.qdrant_host, port=CONFIG.qdrant_port)
    return _client


def _collection_exists(client: QdrantClient, name: str) -> bool:
    """Check if a collection already exists."""
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def collection_has_sparse(client: QdrantClient, collection_name: str) -> bool:
    """Return True if the collection has a 'sparse' named vector configured.

    Qdrant does not support adding sparse vectors to an existing collection —
    they must be present at creation time. Callers use this to decide whether
    to store/query sparse vectors or fall back to dense-only.

    Result is cached — the collection schema never changes at runtime.
    """
    if collection_name in _has_sparse_cache:
        return _has_sparse_cache[collection_name]

    try:
        info = client.get_collection(collection_name)
        sparse = getattr(info.config.params, "sparse_vectors", None) or {}
        result = "sparse" in sparse
    except Exception:
        result = False

    _has_sparse_cache[collection_name] = result
    return result


def invalidate_has_sparse_cache(collection_name: str) -> None:
    """Invalidate the sparse cache entry for a collection.

    Called when a collection's schema may have changed (e.g., during testing).
    """
    _has_sparse_cache.pop(collection_name, None)


def ensure_collections() -> QdrantClient:
    """Ensure all relay collections exist in Qdrant. Returns the client."""
    client = get_client()
    if not _collection_exists(client, CONFIG.documents_collection):
        client.create_collection(
            collection_name=CONFIG.documents_collection,
            vectors_config={
                "semantic": VectorParams(
                    size=CONFIG.semantic_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        # Create payload indices for temporal filtering
        for field_name, schema_type in [
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("epoch_id", PayloadSchemaType.INTEGER),
            ("valid_from", PayloadSchemaType.KEYWORD),
            ("valid_to", PayloadSchemaType.KEYWORD),
            ("superseded_by", PayloadSchemaType.KEYWORD),
            ("supersedes", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=CONFIG.documents_collection,
                field_name=field_name,
                field_schema=schema_type,
            )

    # relay_epochs
    if not _collection_exists(client, CONFIG.epochs_collection):
        client.create_collection(
            collection_name=CONFIG.epochs_collection,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        for field_name, schema_type in [
            ("epoch_id", PayloadSchemaType.INTEGER),
            ("tenant_id", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=CONFIG.epochs_collection,
                field_name=field_name,
                field_schema=schema_type,
            )

    # relay_retrieval_logs
    if not _collection_exists(client, CONFIG.logs_collection):
        client.create_collection(
            collection_name=CONFIG.logs_collection,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        for field_name, schema_type in [
            ("query_hash", PayloadSchemaType.KEYWORD),
            ("epoch_id", PayloadSchemaType.INTEGER),
            ("tenant_id", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=CONFIG.logs_collection,
                field_name=field_name,
                field_schema=schema_type,
            )

    return client
