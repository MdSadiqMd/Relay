"""Epoch management — create, list, and inspect semantic epochs."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from relay.config import CONFIG
from relay.merkle import (
    build_supersession_dag,
    compute_leaf,
    compute_merkle_root,
    toposort_docs,
)
from relay.models import DocumentPayload, EpochPayload


def _scroll_all_docs(client, tenant_id: str, epoch_id: int) -> list[DocumentPayload]:
    """Scroll through all documents for a given epoch and tenant."""
    all_docs: list[DocumentPayload] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=CONFIG.documents_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                    FieldCondition(key="epoch_id", match=MatchValue(value=epoch_id)),
                ]
            ),
            limit=10000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_docs.extend([DocumentPayload(**point.payload) for point in results])
        if next_offset is None:
            break
        offset = next_offset

    return all_docs


def _compute_epoch_merkle(docs: list[DocumentPayload]) -> str:
    """Compute Merkle root from a list of document payloads.

    Docs are topologically sorted by supersession DAG before hashing so the
    Merkle root encodes lineage structure, not arbitrary order.
    """
    dag = build_supersession_dag(docs)
    ordered_docs = toposort_docs(docs, dag)
    leaves = []
    for doc in ordered_docs:
        leaf = compute_leaf(
            doc_id=doc.doc_id,
            content_hash=doc.content_hash,
            embedding_hash=doc.embedding_hash,
            model_version=doc.model_version,
            valid_from=doc.valid_from,
            valid_to=doc.valid_to,
            supersedes=doc.supersedes,
        )
        leaves.append(leaf)
    return compute_merkle_root(leaves, ordered=True)


def get_next_epoch_id(client, tenant_id: str) -> int:
    """Determine the next epoch ID for a tenant.

    Uses client.count() which is O(1) with payload indexing — no full scroll.
    Epoch IDs are strictly sequential (1, 2, 3, ...) since epochs are immutable
    and never deleted, so count == max epoch ID.
    """
    count_result = client.count(
        collection_name=CONFIG.epochs_collection,
        count_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            ]
        ),
        exact=True,
    )
    return count_result.count + 1


def get_current_epoch_id(client, tenant_id: str) -> Optional[int]:
    """Get the current (latest) epoch ID for a tenant, or None if none exist."""
    count_result = client.count(
        collection_name=CONFIG.epochs_collection,
        count_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            ]
        ),
        exact=True,
    )
    if count_result.count == 0:
        return None
    return count_result.count


def create_epoch(
    client,
    tenant_id: str,
    model_version: str,
    leaf_hashes: list[bytes] | None = None,
    doc_ids: list[str] | None = None,
    epoch_id: int | None = None,
) -> EpochPayload:
    """Create a new semantic epoch.

    If leaf_hashes and doc_ids are provided, uses them directly (no Qdrant scroll).
    Otherwise falls back to scrolling all documents tagged with this epoch
    (backward compat for legacy callers).

    If epoch_id is provided, uses it instead of computing the next ID
    (avoids an extra scroll when the caller already knows the epoch ID).

    Returns the epoch as a validated Pydantic model.
    """
    if epoch_id is None:
        epoch_id = get_next_epoch_id(client, tenant_id)
    parent_epoch = epoch_id - 1 if epoch_id > 1 else None
    now = datetime.now(timezone.utc).isoformat()

    if leaf_hashes is not None and doc_ids is not None:
        merkle_root = compute_merkle_root(leaf_hashes, ordered=True)
        leaf_hashes_hex = [h.hex() for h in leaf_hashes]
        doc_count = len(leaf_hashes)
    else:
        docs = _scroll_all_docs(client, tenant_id, epoch_id)
        merkle_root = _compute_epoch_merkle(docs)
        leaf_hashes_hex = []
        doc_ids = [d.doc_id for d in docs]
        doc_count = len(docs)

    epoch = EpochPayload(
        epoch_id=epoch_id,
        tenant_id=tenant_id,
        created_at=now,
        merkle_root=merkle_root,
        doc_count=doc_count,
        model_version=model_version,
        parent_epoch=parent_epoch,
        leaf_hashes=leaf_hashes_hex,
        doc_ids=doc_ids or [],
    )

    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=CONFIG.epochs_collection,
        points=[
            PointStruct(
                id=point_id,
                vector=[0.0],  # dummy vector for payload-only collection
                payload=epoch.model_dump(),
            )
        ],
    )

    return epoch


def refresh_epoch_merkle(client, tenant_id: str, epoch_id: int) -> str:
    """Recompute and update the Merkle root for an existing epoch.

    Used after document mutations (supersession, updates).
    Returns the new Merkle root.
    """
    docs = _scroll_all_docs(client, tenant_id, epoch_id)
    new_root = _compute_epoch_merkle(docs)

    # Find the epoch point and update it
    results, _ = client.scroll(
        collection_name=CONFIG.epochs_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="epoch_id", match=MatchValue(value=epoch_id)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if results:
        point = results[0]
        epoch = EpochPayload(**point.payload)
        updated = epoch.model_copy(
            update={"merkle_root": new_root, "doc_count": len(docs)}
        )

        client.upsert(
            collection_name=CONFIG.epochs_collection,
            points=[
                PointStruct(
                    id=point.id,
                    vector=[0.0],
                    payload=updated.model_dump(),
                )
            ],
        )

    return new_root


def list_epochs(client, tenant_id: str) -> list[EpochPayload]:
    """List all epochs for a tenant, sorted by epoch_id."""
    results, _ = client.scroll(
        collection_name=CONFIG.epochs_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            ]
        ),
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    epochs = [EpochPayload(**p.payload) for p in results]
    epochs.sort(key=lambda e: e.epoch_id)
    return epochs


def get_epoch(client, tenant_id: str, epoch_id: int) -> Optional[EpochPayload]:
    """Get a single epoch by ID."""
    results, _ = client.scroll(
        collection_name=CONFIG.epochs_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="epoch_id", match=MatchValue(value=epoch_id)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        return EpochPayload(**results[0].payload)
    return None


def resolve_epoch_at(client, tenant_id: str, timestamp: str) -> Optional[EpochPayload]:
    """Resolve the active epoch at a given timestamp.

    Finds the latest epoch whose created_at <= timestamp.
    """
    all_epochs = list_epochs(client, tenant_id)
    if not all_epochs:
        return None

    candidates = [e for e in all_epochs if e.created_at <= timestamp]
    if not candidates:
        return all_epochs[0]
    return candidates[-1]
