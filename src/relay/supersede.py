"""Supersession logic — link old→new documents, create new epoch with copies.

Original docs in their epoch are NEVER mutated (epochs are immutable).
Instead, we create copies of both docs in a new epoch with updated lineage metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from relay.collections import ensure_collections
from relay.config import CONFIG
from relay.epochs import create_epoch, get_next_epoch_id
from relay.models import DocumentPayload, SupersedeResult


def _find_doc_by_name(client, tenant_id: str, source_file: str) -> Optional[tuple]:
    """Find a document by its source filename. Returns (point_id, payload, vectors) or None."""
    results, _ = client.scroll(
        collection_name=CONFIG.documents_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="source_file", match=MatchValue(value=source_file)),
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    if not results:
        return None

    # Return the latest (not yet superseded) doc
    for point in results:
        if point.payload.get("superseded_by") is None:
            return (point.id, DocumentPayload(**point.payload), point.vector)

    # If all are superseded, return the last one
    point = results[-1]
    return (point.id, DocumentPayload(**point.payload), point.vector)


def _find_doc_by_id(client, tenant_id: str, doc_id: str) -> Optional[tuple]:
    """Find a document by its doc_id. Returns (point_id, payload, vectors) or None."""
    results, _ = client.scroll(
        collection_name=CONFIG.documents_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not results:
        return None
    point = results[0]
    return (point.id, DocumentPayload(**point.payload), point.vector)


def supersede(
    old_doc_ref: str,
    new_doc_ref: str,
    tenant_id: str,
) -> SupersedeResult:
    """Supersede an old document with a new one.

    Creates a new epoch containing copies of both documents with updated
    lineage metadata. The original epoch remains immutable.

    Returns:
        SupersedeResult with old_doc_id, new_doc_id, epoch details
    """
    client = ensure_collections()

    # Find old doc
    old_result = _find_doc_by_name(client, tenant_id, old_doc_ref)
    if old_result is None:
        old_result = _find_doc_by_id(client, tenant_id, old_doc_ref)
    if old_result is None:
        raise ValueError(f"Old document not found: {old_doc_ref}")

    _, old_doc, old_vectors = old_result

    # Find new doc
    new_result = _find_doc_by_name(client, tenant_id, new_doc_ref)
    if new_result is None:
        new_result = _find_doc_by_id(client, tenant_id, new_doc_ref)
    if new_result is None:
        raise ValueError(f"New document not found: {new_doc_ref}")

    _, new_doc, new_vectors = new_result

    # Determine new epoch
    next_epoch_id = get_next_epoch_id(client, tenant_id)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create COPY of old doc in new epoch with superseded_by + valid_to set
    old_copy = old_doc.model_copy(
        update={
            "superseded_by": new_doc.doc_id,
            "valid_to": now,
            "epoch_id": next_epoch_id,
        }
    )

    # Create COPY of new doc in new epoch with supersedes set
    new_copy = new_doc.model_copy(
        update={
            "supersedes": old_doc.doc_id,
            "epoch_id": next_epoch_id,
        }
    )

    # Upsert old doc copy (new point ID so original is preserved)
    client.upsert(
        collection_name=CONFIG.documents_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=old_vectors,
                payload=old_copy.model_dump(),
            )
        ],
    )

    # Upsert new doc copy
    client.upsert(
        collection_name=CONFIG.documents_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=new_vectors,
                payload=new_copy.model_dump(),
            )
        ],
    )

    # Create the new epoch
    epoch_data = create_epoch(client, tenant_id, CONFIG.model_name)

    return SupersedeResult(
        old_doc_id=old_doc.doc_id,
        new_doc_id=new_doc.doc_id,
        old_valid_to=now,
        epoch=epoch_data,
    )
