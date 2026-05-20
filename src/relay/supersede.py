"""Supersession logic — link old→new documents, create new epoch with copies.

Original docs in their epoch are NEVER mutated (epochs are immutable).
Instead, we create copies of both docs in a new epoch with updated lineage metadata.

A new document can supersede multiple old documents (merge scenario), forming a
proper DAG rather than a linked list.
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

    for point in results:
        if point.payload.get("superseded_by") is None:
            return (point.id, DocumentPayload(**point.payload), point.vector)

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


def _resolve_doc(client, tenant_id: str, ref: str) -> tuple:
    """Resolve a doc ref (filename or doc_id). Raises ValueError if not found."""
    result = _find_doc_by_name(client, tenant_id, ref)
    if result is None:
        result = _find_doc_by_id(client, tenant_id, ref)
    if result is None:
        raise ValueError(f"Document not found: {ref}")
    return result


def supersede(
    old_doc_refs: list[str],
    new_doc_ref: str,
    tenant_id: str,
) -> SupersedeResult:
    """Supersede one or more old documents with a new one.

    Creates a new epoch containing copies of all involved documents with updated
    lineage metadata. The original epoch remains immutable.

    The new document's supersedes field becomes a list of all old doc IDs,
    forming a DAG node with multiple parents.

    Returns:
        SupersedeResult with old_doc_ids, new_doc_id, epoch details
    """
    client = ensure_collections()

    old_results = [_resolve_doc(client, tenant_id, ref) for ref in old_doc_refs]
    old_docs = [r[1] for r in old_results]
    old_vectors_list = [r[2] for r in old_results]

    _, new_doc, new_vectors = _resolve_doc(client, tenant_id, new_doc_ref)

    next_epoch_id = get_next_epoch_id(client, tenant_id)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    old_doc_ids = [doc.doc_id for doc in old_docs]

    for old_doc, old_vectors in zip(old_docs, old_vectors_list):
        old_copy = old_doc.model_copy(
            update={
                "superseded_by": new_doc.doc_id,
                "valid_to": now,
                "epoch_id": next_epoch_id,
            }
        )
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

    new_copy = new_doc.model_copy(
        update={
            "supersedes": old_doc_ids,
            "epoch_id": next_epoch_id,
        }
    )
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

    epoch_data = create_epoch(client, tenant_id, CONFIG.model_name)

    return SupersedeResult(
        old_doc_ids=old_doc_ids,
        new_doc_id=new_doc.doc_id,
        old_valid_to=now,
        epoch=epoch_data,
    )
