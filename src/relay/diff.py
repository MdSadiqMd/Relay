"""Semantic diff between epochs."""

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue

from relay.collections import ensure_collections
from relay.config import CONFIG
from relay.models import (
    DiffResult,
    DiffSummary,
    DriftLevel,
    DocChange,
    DocSummary,
    SupersessionInfo,
)


def _scroll_epoch_docs(client, tenant_id: str, epoch_id: int) -> list[dict]:
    """Get all documents for a specific epoch (with vectors for drift analysis)."""
    all_docs: list[dict] = []
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
            with_vectors=True,
        )
        for point in results:
            doc = dict(point.payload)
            vectors = point.vector
            if isinstance(vectors, dict) and "semantic" in vectors:
                doc["_vector"] = vectors["semantic"]
            all_docs.append(doc)

        if next_offset is None:
            break
        offset = next_offset
    return all_docs


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(v1), np.array(v2)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def diff_epochs(
    from_epoch: int,
    to_epoch: int,
    tenant_id: str,
) -> DiffResult:
    """Compare semantic states between two epochs.

    Returns:
        DiffResult with added, removed, changed, superseded docs, and drift level
    """
    client = ensure_collections()

    from_docs = _scroll_epoch_docs(client, tenant_id, from_epoch)
    to_docs = _scroll_epoch_docs(client, tenant_id, to_epoch)

    from_by_id = {d["doc_id"]: d for d in from_docs}
    to_by_id = {d["doc_id"]: d for d in to_docs}

    from_ids = set(from_by_id.keys())
    to_ids = set(to_by_id.keys())

    added = [
        DocSummary(
            doc_id=to_by_id[did]["doc_id"],
            source_file=to_by_id[did].get("source_file"),
            content_hash=to_by_id[did].get("content_hash"),
        )
        for did in (to_ids - from_ids)
    ]

    removed = [
        DocSummary(
            doc_id=from_by_id[did]["doc_id"],
            source_file=from_by_id[did].get("source_file"),
            content_hash=from_by_id[did].get("content_hash"),
        )
        for did in (from_ids - to_ids)
    ]

    common_ids = from_ids & to_ids
    changed: list[DocChange] = []
    drift_scores: list[float] = []

    for did in common_ids:
        old_doc = from_by_id[did]
        new_doc = to_by_id[did]

        if (
            old_doc["content_hash"] != new_doc["content_hash"]
            or old_doc["embedding_hash"] != new_doc["embedding_hash"]
        ):
            cos_sim = None
            if "_vector" in old_doc and "_vector" in new_doc:
                cos_sim = round(
                    _cosine_similarity(old_doc["_vector"], new_doc["_vector"]), 4
                )
                drift_scores.append(1.0 - cos_sim)

            changed.append(
                DocChange(
                    doc_id=did,
                    old_content_hash=old_doc["content_hash"],
                    new_content_hash=new_doc["content_hash"],
                    content_changed=old_doc["content_hash"] != new_doc["content_hash"],
                    embedding_changed=old_doc["embedding_hash"]
                    != new_doc["embedding_hash"],
                    cosine_similarity=cos_sim,
                )
            )

    # Superseded documents
    superseded = [
        SupersessionInfo(
            doc_id=doc["doc_id"],
            supersedes=doc["supersedes"]
            if isinstance(doc["supersedes"], list)
            else [doc["supersedes"]],
            source_file=doc.get("source_file"),
        )
        for doc in to_docs
        if doc.get("supersedes")
    ]

    # Overall drift level
    if drift_scores:
        avg_drift = sum(drift_scores) / len(drift_scores)
        if avg_drift > 0.3:
            drift_level = DriftLevel.HIGH
        elif avg_drift > 0.1:
            drift_level = DriftLevel.MEDIUM
        else:
            drift_level = DriftLevel.LOW
    else:
        drift_level = (
            DriftLevel.NONE
            if not (added or removed or changed)
            else DriftLevel.STRUCTURAL
        )

    return DiffResult(
        epoch_from=from_epoch,
        epoch_to=to_epoch,
        added=added,
        removed=removed,
        changed=changed,
        superseded=superseded,
        semantic_drift=drift_level,
        summary=DiffSummary(
            added_count=len(added),
            removed_count=len(removed),
            changed_count=len(changed),
            superseded_count=len(superseded),
        ),
    )
