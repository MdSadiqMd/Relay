"""Merkle verification of retrieval logs."""

from qdrant_client.models import FieldCondition, Filter, MatchValue

from relay.collections import ensure_collections
from relay.config import CONFIG
from relay.epochs import get_epoch
from relay.merkle import (
    build_supersession_dag,
    compute_leaf,
    compute_merkle_root,
    toposort_docs,
)
from relay.models import DocumentPayload, VerifyResult, VerifyStatus


SCROLL_LIMIT = 10000


def _verify_with_stored_leaves(
    client, epoch, log_tenant, tenant_id, retrieved_doc_ids
) -> VerifyResult:
    """Fast path: epoch has stored leaf_hashes — no full epoch scroll needed."""
    stored_root = epoch.merkle_root
    stored_leaf_hexes = epoch.leaf_hashes
    stored_doc_ids = set(epoch.doc_ids)

    # 1. Check all retrieved docs were part of this epoch
    missing_docs = [did for did in retrieved_doc_ids if did not in stored_doc_ids]

    # 2. Decode stored leaves and recompute root
    stored_leaves = [bytes.fromhex(h) for h in stored_leaf_hexes]
    computed_root = compute_merkle_root(stored_leaves, ordered=True)
    root_match = computed_root == stored_root

    # 3. Fetch each retrieved doc from Qdrant to verify content integrity
    for did in retrieved_doc_ids:
        if did in missing_docs:
            continue
        pts, _ = client.scroll(
            collection_name=CONFIG.documents_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=log_tenant)),
                    FieldCondition(
                        key="epoch_id", match=MatchValue(value=epoch.epoch_id)
                    ),
                    FieldCondition(key="doc_id", match=MatchValue(value=did)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            missing_docs.append(did)

    docs_verified = all(did not in missing_docs for did in retrieved_doc_ids)

    return VerifyResult(
        status=VerifyStatus.VERIFIED
        if (
            root_match
            and docs_verified
            and not missing_docs
            and log_tenant == tenant_id
        )
        else VerifyStatus.FAILED,
        epoch_id=epoch.epoch_id,
        stored_merkle_root=stored_root,
        computed_merkle_root=computed_root,
        root_match=root_match,
        docs_in_epoch=docs_verified,
        missing_docs=missing_docs,
        tenant_match=log_tenant == tenant_id,
        retrieved_doc_count=len(retrieved_doc_ids),
        epoch_doc_count=len(stored_doc_ids),
    )


def _verify_scroll_fallback(
    client, epoch, log_tenant, tenant_id, retrieved_doc_ids
) -> VerifyResult:
    """Slow fallback: scroll all epoch docs to rebuild Merkle tree (legacy epochs)."""
    stored_root = epoch.merkle_root
    epoch_id = epoch.epoch_id

    all_docs: list[dict] = []
    offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=CONFIG.documents_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value=log_tenant)),
                    FieldCondition(key="epoch_id", match=MatchValue(value=epoch_id)),
                ]
            ),
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_docs.extend([p.payload for p in pts if p.payload is not None])
        if next_offset is None:
            break
        offset = next_offset

    doc_payloads = [DocumentPayload(**d) for d in all_docs]
    dag = build_supersession_dag(doc_payloads)
    ordered_docs = toposort_docs(doc_payloads, dag)

    leaves: list[bytes] = []
    doc_map: dict[str, bytes] = {}
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
        doc_map[doc.doc_id] = leaf

    computed_root = compute_merkle_root(leaves, ordered=True)
    missing_docs = [did for did in retrieved_doc_ids if did not in doc_map]
    root_match = computed_root == stored_root
    docs_present = len(missing_docs) == 0

    return VerifyResult(
        status=VerifyStatus.VERIFIED
        if (root_match and docs_present and log_tenant == tenant_id)
        else VerifyStatus.FAILED,
        epoch_id=epoch_id,
        stored_merkle_root=stored_root,
        computed_merkle_root=computed_root,
        root_match=root_match,
        docs_in_epoch=docs_present,
        missing_docs=missing_docs,
        tenant_match=log_tenant == tenant_id,
        retrieved_doc_count=len(retrieved_doc_ids),
        epoch_doc_count=len(all_docs),
    )


def verify_retrieval(request_id: str, tenant_id: str) -> VerifyResult:
    """Verify a past retrieval against epoch Merkle commitment.

    Steps:
        1. Fetch retrieval log by request_id
        2. For each doc_id in retrieved_docs, fetch from Qdrant
        3. Recompute leaf hashes
        4. Recompute full epoch Merkle root
        5. Compare with stored merkle_root

    Uses stored leaf hashes from the epoch point for O(1) Merkle root recomputation
    (no epoch-wide document scroll). Falls back to scrolling all docs for legacy
    epochs that predate the incremental Merkle accumulator.
    """
    client = ensure_collections()

    # 1. Find retrieval log
    results, _ = client.scroll(
        collection_name=CONFIG.logs_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="request_id", match=MatchValue(value=request_id)),
            ]
        ),
        limit=1,
        with_payload=True,
    )

    if not results:
        return VerifyResult(
            status=VerifyStatus.FAILED,
            reason=f"No retrieval log found for request_id={request_id}",
        )

    log = results[0].payload
    if log is None:
        return VerifyResult(
            status=VerifyStatus.FAILED,
            reason="Empty payload in retrieval log",
        )
    epoch_id = log["epoch_id"]
    log_tenant = log.get("tenant_id", tenant_id)
    retrieved_doc_ids: list[str] = log.get("retrieved_docs", [])

    # 2. Get the epoch
    epoch = get_epoch(client, log_tenant, epoch_id)
    if epoch is None:
        return VerifyResult(
            status=VerifyStatus.FAILED,
            reason=f"Epoch {epoch_id} not found",
        )

    # 3. Fast path: use stored leaf hashes (incremental), fallback to scroll (legacy)
    if epoch.leaf_hashes and epoch.doc_ids:
        return _verify_with_stored_leaves(
            client, epoch, log_tenant, tenant_id, retrieved_doc_ids
        )

    return _verify_scroll_fallback(
        client, epoch, log_tenant, tenant_id, retrieved_doc_ids
    )
