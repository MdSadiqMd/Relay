"""Temporal retrieval engine — time-travel queries with epoch pinning.

Provides both raw ``QueryResult`` output (``query()``) and LlamaIndex-native
``NodeWithScore`` output (``query_nodes()``) — the latter weaves LlamaIndex
directly into relay's core retrieval layer.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
)

from relay.collections import (
    collection_has_sparse,
    collection_has_video,
    ensure_collections,
)
from relay.config import CONFIG
from relay.embeddings import embed, sparse_embed
from relay.epochs import get_current_epoch_id, get_epoch, resolve_epoch_at
from relay.models import (
    QueryResult,
    QueryResultItem,
    RetrievalLogPayload,
    RetrievalPolicy,
)


def query(
    text: str,
    tenant_id: str,
    at: Optional[str] = None,
    epoch_id: Optional[int] = None,
    retrieval_policy: str = "dense",
    top_k: int = 5,
) -> QueryResult:
    """Execute a temporal semantic query.

    Modes:
        - Default: uses latest epoch
        - --at TIMESTAMP: resolves epoch at that time
        - --epoch N: pins to exact epoch
    """
    client = ensure_collections()

    # Resolve epoch
    if epoch_id is not None:
        target_epoch = get_epoch(client, tenant_id, epoch_id)
        if target_epoch is None:
            raise ValueError(f"Epoch {epoch_id} not found for tenant {tenant_id}")
        resolved_epoch_id = epoch_id
    elif at is not None:
        target_epoch = resolve_epoch_at(client, tenant_id, at)
        if target_epoch is None:
            raise ValueError(f"No epoch found at or before {at}")
        resolved_epoch_id = target_epoch.epoch_id
    else:
        current_epoch = get_current_epoch_id(client, tenant_id)
        if current_epoch is None:
            raise ValueError(f"No epochs found for tenant {tenant_id}")
        resolved_epoch_id = current_epoch

    # Build filter — epoch + tenant at Qdrant level; temporal validity post-filtered
    must_conditions = [
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="epoch_id", match=MatchValue(value=resolved_epoch_id)),
    ]
    qdrant_filter = Filter(must=must_conditions)  # type: ignore[arg-type]

    try:
        policy = RetrievalPolicy(retrieval_policy.lower())
    except ValueError:
        policy = RetrievalPolicy.DENSE

    if policy == RetrievalPolicy.MULTIMODAL:
        if not collection_has_video(client, CONFIG.documents_collection):
            raise ValueError(
                "Multimodal retrieval requires a collection with a 'video' "
                "named vector. Recreate the collection so documents are "
                "ingested with both semantic and video vectors."
            )
        from pkg.twelvelabs.embed import compute_text_query_embedding

        dense_vector = embed(text)
        video_query_vector = compute_text_query_embedding(text)
        results = client.query_points(
            collection_name=CONFIG.documents_collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="semantic",
                    filter=qdrant_filter,
                    limit=top_k * 3,
                ),
                Prefetch(
                    query=video_query_vector,
                    using="video",
                    filter=qdrant_filter,
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k * 3,
            with_payload=True,
        )
    elif policy == RetrievalPolicy.HYBRID:
        if not collection_has_sparse(client, CONFIG.documents_collection):
            raise ValueError(
                "Hybrid retrieval requires a collection with sparse vectors. "
                "Recreate the collection (e.g. `just nuke && just demo`) so documents "
                "are ingested with both dense and sparse vectors."
            )
        dense_vector = embed(text)
        sparse_indices, sparse_values = sparse_embed(text)
        results = client.query_points(
            collection_name=CONFIG.documents_collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="semantic",
                    filter=qdrant_filter,
                    limit=top_k * 3,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    filter=qdrant_filter,
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k * 3,
            with_payload=True,
        )
    else:
        dense_vector = embed(text)
        results = client.query_points(
            collection_name=CONFIG.documents_collection,
            query=dense_vector,
            using="semantic",
            query_filter=qdrant_filter,
            limit=top_k * 3,
            with_payload=True,
        )

    # Post-filter: temporal validity + supersession
    filtered: list[QueryResultItem] = []
    for pt in results.points:
        p = pt.payload
        if p is None:
            continue

        if at is not None:
            # Skip docs not yet valid at query time
            valid_from = p.get("valid_from")
            if valid_from and valid_from > at:
                continue
            # Skip docs that expired before query time
            valid_to = p.get("valid_to")
            if valid_to and valid_to <= at:
                continue
            # Skip superseded docs
            if p.get("superseded_by"):
                continue

        filtered.append(
            QueryResultItem(
                doc_id=p.get("doc_id", ""),
                score=pt.score,
                source_file=p.get("source_file"),
                content_hash=p.get("content_hash"),
                valid_from=p.get("valid_from"),
                valid_to=p.get("valid_to"),
                supersedes=p.get("supersedes"),
                superseded_by=p.get("superseded_by"),
                semantic_tags=p.get("semantic_tags", []),
            )
        )
        # Stop once we have top_k valid results. Qdrant returns in score
        # order (RRF for hybrid, cosine for dense) so the first top_k valid
        # entries are the highest-scoring. If many results are filtered out
        # (e.g., superseded) the user gets fewer than top_k — acceptable.
        if len(filtered) >= top_k:
            break

    # Log retrieval
    query_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    log = RetrievalLogPayload(
        query_hash=query_hash,
        query_text_preview=text[:200],
        epoch_id=resolved_epoch_id,
        tenant_id=tenant_id,
        retrieved_docs=[r.doc_id for r in filtered],
        retrieval_policy=retrieval_policy,
        timestamp=now,
        request_id=log_id,
        at_timestamp=at,
    )

    client.upsert(
        collection_name=CONFIG.logs_collection,
        points=[
            PointStruct(
                id=log_id,
                vector=[0.0],
                payload=log.model_dump(),
            )
        ],
    )

    return QueryResult(
        query_text=text,
        epoch_id=resolved_epoch_id,
        at=at,
        retrieval_policy=retrieval_policy,
        request_id=log_id,
        result_count=len(filtered),
        results=filtered,
    )


def query_nodes(
    text: str,
    tenant_id: str,
    at: Optional[str] = None,
    epoch_id: Optional[int] = None,
    retrieval_policy: str = "dense",
    top_k: int = 5,
    text_resolver: Optional[Callable[[str, Optional[str]], str]] = None,
) -> tuple[list[NodeWithScore], int]:
    """Execute a temporal semantic query and return results as LlamaIndex nodes.

    Same retrieval logic as :func:`query` but returns ``NodeWithScore`` objects
    with resolved document text via the optional ``text_resolver`` callback.
    This weaves LlamaIndex directly into relay's core retrieval layer.

    Returns:
        ``(nodes, epoch_id)`` — the nodes and the resolved epoch.
    """
    result = query(
        text=text,
        tenant_id=tenant_id,
        at=at,
        epoch_id=epoch_id,
        retrieval_policy=retrieval_policy,
        top_k=top_k,
    )
    nodes: list[NodeWithScore] = []
    for item in result.results:
        text_content = ""
        if text_resolver is not None:
            text_content = text_resolver(item.doc_id, item.source_file)
        node = TextNode(
            id_=item.doc_id,
            text=text_content,
            metadata={
                "doc_id": item.doc_id,
                "score": item.score,
                "source_file": item.source_file,
                "content_hash": item.content_hash,
                "valid_from": item.valid_from,
                "valid_to": item.valid_to,
                "supersedes": item.supersedes,
                "superseded_by": item.superseded_by,
                "semantic_tags": item.semantic_tags,
                "epoch_id": result.epoch_id,
                "request_id": result.request_id,
            },
        )
        nodes.append(NodeWithScore(node=node, score=item.score))
    return nodes, result.epoch_id
