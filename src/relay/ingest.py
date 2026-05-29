"""Ingest pipeline — read file, hash, embed, upsert, epoch management."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client.models import PointStruct, SparseVector

from relay.collections import (
    collection_has_sparse,
    collection_has_video,
    ensure_collections,
)
from relay.config import CONFIG
from relay.embeddings import (
    content_hash,
    embed,
    embedding_hash,
    sparse_embed,
    video_embedding_hash as compute_video_embedding_hash,
)
from relay.epochs import create_epoch, get_next_epoch_id
from relay.merkle import compute_leaf
from relay.models import DocumentPayload, IngestResult


def ingest_text(
    text: str,
    tenant_id: str,
    valid_from: str,
    valid_to: Optional[str] = None,
    supersedes: Optional[list[str]] = None,
    semantic_tags: Optional[list[str]] = None,
    source_file: str = "inline.txt",
    doc_id: Optional[str] = None,
    video_vector: Optional[list[float]] = None,
) -> IngestResult:
    """Ingest raw text into relay.

    Steps:
        1. Compute content_hash = SHA256(text)
        2. Embed text → semantic vector
        3. Compute embedding_hash = SHA256(embedding)
        4. (Optional) Accept pre-computed ``video_vector`` for multimodal
        5. Generate or use provided doc_id
        6. Upsert to relay_documents with full payload
        7. Create new immutable epoch with Merkle root

    Args:
        video_vector: Pre-computed 1024-d TwelveLabs video embedding.
            When provided, stored as a "video" named vector in Qdrant and
            its hash is written to ``DocumentPayload.video_embedding_hash``.
    Returns:
        IngestResult with doc_id, epoch_id, hashes, merkle_root
    """
    client = ensure_collections()

    # 1. Content hash
    c_hash = content_hash(text)

    # 2. Embed (dense always; sparse only when collection supports it)
    vector = embed(text)
    has_sparse = collection_has_sparse(client, CONFIG.documents_collection)
    has_video = collection_has_video(client, CONFIG.documents_collection)
    sparse_indices, sparse_values = sparse_embed(text) if has_sparse else ([], [])

    # 3. Embedding hash
    e_hash = embedding_hash(vector)
    v_hash = compute_video_embedding_hash(video_vector) if video_vector else None

    # 4. Generate doc_id if not provided
    if doc_id is None:
        path_stem = Path(source_file).stem
        doc_id = f"{path_stem}_{uuid.uuid4().hex[:8]}"

    # 5. Determine epoch — each ingest creates a new immutable epoch
    epoch_id = get_next_epoch_id(client, tenant_id)

    now = datetime.now(timezone.utc).isoformat()

    doc = DocumentPayload(
        doc_id=doc_id,
        tenant_id=tenant_id,
        content_hash=c_hash,
        embedding_hash=e_hash,
        video_embedding_hash=v_hash,
        model_version=CONFIG.model_name,
        valid_from=valid_from,
        valid_to=valid_to,
        epoch_id=epoch_id,
        supersedes=supersedes or [],
        superseded_by=None,
        created_at=now,
        semantic_tags=semantic_tags or [],
        source_file=source_file,
    )

    # 6. Upsert document
    point_id = str(uuid.uuid4())
    vectors: dict = {"semantic": vector}
    if has_sparse:
        vectors["sparse"] = SparseVector(indices=sparse_indices, values=sparse_values)
    if video_vector is not None and has_video:
        vectors["video"] = video_vector

    client.upsert(
        collection_name=CONFIG.documents_collection,
        points=[
            PointStruct(
                id=point_id,
                vector=vectors,
                payload=doc.model_dump(),
            )
        ],
    )

    # 7. Compute leaf hash and create new immutable epoch
    leaf = compute_leaf(
        doc_id=doc.doc_id,
        content_hash=doc.content_hash,
        embedding_hash=doc.embedding_hash,
        model_version=doc.model_version,
        valid_from=doc.valid_from,
        valid_to=doc.valid_to,
        supersedes=doc.supersedes,
        video_embedding_hash=doc.video_embedding_hash,
    )
    epoch_data = create_epoch(
        client,
        tenant_id,
        CONFIG.model_name,
        leaf_hashes=[leaf],
        doc_ids=[doc.doc_id],
        epoch_id=epoch_id,
    )
    merkle_root = epoch_data.merkle_root
    final_epoch_id = epoch_data.epoch_id

    return IngestResult(
        doc_id=doc_id,
        epoch_id=final_epoch_id,
        content_hash=c_hash,
        embedding_hash=e_hash,
        merkle_root=merkle_root,
        source_file=source_file,
    )


def ingest_file(
    file_path: str,
    tenant_id: str,
    valid_from: str,
    valid_to: Optional[str] = None,
    supersedes: Optional[list[str]] = None,
    semantic_tags: Optional[list[str]] = None,
) -> IngestResult:
    """Ingest a file into relay.

    Reads file content then delegates to :func:`ingest_text`.

    Returns:
        IngestResult with doc_id, epoch_id, hashes, merkle_root
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = path.read_text(encoding="utf-8")

    return ingest_text(
        text=text,
        tenant_id=tenant_id,
        valid_from=valid_from,
        valid_to=valid_to,
        supersedes=supersedes,
        semantic_tags=semantic_tags,
        source_file=path.name,
    )
