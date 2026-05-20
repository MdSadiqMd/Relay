"""Ingest pipeline — read file, hash, embed, upsert, epoch management."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client.models import PointStruct

from relay.collections import ensure_collections
from relay.config import CONFIG
from relay.embeddings import content_hash, embed, embedding_hash
from relay.epochs import create_epoch, get_current_epoch_id, refresh_epoch_merkle
from relay.models import DocumentPayload, IngestResult


def ingest_file(
    file_path: str,
    tenant_id: str,
    valid_from: str,
    valid_to: Optional[str] = None,
    supersedes: Optional[list[str]] = None,
    semantic_tags: Optional[list[str]] = None,
) -> IngestResult:
    """Ingest a file into relay.

    Steps:
        1. Read file content
        2. Compute content_hash = SHA256(text)
        3. Embed text → vector
        4. Compute embedding_hash = SHA256(embedding)
        5. Generate doc_id
        6. Upsert to relay_documents with full payload
        7. Create/update epoch with recomputed Merkle root

    Returns:
        IngestResult with doc_id, epoch_id, hashes, merkle_root
    """
    client = ensure_collections()

    # 1. Read file
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = path.read_text(encoding="utf-8")

    # 2. Content hash
    c_hash = content_hash(text)

    # 3. Embed
    vector = embed(text)

    # 4. Embedding hash
    e_hash = embedding_hash(vector)

    # 5. Generate doc_id (use filename stem + short uuid for uniqueness)
    doc_id = f"{path.stem}_{uuid.uuid4().hex[:8]}"

    # 6. Determine epoch
    current_epoch = get_current_epoch_id(client, tenant_id)
    if current_epoch is None:
        epoch_id = 1
    else:
        epoch_id = current_epoch

    now = datetime.now(timezone.utc).isoformat()

    doc = DocumentPayload(
        doc_id=doc_id,
        tenant_id=tenant_id,
        content_hash=c_hash,
        embedding_hash=e_hash,
        model_version=CONFIG.model_name,
        valid_from=valid_from,
        valid_to=valid_to,
        epoch_id=epoch_id,
        supersedes=supersedes or [],
        superseded_by=None,
        created_at=now,
        semantic_tags=semantic_tags or [],
        source_file=path.name,
    )

    # 7. Upsert document
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=CONFIG.documents_collection,
        points=[
            PointStruct(
                id=point_id,
                vector={"semantic": vector},
                payload=doc.model_dump(),
            )
        ],
    )

    # 8. Create or refresh epoch
    if current_epoch is None:
        epoch_data = create_epoch(client, tenant_id, CONFIG.model_name)
        merkle_root = epoch_data.merkle_root
        final_epoch_id = epoch_data.epoch_id
    else:
        merkle_root = refresh_epoch_merkle(client, tenant_id, epoch_id)
        final_epoch_id = epoch_id

    return IngestResult(
        doc_id=doc_id,
        epoch_id=final_epoch_id,
        content_hash=c_hash,
        embedding_hash=e_hash,
        merkle_root=merkle_root,
        source_file=path.name,
    )
