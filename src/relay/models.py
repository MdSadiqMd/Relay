"""Pydantic models for all relay domain objects.

Every data structure flowing through relay is typed here — no more raw dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetrievalPolicy(str, Enum):
    """Supported retrieval policies."""

    DENSE = "dense"
    HYBRID = "hybrid"
    NAMED_VECTORS = "named_vectors"


class VerifyStatus(str, Enum):
    """Verification result status."""

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class DriftLevel(str, Enum):
    """Semantic drift severity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRUCTURAL = "structural"


class DocumentPayload(BaseModel):
    """A document stored in relay_documents."""

    doc_id: str
    tenant_id: str
    content_hash: str
    embedding_hash: str
    model_version: str
    valid_from: str
    valid_to: Optional[str] = None
    epoch_id: int
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    created_at: str
    semantic_tags: list[str] = Field(default_factory=list)
    source_file: str


class EpochPayload(BaseModel):
    """An immutable semantic epoch stored in relay_epochs."""

    epoch_id: int
    tenant_id: str
    created_at: str
    merkle_root: str
    doc_count: int
    model_version: str
    parent_epoch: Optional[int] = None
    leaf_hashes: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)


class RetrievalLogPayload(BaseModel):
    """A retrieval log entry stored in relay_retrieval_logs."""

    query_hash: str
    query_text_preview: str
    epoch_id: int
    tenant_id: str
    retrieved_docs: list[str] = Field(default_factory=list)
    retrieval_policy: str = RetrievalPolicy.DENSE
    timestamp: str
    request_id: str
    at_timestamp: Optional[str] = None


class IngestResult(BaseModel):
    """Result returned from the ingest pipeline."""

    doc_id: str
    epoch_id: int
    content_hash: str
    embedding_hash: str
    merkle_root: str
    source_file: str


class SupersedeResult(BaseModel):
    """Result returned from supersession."""

    old_doc_ids: list[str]
    new_doc_id: str
    old_valid_to: str
    epoch: EpochPayload


class QueryResultItem(BaseModel):
    """A single document result from a query."""

    doc_id: str
    score: float
    source_file: Optional[str] = None
    content_hash: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    semantic_tags: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Full result from a temporal semantic query."""

    query_text: str
    epoch_id: int
    at: Optional[str] = None
    retrieval_policy: str = RetrievalPolicy.DENSE
    request_id: str
    result_count: int
    results: list[QueryResultItem] = Field(default_factory=list)


class DocSummary(BaseModel):
    """Lightweight doc summary used in diff reports."""

    doc_id: str
    source_file: Optional[str] = None
    content_hash: Optional[str] = None


class DocChange(BaseModel):
    """A changed document between epochs."""

    doc_id: str
    old_content_hash: str
    new_content_hash: str
    content_changed: bool
    embedding_changed: bool
    cosine_similarity: Optional[float] = None


class SupersessionInfo(BaseModel):
    """A supersession relationship in a diff."""

    doc_id: str
    supersedes: list[str]
    source_file: Optional[str] = None


class DiffSummary(BaseModel):
    """Aggregate counts for a diff."""

    added_count: int
    removed_count: int
    changed_count: int
    superseded_count: int


class DiffResult(BaseModel):
    """Full result from an epoch diff."""

    epoch_from: int
    epoch_to: int
    added: list[DocSummary] = Field(default_factory=list)
    removed: list[DocSummary] = Field(default_factory=list)
    changed: list[DocChange] = Field(default_factory=list)
    superseded: list[SupersessionInfo] = Field(default_factory=list)
    semantic_drift: DriftLevel = DriftLevel.NONE
    summary: DiffSummary


class VerifyResult(BaseModel):
    """Result from Merkle verification of a retrieval."""

    status: VerifyStatus
    epoch_id: Optional[int] = None
    stored_merkle_root: Optional[str] = None
    computed_merkle_root: Optional[str] = None
    root_match: Optional[bool] = None
    docs_in_epoch: Optional[bool] = None
    missing_docs: list[str] = Field(default_factory=list)
    tenant_match: Optional[bool] = None
    query_preview: str = ""
    retrieval_policy: str = ""
    retrieved_doc_count: int = 0
    epoch_doc_count: int = 0
    reason: Optional[str] = None
