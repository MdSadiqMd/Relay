"""Unit tests for Pydantic models — no external deps required."""

import pytest
from pydantic import ValidationError

from relay.models import (
    DiffResult,
    DiffSummary,
    DocChange,
    DocSummary,
    DocumentPayload,
    DriftLevel,
    EpochPayload,
    IngestResult,
    QueryResult,
    QueryResultItem,
    RetrievalLogPayload,
    RetrievalPolicy,
    SupersedeResult,
    SupersessionInfo,
    VerifyResult,
    VerifyStatus,
)


class TestEnums:
    def test_retrieval_policy_values(self):
        assert RetrievalPolicy.DENSE == "dense"
        assert RetrievalPolicy.HYBRID == "hybrid"
        assert RetrievalPolicy.NAMED_VECTORS == "named_vectors"

    def test_verify_status_values(self):
        assert VerifyStatus.VERIFIED == "VERIFIED"
        assert VerifyStatus.FAILED == "FAILED"

    def test_drift_level_ordering(self):
        levels = [
            DriftLevel.NONE,
            DriftLevel.LOW,
            DriftLevel.MEDIUM,
            DriftLevel.HIGH,
            DriftLevel.STRUCTURAL,
        ]
        assert len(levels) == 5


class TestDocumentPayload:
    def test_create_minimal(self):
        doc = DocumentPayload(
            doc_id="test_001",
            tenant_id="default",
            content_hash="abc123",
            embedding_hash="def456",
            model_version="all-MiniLM-L6-v2",
            valid_from="2024-01-01",
            epoch_id=1,
            created_at="2024-01-01T00:00:00Z",
            source_file="test.md",
        )
        assert doc.doc_id == "test_001"
        assert doc.valid_to is None
        assert doc.supersedes == []
        assert doc.superseded_by is None
        assert doc.semantic_tags == []

    def test_roundtrip_serialization(self):
        doc = DocumentPayload(
            doc_id="test_001",
            tenant_id="default",
            content_hash="abc123",
            embedding_hash="def456",
            model_version="all-MiniLM-L6-v2",
            valid_from="2024-01-01",
            valid_to="2025-01-01",
            epoch_id=1,
            supersedes=["old_doc"],
            created_at="2024-01-01T00:00:00Z",
            semantic_tags=["kafka", "streaming"],
            source_file="test.md",
        )
        data = doc.model_dump()
        restored = DocumentPayload(**data)
        assert restored == doc

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DocumentPayload(
                doc_id="test_001",
                # missing tenant_id and others
            )


class TestEpochPayload:
    def test_create_with_parent(self):
        epoch = EpochPayload(
            epoch_id=2,
            tenant_id="default",
            created_at="2024-06-01T00:00:00Z",
            merkle_root="aabbccdd",
            doc_count=5,
            model_version="all-MiniLM-L6-v2",
            parent_epoch=1,
        )
        assert epoch.parent_epoch == 1

    def test_create_without_parent(self):
        epoch = EpochPayload(
            epoch_id=1,
            tenant_id="default",
            created_at="2024-01-01T00:00:00Z",
            merkle_root="aabbccdd",
            doc_count=3,
            model_version="all-MiniLM-L6-v2",
        )
        assert epoch.parent_epoch is None

    def test_json_serialization(self):
        epoch = EpochPayload(
            epoch_id=1,
            tenant_id="t",
            created_at="now",
            merkle_root="root",
            doc_count=0,
            model_version="v1",
        )
        json_str = epoch.model_dump_json()
        assert '"epoch_id":1' in json_str


class TestIngestResult:
    def test_fields(self):
        result = IngestResult(
            doc_id="doc_abc",
            epoch_id=1,
            content_hash="ch",
            embedding_hash="eh",
            merkle_root="mr",
            source_file="kafka.md",
        )
        assert result.source_file == "kafka.md"


class TestSupersedeResult:
    def test_nested_epoch(self):
        epoch = EpochPayload(
            epoch_id=2,
            tenant_id="default",
            created_at="now",
            merkle_root="root",
            doc_count=2,
            model_version="v1",
        )
        result = SupersedeResult(
            old_doc_ids=["old"],
            new_doc_id="new",
            old_valid_to="2025-01-01",
            epoch=epoch,
        )
        assert result.epoch.epoch_id == 2


class TestQueryResult:
    def test_empty_results(self):
        qr = QueryResult(
            query_text="hello",
            epoch_id=1,
            request_id="req-1",
            result_count=0,
        )
        assert qr.results == []
        assert qr.at is None

    def test_with_results(self):
        item = QueryResultItem(
            doc_id="d1",
            score=0.95,
            source_file="test.md",
            valid_from="2024-01-01",
        )
        qr = QueryResult(
            query_text="hello",
            epoch_id=1,
            request_id="req-1",
            result_count=1,
            results=[item],
        )
        assert qr.results[0].score == 0.95


class TestDiffResult:
    def test_structural_drift(self):
        result = DiffResult(
            epoch_from=1,
            epoch_to=2,
            added=[DocSummary(doc_id="new_doc")],
            semantic_drift=DriftLevel.STRUCTURAL,
            summary=DiffSummary(
                added_count=1,
                removed_count=0,
                changed_count=0,
                superseded_count=0,
            ),
        )
        assert result.semantic_drift == DriftLevel.STRUCTURAL
        assert result.summary.added_count == 1

    def test_doc_change_model(self):
        change = DocChange(
            doc_id="d1",
            old_content_hash="aaa",
            new_content_hash="bbb",
            content_changed=True,
            embedding_changed=False,
            cosine_similarity=0.98,
        )
        assert change.content_changed is True
        assert change.cosine_similarity == 0.98

    def test_supersession_info(self):
        sup = SupersessionInfo(
            doc_id="new_doc",
            supersedes=["old_doc"],
            source_file="nats.md",
        )
        assert sup.supersedes == ["old_doc"]


class TestRetrievalLogPayload:
    def test_defaults(self):
        log = RetrievalLogPayload(
            query_hash="qh",
            query_text_preview="hello world",
            epoch_id=1,
            tenant_id="default",
            timestamp="now",
            request_id="req-1",
        )
        assert log.retrieved_docs == []
        assert log.at_timestamp is None


class TestVerifyResult:
    def test_verified(self):
        result = VerifyResult(
            status=VerifyStatus.VERIFIED,
            epoch_id=1,
            stored_merkle_root="aaa",
            computed_merkle_root="aaa",
            root_match=True,
            docs_in_epoch=True,
            tenant_match=True,
            retrieved_doc_count=2,
            epoch_doc_count=2,
        )
        assert result.status == VerifyStatus.VERIFIED

    def test_failed_with_reason(self):
        result = VerifyResult(
            status=VerifyStatus.FAILED,
            reason="Epoch not found",
        )
        assert result.reason == "Epoch not found"
        assert result.missing_docs == []
