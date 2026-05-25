"""Integration tests — requires a running Qdrant instance.

These tests exercise the full ingest → query → supersede → diff → verify
pipeline against a real Qdrant. They are automatically skipped if Qdrant
is not reachable (via the `qdrant_client` fixture in conftest.py).

Uses a dedicated test tenant to avoid polluting production data.
"""

import uuid
from pathlib import Path

import pytest
from qdrant_client.models import Distance, VectorParams

from relay.collections import collection_has_sparse, ensure_collections
from relay.config import CONFIG
from relay.models import (
    IngestResult,
    QueryResult,
    SupersedeResult,
    VerifyStatus,
)

TEST_TENANT = f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client(qdrant_client):
    """Ensure collections exist and return the client."""
    return ensure_collections()


@pytest.fixture(scope="module")
def priv(priv_dir) -> Path:
    return priv_dir


class TestIngestPipeline:
    """Phase 1: Ingest documents."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, client, priv):
        self.__class__._client = client
        self.__class__._priv = priv

    def test_ingest_kafka(self):
        from relay.ingest import ingest_file

        result = ingest_file(
            file_path=str(self._priv / "kafka.md"),
            tenant_id=TEST_TENANT,
            valid_from="2024-01-01",
            semantic_tags=["kafka", "streaming"],
        )
        assert isinstance(result, IngestResult)
        assert result.epoch_id == 1
        assert result.source_file == "kafka.md"
        assert len(result.content_hash) == 64
        assert len(result.merkle_root) == 64
        self.__class__._kafka_doc_id = result.doc_id

    def test_ingest_nats(self):
        from relay.ingest import ingest_file

        result = ingest_file(
            file_path=str(self._priv / "nats_migration.md"),
            tenant_id=TEST_TENANT,
            valid_from="2025-01-01",
            semantic_tags=["nats", "migration"],
        )
        assert isinstance(result, IngestResult)
        assert result.source_file == "nats_migration.md"
        self.__class__._nats_doc_id = result.doc_id


class TestQueryPipeline:
    """Phase 2: Query the ingested documents."""

    def test_query_latest(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            top_k=5,
        )
        assert isinstance(result, QueryResult)
        assert result.result_count > 0
        assert result.epoch_id >= 1
        assert len(result.request_id) > 0
        self.__class__._request_id = result.request_id

    def test_query_returns_results(self, client):
        from relay.query import query

        result = query(
            text="message broker",
            tenant_id=TEST_TENANT,
            top_k=2,
        )
        assert result.result_count <= 2
        for item in result.results:
            assert item.doc_id
            assert isinstance(item.score, float)


class TestHybridRetrieval:
    """Hybrid (dense + sparse / RRF) retrieval tests — runs after ingest."""

    def test_collection_has_sparse(self, client):
        assert collection_has_sparse(client, CONFIG.documents_collection) is True

    def test_hybrid_query_returns_results(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            retrieval_policy="hybrid",
            top_k=5,
        )
        assert isinstance(result, QueryResult)
        assert result.result_count > 0
        assert result.retrieval_policy == "hybrid"

    def test_hybrid_rrf_scores_normalized(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            retrieval_policy="hybrid",
            top_k=5,
        )
        for item in result.results:
            assert 0.0 <= item.score <= 1.0

    def test_hybrid_logs_retrieval_policy(self, client):
        from relay.query import query
        from relay.collections import get_client
        from relay.config import CONFIG as cfg

        result = query(
            text="message broker",
            tenant_id=TEST_TENANT,
            retrieval_policy="hybrid",
            top_k=3,
        )
        # Log entry should record the policy as "hybrid"
        logs = get_client().scroll(
            collection_name=cfg.logs_collection,
            scroll_filter=None,
            limit=100,
            with_payload=True,
        )[0]
        matching = [
            p.payload
            for p in logs
            if p.payload and p.payload.get("request_id") == result.request_id
        ]
        assert len(matching) == 1
        assert matching[0]["retrieval_policy"] == "hybrid"

    def test_hybrid_raises_on_dense_only_collection(self, client, monkeypatch):
        from relay.query import query as do_query

        tmp = f"relay_test_no_sparse_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=tmp,
            vectors_config={
                "semantic": VectorParams(size=384, distance=Distance.COSINE)
            },
        )
        assert collection_has_sparse(client, tmp) is False
        monkeypatch.setattr(CONFIG, "documents_collection", tmp)
        try:
            with pytest.raises(ValueError, match="sparse vectors"):
                do_query(text="test", tenant_id=TEST_TENANT, retrieval_policy="hybrid")
        finally:
            client.delete_collection(tmp)

    def test_collection_has_sparse_false_for_dense_only(self, client):
        tmp = f"relay_test_dense_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=tmp,
            vectors_config={
                "semantic": VectorParams(size=384, distance=Distance.COSINE)
            },
        )
        try:
            assert collection_has_sparse(client, tmp) is False
        finally:
            client.delete_collection(tmp)

    def test_invalid_policy_falls_back_to_dense(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            retrieval_policy="nonsense_policy",
            top_k=3,
        )
        assert result.result_count > 0


class TestSupersedePipeline:
    """Phase 3: Supersede kafka → nats."""

    def test_supersede(self, client):
        from relay.supersede import supersede

        result = supersede(
            old_doc_refs=["kafka.md"],
            new_doc_ref="nats_migration.md",
            tenant_id=TEST_TENANT,
        )
        assert isinstance(result, SupersedeResult)
        assert result.epoch.epoch_id == 3
        assert result.old_valid_to  # should be set
        assert result.old_doc_ids  # list of superseded doc ids
        self.__class__._supersede_epoch = result.epoch.epoch_id


class TestTimeTravel:
    """Phase 4: Historical queries with --at."""

    def test_historical_query(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            at="2024-06-01",
        )
        assert isinstance(result, QueryResult)
        # Should find docs valid at 2024-06-01 and not superseded
        for item in result.results:
            if item.valid_from:
                assert item.valid_from <= "2024-06-01"

    def test_epoch_pinned_query(self, client):
        from relay.query import query

        result = query(
            text="event bus",
            tenant_id=TEST_TENANT,
            epoch_id=1,
        )
        assert result.epoch_id == 1


class TestDiffPipeline:
    """Phase 5: Diff between epochs."""

    def test_diff_epochs(self, client):
        from relay.diff import diff_epochs

        result = diff_epochs(
            from_epoch=1,
            to_epoch=3,
            tenant_id=TEST_TENANT,
        )
        assert result.epoch_from == 1
        assert result.epoch_to == 3
        assert result.summary.superseded_count >= 1
        # Supersession info should reference the correct docs
        for sup in result.superseded:
            assert sup.supersedes  # old doc id


class TestVerifyPipeline:
    """Phase 6: Verify retrieval integrity."""

    def test_verify_latest_retrieval(self, client):
        from relay.query import query
        from relay.verify import verify_retrieval

        # First, do a query to get a request_id
        qr = query(
            text="event bus",
            tenant_id=TEST_TENANT,
        )

        result = verify_retrieval(
            request_id=qr.request_id,
            tenant_id=TEST_TENANT,
        )
        assert result.status == VerifyStatus.VERIFIED
        assert result.root_match is True
        assert result.tenant_match is True

    def test_verify_nonexistent_request(self, client):
        from relay.verify import verify_retrieval

        result = verify_retrieval(
            request_id="nonexistent-id",
            tenant_id=TEST_TENANT,
        )
        assert result.status == VerifyStatus.FAILED
        assert result.reason is not None


class TestEpochManagement:
    """Epoch listing and inspection."""

    def test_list_epochs(self, client):
        from relay.epochs import list_epochs

        epochs = list_epochs(client, TEST_TENANT)
        assert len(epochs) >= 2
        assert epochs[0].epoch_id == 1
        assert epochs[1].epoch_id == 2

    def test_get_epoch(self, client):
        from relay.epochs import get_epoch

        epoch = get_epoch(client, TEST_TENANT, 1)
        assert epoch is not None
        assert epoch.doc_count >= 1
        assert len(epoch.merkle_root) == 64

    def test_get_nonexistent_epoch(self, client):
        from relay.epochs import get_epoch

        epoch = get_epoch(client, TEST_TENANT, 999)
        assert epoch is None
