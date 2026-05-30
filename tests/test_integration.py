"""Integration tests — requires a running Qdrant instance.

These tests exercise the full ingest → query → supersede → diff → verify
pipeline against a real Qdrant. They are automatically skipped if Qdrant
is not reachable (via the `qdrant_client` fixture in conftest.py).

Uses a dedicated test tenant to avoid polluting production data.
"""

import uuid
from pathlib import Path

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from relay.collections import (
    _has_sparse_cache,
    collection_has_sparse,
    ensure_collections,
    invalidate_has_sparse_cache,
)
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

    def test_epoch_has_leaf_hashes(self, client):
        from relay.epochs import get_epoch

        epoch = get_epoch(client, TEST_TENANT, 1)
        assert epoch is not None
        assert len(epoch.leaf_hashes) == 1
        assert len(epoch.doc_ids) == 1
        assert epoch.doc_ids[0] == self.__class__._kafka_doc_id
        assert len(epoch.leaf_hashes[0]) == 64  # hex-encoded SHA256


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
        # Retrieve the log entry by point ID — more reliable than scroll
        # (Qdrant scroll has eventual-consistency gaps for recent writes).
        points = get_client().retrieve(
            collection_name=cfg.logs_collection,
            ids=[result.request_id],
            with_payload=True,
        )
        assert len(points) == 1
        payload = points[0].payload
        assert payload is not None
        assert payload["retrieval_policy"] == "hybrid"

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

    def test_collection_has_sparse_cache_hit(self, client):
        _has_sparse_cache.clear()
        assert CONFIG.documents_collection not in _has_sparse_cache
        result = collection_has_sparse(client, CONFIG.documents_collection)
        assert result is True
        assert _has_sparse_cache[CONFIG.documents_collection] is True

    def test_collection_has_sparse_cache_negative(self, client):
        tmp = f"relay_test_cache_neg_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=tmp,
            vectors_config={
                "semantic": VectorParams(size=384, distance=Distance.COSINE)
            },
        )
        try:
            assert tmp not in _has_sparse_cache
            result = collection_has_sparse(client, tmp)
            assert result is False
            assert _has_sparse_cache[tmp] is False
        finally:
            client.delete_collection(tmp)
            invalidate_has_sparse_cache(tmp)

    def test_invalidate_has_sparse_cache_clears_entry(self, client):
        _has_sparse_cache.clear()
        collection_has_sparse(client, CONFIG.documents_collection)
        assert CONFIG.documents_collection in _has_sparse_cache
        invalidate_has_sparse_cache(CONFIG.documents_collection)
        assert CONFIG.documents_collection not in _has_sparse_cache

    def test_invalid_policy_falls_back_to_dense(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            retrieval_policy="nonsense_policy",
            top_k=3,
        )
        assert result.result_count > 0


class TestVideoCollection:
    """collection_has_video() and related cache tests."""

    def test_collection_has_video_false_for_semantic_only(self, client):
        """Existing relay_documents (no video vector) returns False."""
        from relay.collections import collection_has_video

        assert collection_has_video(client, CONFIG.documents_collection) is False

    def test_collection_has_video_true(self, client):
        """A collection created with a 'video' named vector returns True."""
        from relay.collections import collection_has_video, invalidate_has_video_cache

        tmp = f"relay_test_video_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=tmp,
            vectors_config={
                "semantic": VectorParams(size=384, distance=Distance.COSINE),
                "video": VectorParams(size=1024, distance=Distance.COSINE),
            },
        )
        try:
            invalidate_has_video_cache(tmp)
            assert collection_has_video(client, tmp) is True
        finally:
            client.delete_collection(tmp)

    def test_collection_has_video_cache_hit(self, client):
        from relay.collections import _has_video_cache, collection_has_video

        _has_video_cache.clear()
        assert CONFIG.documents_collection not in _has_video_cache
        result = collection_has_video(client, CONFIG.documents_collection)
        assert result is False  # existing test collection has no video vector
        assert CONFIG.documents_collection in _has_video_cache

    def test_multimodal_raises_on_no_video_collection(self, client, monkeypatch):
        """Multimodal policy raises on a collection without video vector."""
        from relay.query import query as do_query
        from relay.collections import invalidate_has_video_cache

        tmp = f"relay_test_no_video_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=tmp,
            vectors_config={
                "semantic": VectorParams(size=384, distance=Distance.COSINE)
            },
        )
        monkeypatch.setattr(CONFIG, "documents_collection", tmp)
        invalidate_has_video_cache(tmp)
        try:
            with pytest.raises(ValueError, match="Multimodal retrieval requires"):
                do_query(
                    text="test", tenant_id=TEST_TENANT, retrieval_policy="multimodal"
                )
        finally:
            client.delete_collection(tmp)

    def test_invalidate_has_video_cache_clears_entry(self, client):
        from relay.collections import (
            _has_video_cache,
            collection_has_video,
            invalidate_has_video_cache,
        )

        _has_video_cache.clear()
        collection_has_video(client, CONFIG.documents_collection)
        assert CONFIG.documents_collection in _has_video_cache
        invalidate_has_video_cache(CONFIG.documents_collection)
        assert CONFIG.documents_collection not in _has_video_cache


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

    def test_supersede_epoch_has_leaf_hashes(self, client):
        from relay.epochs import get_epoch
        from relay.collections import get_client
        from relay.config import CONFIG as cfg

        epoch = get_epoch(client, TEST_TENANT, 3)
        assert epoch is not None
        assert len(epoch.leaf_hashes) == 2
        assert len(epoch.doc_ids) == 2
        for h in epoch.leaf_hashes:
            assert len(h) == 64
        # Verify all doc_ids exist in the epoch's documents
        for did in epoch.doc_ids:
            pts, _ = get_client().scroll(
                collection_name=cfg.documents_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=did)),
                        FieldCondition(key="epoch_id", match=MatchValue(value=3)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            assert len(pts) == 1, f"doc_id {did} not found in epoch 3"


class TestTimeTravel:
    """Phase 4: Historical queries with --at."""

    def test_historical_query(self, client):
        from relay.query import query

        result = query(
            text="event bus architecture",
            tenant_id=TEST_TENANT,
            at="2025-06-01",
        )
        assert isinstance(result, QueryResult)
        assert result.result_count > 0
        # Should find docs valid at 2025-06-01 and not superseded
        for item in result.results:
            assert item.valid_from <= "2025-06-01"

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
        assert len(epoch.leaf_hashes) == epoch.doc_count
        assert len(epoch.doc_ids) == epoch.doc_count
        for h in epoch.leaf_hashes:
            assert len(h) == 64

    def test_get_nonexistent_epoch(self, client):
        from relay.epochs import get_epoch

        epoch = get_epoch(client, TEST_TENANT, 999)
        assert epoch is None

    def test_get_next_epoch_id(self, client):
        from relay.epochs import get_next_epoch_id

        nid = get_next_epoch_id(client, TEST_TENANT)
        assert isinstance(nid, int)
        assert nid > 0

    def test_get_next_epoch_id_no_epochs(self, client):
        from relay.epochs import get_next_epoch_id

        nid = get_next_epoch_id(client, "nonexistent_tenant")
        assert nid == 1

    def test_get_current_epoch_id(self, client):
        from relay.epochs import get_current_epoch_id

        cid = get_current_epoch_id(client, TEST_TENANT)
        assert isinstance(cid, int)
        assert cid >= 2

    def test_get_current_epoch_id_nonexistent(self, client):
        from relay.epochs import get_current_epoch_id

        cid = get_current_epoch_id(client, "nonexistent_tenant")
        assert cid is None
