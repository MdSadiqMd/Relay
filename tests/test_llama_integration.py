"""Integration tests for pkg.llamaindex — requires a running Qdrant.

These tests verify RelayRetriever, LocalHFLLM, and create_query_engine against
a real Qdrant instance.  They are skipped if Qdrant is not reachable.

Uses a dedicated test tenant to avoid polluting production data.
"""

import uuid
from pathlib import Path

import pytest

from pkg.llamaindex import RelayRetriever

TEST_TENANT = f"test_llama_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client(qdrant_client):
    from relay.collections import ensure_collections

    return ensure_collections()


@pytest.fixture(scope="module")
def priv(priv_dir) -> Path:
    return priv_dir


class TestIngestForLlama:
    """Seed test documents needed by the LlamaIndex test classes."""

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
        assert result.epoch_id == 1
        self.__class__._kafka_doc_id = result.doc_id

    def test_ingest_nats(self):
        from relay.ingest import ingest_file

        result = ingest_file(
            file_path=str(self._priv / "nats_migration.md"),
            tenant_id=TEST_TENANT,
            valid_from="2025-01-01",
            semantic_tags=["nats", "migration"],
        )
        assert result.source_file == "nats_migration.md"
        self.__class__._nats_doc_id = result.doc_id


class TestRelayRetriever:
    """Tests for RelayRetriever — reads data ingested by TestIngestForLlama."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, client):
        self.__class__._client = client
        self.__class__._retriever = RelayRetriever(
            tenant_id=TEST_TENANT,
            top_k=5,
            text_resolver=lambda did, sf: f"resolved text for {did}",
        )

    def test_class_name(self):
        assert self.__class__._retriever.class_name() == "RelayRetriever"

    def test_retrieve_returns_nodes(self):
        nodes = self.__class__._retriever.retrieve("event bus architecture")
        assert len(nodes) > 0
        for nws in nodes:
            assert nws.node.id_ is not None
            assert isinstance(nws.score, float)
            assert nws.node.metadata["doc_id"] == nws.node.id_
            assert "epoch_id" in nws.node.metadata
            assert "request_id" in nws.node.metadata

    def test_text_resolver_populates_text(self):
        nodes = self.__class__._retriever.retrieve("message broker")
        assert len(nodes) > 0
        for nws in nodes:
            did = nws.node.id_
            expected_text = f"resolved text for {did}"
            assert nws.node.text == expected_text

    def test_retrieve_empty_text_when_no_resolver(self):
        retriever = RelayRetriever(
            tenant_id=TEST_TENANT,
            top_k=3,
        )
        nodes = retriever.retrieve("event bus architecture")
        assert len(nodes) > 0
        for nws in nodes:
            assert nws.node.text == ""

    def test_retrieve_with_epoch_returns_epoch_id(self):
        nodes, epoch_id = self.__class__._retriever.retrieve_with_epoch(
            "event bus architecture"
        )
        assert len(nodes) > 0
        assert isinstance(epoch_id, int)
        assert epoch_id >= 1

    def test_retrieve_dense_policy(self):
        retriever = RelayRetriever(
            tenant_id=TEST_TENANT,
            top_k=3,
            retrieval_policy="dense",
            text_resolver=lambda did, sf: f"resolved text for {did}",
        )
        nodes = retriever.retrieve("event bus architecture")
        assert len(nodes) > 0
        for nws in nodes:
            assert nws.node.id_ is not None

    def test_epoch_pinned_retrieval(self):
        retriever = RelayRetriever(
            tenant_id=TEST_TENANT,
            top_k=3,
            epoch_id=1,
            text_resolver=lambda did, sf: f"resolved text for {did}",
        )
        nodes = retriever.retrieve("event bus architecture")
        assert len(nodes) > 0


class TestRelayRetrieverMetadata:
    """Test that the metadata attached to returned nodes is correct."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, client):
        self.__class__._retriever = RelayRetriever(
            tenant_id=TEST_TENANT,
            top_k=1,
            text_resolver=lambda did, sf: f"resolved text for {did}",
        )

    def test_metadata_has_all_fields(self):
        nodes = self.__class__._retriever.retrieve("event bus architecture")
        assert len(nodes) >= 1
        meta = nodes[0].node.metadata
        assert "doc_id" in meta
        assert "score" in meta
        assert "source_file" in meta
        assert "content_hash" in meta
        assert "valid_from" in meta
        assert "valid_to" in meta
        assert "supersedes" in meta
        assert "superseded_by" in meta
        assert "semantic_tags" in meta
        assert "epoch_id" in meta
        assert "request_id" in meta

    def test_metadata_values_are_consistent(self):
        nodes = self.__class__._retriever.retrieve("event bus architecture")
        assert len(nodes) >= 1
        nws = nodes[0]
        meta = nws.node.metadata
        assert meta["doc_id"] == nws.node.id_
        assert meta["score"] == nws.score
        assert meta["epoch_id"] >= 1
        assert len(meta["request_id"]) > 0


class TestLocalHFLLM:
    """Unit-level smoke tests for LocalHFLLM — does NOT load a real model.

    Loading a HuggingFace model (~270 MB) on every test run is too expensive.
    These tests verify the class contract without instantiation.
    """

    def test_class_name_default(self):
        from pkg.llamaindex import LocalHFLLM

        assert LocalHFLLM.class_name() == "LocalHFLLM"

    def test_metadata_property_defaults(self):
        from pkg.llamaindex import DEFAULT_LLM_MODEL

        assert DEFAULT_LLM_MODEL == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class TestCreateQueryEngine:
    """Unit-level smoke tests for create_query_engine — does NOT load a model."""

    def test_relay_import(self):
        from pkg.llamaindex import create_query_engine

        assert callable(create_query_engine)

    def test_relay_retriever_import(self):
        from pkg.llamaindex import RelayRetriever

        assert issubclass(RelayRetriever.__class__, type)
