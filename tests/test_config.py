"""Unit tests for relay config — pydantic-settings validation."""

from relay.config import RelayConfig


class TestRelayConfig:
    def test_defaults(self):
        cfg = RelayConfig()
        assert cfg.qdrant_host == "localhost"
        assert cfg.qdrant_port == 6333
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.semantic_dim == 384
        assert cfg.default_tenant == "default"
        assert cfg.default_top_k == 5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RELAY_QDRANT_HOST", "qdrant-server")
        monkeypatch.setenv("RELAY_QDRANT_PORT", "9999")
        monkeypatch.setenv("RELAY_DEFAULT_TENANT", "acme")
        cfg = RelayConfig()
        assert cfg.qdrant_host == "qdrant-server"
        assert cfg.qdrant_port == 9999
        assert cfg.default_tenant == "acme"

    def test_collection_names(self):
        cfg = RelayConfig()
        assert cfg.documents_collection == "relay_documents"
        assert cfg.epochs_collection == "relay_epochs"
        assert cfg.logs_collection == "relay_retrieval_logs"
