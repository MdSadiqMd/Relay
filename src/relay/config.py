"""Configuration and settings for relay"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class RelayConfig(BaseSettings):
    """Relay configuration — loaded from env vars with RELAY_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RELAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embedding models
    model_name: str = "all-MiniLM-L6-v2"
    semantic_dim: int = 384
    # Qdrant/bm25 — fast (10MB, no ONNX). For SPLADE: prithivida/Splade_PP_en_v1
    sparse_model_name: str = "Qdrant/bm25"

    # Collection names
    documents_collection: str = "relay_documents"
    epochs_collection: str = "relay_epochs"
    logs_collection: str = "relay_retrieval_logs"

    # Defaults
    default_tenant: str = "default"
    default_retrieval_policy: str = "dense"
    default_top_k: int = 5

    # LLM provider — "local" (HF), "openai", or "anthropic"
    llm_provider: str = "local"
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"


CONFIG = RelayConfig()
