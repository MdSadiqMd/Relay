"""Shared test fixtures for the relay test suite.

Provides a conftest that:
  - Resolves `PRIV_DIR` (src/priv) so tests reference fixture files portably.
  - Provides a `qdrant_client` fixture that verifies Qdrant is reachable
    before any integration test runs.
"""

import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

PRIV_DIR = Path(__file__).resolve().parent.parent / "src" / "priv"


@pytest.fixture(scope="session")
def priv_dir() -> Path:
    """Path to the src/priv directory containing fixture markdown files."""
    assert PRIV_DIR.exists(), f"src/priv not found at {PRIV_DIR}"
    return PRIV_DIR


@pytest.fixture(scope="session")
def qdrant_client() -> QdrantClient:
    """Return a QdrantClient connected to the local instance.

    Skips the test if Qdrant is not reachable.
    """
    host = os.environ.get("RELAY_QDRANT_HOST", "localhost")
    port = int(os.environ.get("RELAY_QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=5)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant is not reachable — skipping integration test")
    return client
