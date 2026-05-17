"""Unit tests for the embeddings module — hashing functions only.

The actual embedding model is NOT loaded in these tests (too slow for unit).
We only test the deterministic hashing helpers.
"""

from relay.embeddings import content_hash, embedding_hash


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_hex(self):
        h = content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest
        int(h, 16)  # valid hex

    def test_unicode(self):
        h = content_hash("日本語テスト 🚀")
        assert len(h) == 64


class TestEmbeddingHash:
    def test_deterministic(self):
        vec = [0.1, 0.2, 0.3]
        assert embedding_hash(vec) == embedding_hash(vec)

    def test_different_vectors(self):
        assert embedding_hash([0.1, 0.2]) != embedding_hash([0.3, 0.4])

    def test_order_matters(self):
        assert embedding_hash([0.1, 0.2]) != embedding_hash([0.2, 0.1])

    def test_empty_vector(self):
        h = embedding_hash([])
        assert isinstance(h, str)
        assert len(h) == 64

    def test_returns_hex(self):
        h = embedding_hash([1.0, 2.0, 3.0])
        int(h, 16)  # valid hex
