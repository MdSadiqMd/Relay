"""Tests for the embeddings module"""

from relay.embeddings import content_hash, embedding_hash
from relay.config import CONFIG


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("hello") != content_hash("world")

    def test_returns_hex(self):
        h = content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)

    def test_unicode(self):
        h = content_hash("日本語テスト 🚀")
        assert len(h) == 64


class TestVideoEmbeddingHash:
    def test_deterministic(self):
        from relay.embeddings import video_embedding_hash

        vec = [0.1] * 1024
        assert video_embedding_hash(vec) == video_embedding_hash(vec)

    def test_different_vectors(self):
        from relay.embeddings import video_embedding_hash

        assert video_embedding_hash([0.1] * 1024) != video_embedding_hash([0.2] * 1024)

    def test_order_matters(self):
        from relay.embeddings import video_embedding_hash

        v1 = [0.1, 0.2] + [0.0] * 1022
        v2 = [0.2, 0.1] + [0.0] * 1022
        assert video_embedding_hash(v1) != video_embedding_hash(v2)

    def test_returns_hex(self):
        from relay.embeddings import video_embedding_hash

        h = video_embedding_hash([0.5] * 1024)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)

    def test_same_as_embedding_hash_for_same_input(self):
        from relay.embeddings import embedding_hash, video_embedding_hash

        vec = [0.3, 0.7, 0.1]
        assert video_embedding_hash(vec) == embedding_hash(vec)


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
        int(h, 16)


class TestDenseEmbed:
    """Tests for the sentence-transformers dense embedding function."""

    def test_returns_list_of_floats(self):
        from relay.embeddings import embed

        vec = embed("event streaming kafka")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_correct_dimension(self):
        from relay.embeddings import embed

        vec = embed("hello world")
        assert len(vec) == CONFIG.semantic_dim  # 384

    def test_l2_normalised(self):
        import math
        from relay.embeddings import embed

        vec = embed("normalisation check")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self):
        from relay.embeddings import embed

        v1 = embed("reproducible output")
        v2 = embed("reproducible output")
        assert v1 == v2

    def test_different_texts_differ(self):
        from relay.embeddings import embed

        v1 = embed("kafka event streaming broker")
        v2 = embed("authentication credentials password")
        assert v1 != v2

    def test_embedding_hash_pipeline(self):
        """embed() → embedding_hash() produces a stable 64-char hex string."""
        from relay.embeddings import embed

        vec = embed("pipeline integrity check")
        h = embedding_hash(vec)
        assert isinstance(h, str)
        assert len(h) == 64
        assert embedding_hash(vec) == h  # deterministic


class TestSparseEmbed:
    """Tests for the BM25 sparse embedding function."""

    def test_returns_lists(self):
        from relay.embeddings import sparse_embed

        indices, values = sparse_embed("event streaming kafka")
        assert isinstance(indices, list)
        assert isinstance(values, list)

    def test_non_empty_for_content(self):
        from relay.embeddings import sparse_embed

        indices, values = sparse_embed("event streaming kafka")
        assert len(indices) > 0
        assert len(values) > 0

    def test_indices_values_same_length(self):
        from relay.embeddings import sparse_embed

        indices, values = sparse_embed("message broker architecture")
        assert len(indices) == len(values)

    def test_deterministic(self):
        from relay.embeddings import sparse_embed

        i1, v1 = sparse_embed("hello world")
        i2, v2 = sparse_embed("hello world")
        assert i1 == i2
        assert v1 == v2

    def test_different_texts_differ(self):
        from relay.embeddings import sparse_embed

        i1, _ = sparse_embed("kafka event streaming broker")
        i2, _ = sparse_embed("authentication credentials password")
        assert set(i1) != set(i2)

    def test_element_types(self):
        from relay.embeddings import sparse_embed

        indices, values = sparse_embed("test text")
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)

    def test_empty_string_handled(self):
        from relay.embeddings import sparse_embed

        indices, values = sparse_embed("")
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)


class TestEmbedCache:
    """`@lru_cache` on embed() / sparse_embed() — identity & hit counting."""

    def test_embed_returns_same_object_for_same_text(self):
        from relay.embeddings import embed

        v1 = embed("cache identity test")
        v2 = embed("cache identity test")
        assert v1 is v2

    def test_embed_cache_hits(self):
        from relay.embeddings import embed

        embed.cache_clear()
        assert embed.cache_info().hits == 0
        embed("hit me")
        embed("hit me")
        assert embed.cache_info().hits == 1

    def test_sparse_embed_returns_same_objects_for_same_text(self):
        from relay.embeddings import sparse_embed

        i1, v1 = sparse_embed("cache identity test")
        i2, v2 = sparse_embed("cache identity test")
        assert i1 is i2
        assert v1 is v2

    def test_sparse_embed_cache_hits(self):
        from relay.embeddings import sparse_embed

        sparse_embed.cache_clear()
        assert sparse_embed.cache_info().hits == 0
        sparse_embed("hit me")
        sparse_embed("hit me")
        assert sparse_embed.cache_info().hits == 1
