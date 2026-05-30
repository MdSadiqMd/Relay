"""Tests for pkg.twelvelabs — TwelveLabs video analysis integration.

Two tiers:
  - Unit tests: mock the Twelvelabs SDK entirely, no Qdrant required.
  - Integration tests: mock the Twelvelabs SDK but exercise real relay
    ingestion into a running Qdrant instance.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from relay.config import CONFIG
from relay.models import IngestResult


@pytest.fixture(autouse=True)
def _patch_config():
    """Ensure a known API key is set and singleton is reset for every test."""
    from pkg.twelvelabs.client import reset_client

    original = CONFIG.twelve_labs_api_key
    CONFIG.twelve_labs_api_key = "test_key_12345"
    reset_client()
    yield
    CONFIG.twelve_labs_api_key = original
    reset_client()


@pytest.fixture
def mock_twelvelabs():
    """Mock the entire twelvelabs SDK so no real API call is made.

    Patches ``twelvelabs.TwelveLabs`` at import time and resets the
    client singleton so every test gets a fresh mock.
    """
    from pkg.twelvelabs.client import reset_client

    reset_client()

    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.data = (
        "This video discusses the architecture of the NATS messaging system. "
        "Key points include: async communication, JetStream persistence, "
        "and clustering for high availability."
    )
    mock_client.analyze.return_value = mock_result

    with patch("twelvelabs.TwelveLabs", return_value=mock_client) as mock_cls:
        yield mock_cls


class TestClientConfig:
    """Config resolution and error handling."""

    def test_get_api_key_success(self):
        from pkg.twelvelabs.client import get_api_key

        key = get_api_key()
        assert key == "test_key_12345"

    def test_get_api_key_missing(self):
        from pkg.twelvelabs.client import get_api_key, TwelveLabsError

        original = CONFIG.twelve_labs_api_key
        CONFIG.twelve_labs_api_key = ""
        try:
            with pytest.raises(TwelveLabsError, match="API key is required"):
                get_api_key()
        finally:
            CONFIG.twelve_labs_api_key = original

    def test_get_api_key_from_env(self, monkeypatch):
        original = CONFIG.twelve_labs_api_key
        CONFIG.twelve_labs_api_key = ""
        from relay.config import CONFIG as refreshed

        refreshed.twelve_labs_api_key = "env_key_value"
        try:
            from pkg.twelvelabs.client import get_api_key as gak

            assert gak() == "env_key_value"
        finally:
            refreshed.twelve_labs_api_key = original


class TestAnalyzeVideo:
    """analyze_video() — the stateless convenience function."""

    def test_analyze_video_success(self, mock_twelvelabs):
        from pkg.twelvelabs.client import analyze_video

        result = analyze_video(
            video_url="https://example.com/demo.mp4",
            prompt="Summarize this video",
        )

        assert isinstance(result, str)
        assert "NATS" in result
        assert "JetStream" in result
        mock_twelvelabs.assert_called_once()

    def test_analyze_video_default_prompt(self, mock_twelvelabs):
        from pkg.twelvelabs.client import analyze_video

        result = analyze_video(video_url="https://example.com/demo.mp4")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should have used the default prompt
        call_kwargs = mock_twelvelabs.return_value.analyze.call_args[1]
        assert "technical summary" in call_kwargs["prompt"]

    def test_analyze_video_api_error(self):
        from pkg.twelvelabs.client import TwelveLabsError, analyze_video, reset_client

        reset_client()
        mock_client = MagicMock()
        mock_client.analyze.side_effect = ValueError("API failure")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(TwelveLabsError, match="API failure"):
                analyze_video(video_url="https://example.com/bad.mp4")

    def test_analyze_video_missing_key(self):
        from pkg.twelvelabs.client import analyze_video, TwelveLabsError

        original = CONFIG.twelve_labs_api_key
        CONFIG.twelve_labs_api_key = ""
        try:
            with pytest.raises(TwelveLabsError, match="API key is required"):
                analyze_video(video_url="https://example.com/v.mp4")
        finally:
            CONFIG.twelve_labs_api_key = original


class TestTwelveLabsClient:
    """TwelveLabsClient — the reusable class wrapper."""

    def test_analyze_success(self, mock_twelvelabs):
        from pkg.twelvelabs.client import TwelveLabsClient

        client = TwelveLabsClient(api_key="custom_key")
        result = client.analyze(
            video_url="https://example.com/demo.mp4",
            prompt="Summarize",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify custom key was used
        mock_twelvelabs.assert_called_once_with(api_key="custom_key")

    def test_analyze_without_prompt_uses_default(self, mock_twelvelabs):
        from pkg.twelvelabs.client import TwelveLabsClient

        client = TwelveLabsClient(api_key="key")
        client.analyze(video_url="https://example.com/v.mp4")

        call_kwargs = mock_twelvelabs.return_value.analyze.call_args[1]
        assert "technical summary" in call_kwargs["prompt"]

    def test_analyze_error_wraps_exception(self):
        from pkg.twelvelabs.client import TwelveLabsClient, TwelveLabsError

        mock_client = MagicMock()
        mock_client.analyze.side_effect = RuntimeError("network error")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            client = TwelveLabsClient(api_key="key")
            with pytest.raises(TwelveLabsError, match="network error"):
                client.analyze(video_url="https://example.com/v.mp4")


class TestIngestVideoUrl:
    """ingest_video_url() — the full video→relay pipeline."""

    def test_ingest_video_url_flow(self, mock_twelvelabs, monkeypatch):
        """Verify the pipeline: analyze → ingest_text."""
        from pkg.twelvelabs.ingest_video import ingest_video_url

        fake_result = IngestResult(
            doc_id="video_demo_abc123",
            epoch_id=1,
            content_hash="a" * 64,
            embedding_hash="b" * 64,
            merkle_root="c" * 64,
            source_file="video_demo.md",
        )

        # Mock relay.ingest.ingest_text to avoid Qdrant dependency
        with patch("pkg.twelvelabs.ingest_video.ingest_text", return_value=fake_result):
            result = ingest_video_url(
                video_url="https://example.com/demo.mp4",
                tenant_id="test_video",
                valid_from="2026-01-01",
                prompt="Summarize this",
            )

        assert isinstance(result, IngestResult)
        assert result.doc_id == "video_demo_abc123"
        assert result.epoch_id == 1
        assert result.source_file == "video_demo.md"
        mock_twelvelabs.assert_called_once()

    def test_ingest_video_url_tags(self, mock_twelvelabs, monkeypatch):
        """Verify semantic tags are passed through correctly."""
        from pkg.twelvelabs.ingest_video import ingest_video_url

        fake_result = IngestResult(
            doc_id="vid_x",
            epoch_id=2,
            content_hash="a" * 64,
            embedding_hash="b" * 64,
            merkle_root="c" * 64,
            source_file="video_demo.md",
        )

        captured_kwargs = {}

        def fake_ingest_text(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_result

        with patch(
            "pkg.twelvelabs.ingest_video.ingest_text", side_effect=fake_ingest_text
        ):
            result = ingest_video_url(
                video_url="https://example.com/demo.mp4",
                tenant_id="test_video",
                valid_from="2026-01-01",
                semantic_tags=["tutorial", "architecture"],
            )

        # Should include "video" tag plus the custom tags
        assert "video" in captured_kwargs["semantic_tags"]
        assert "tutorial" in captured_kwargs["semantic_tags"]
        assert "architecture" in captured_kwargs["semantic_tags"]
        assert result.epoch_id == 2

    def test_ingest_video_url_error_handling(self):
        """If analyze_video fails, the error propagates."""
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.ingest_video import ingest_video_url

        reset_client()
        mock_client = MagicMock()
        mock_client.analyze.side_effect = ValueError("API failure")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(TwelveLabsError, match="API failure"):
                ingest_video_url(
                    video_url="https://example.com/bad.mp4",
                    tenant_id="test_video",
                    valid_from="2026-01-01",
                )

    def test_url_to_name_extracts_stem(self):
        from pkg.twelvelabs.ingest_video import _url_to_name

        name = _url_to_name("https://cdn.example.com/videos/demo_talk.mp4")
        assert "demo_talk" in name

        name = _url_to_name("https://example.com/video")
        assert "video" in name

    def test_url_to_name_handles_query_params(self):
        from pkg.twelvelabs.ingest_video import _url_to_name

        name = _url_to_name("https://example.com/videos/tech_overview.mp4?token=abc")
        assert "tech_overview" in name


class TestEmbed:
    """TwelveLabs Embed v2 API — compute_video_embedding and compute_text_query_embedding."""

    @pytest.fixture
    def mock_embed_sdk(self):
        """Mock the TwelveLabs SDK's Embed v2 API."""
        from pkg.twelvelabs.client import reset_client

        reset_client()
        mock_client = MagicMock()

        # Mock embed.v_2.tasks.create → task with id
        mock_task = MagicMock()
        mock_task.id = "task_abc123"
        mock_client.embed.v_2.tasks.create.return_value = mock_task

        # Mock embed.v_2.tasks.retrieve → ready with data
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1024
        mock_retrieve = MagicMock()
        mock_retrieve.status = "ready"
        mock_retrieve.data = [mock_data]
        mock_client.embed.v_2.tasks.retrieve.return_value = mock_retrieve

        # Mock embed.v_2.create → result with data
        mock_text_data = MagicMock()
        mock_text_data.embedding = [0.2] * 1024
        mock_embed_result = MagicMock()
        mock_embed_result.data = [mock_text_data]
        mock_client.embed.v_2.create.return_value = mock_embed_result

        with patch("twelvelabs.TwelveLabs", return_value=mock_client) as mock_cls:
            yield mock_cls, mock_client

    def test_compute_video_embedding_success(self, mock_embed_sdk):
        mock_cls, mock_client = mock_embed_sdk
        from pkg.twelvelabs.embed import compute_video_embedding

        vec = compute_video_embedding(
            video_url="https://example.com/video.mp4",
        )

        assert isinstance(vec, list)
        assert len(vec) == 1024
        assert vec[0] == 0.1
        mock_cls.assert_called_once()

    def test_compute_video_embedding_api_error(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_video_embedding

        reset_client()
        mock_client = MagicMock()
        mock_client.embed.v_2.tasks.create.side_effect = ValueError("API failure")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(TwelveLabsError):
                compute_video_embedding(video_url="https://example.com/bad.mp4")

    def test_compute_text_query_embedding_success(self, mock_embed_sdk):
        mock_cls, mock_client = mock_embed_sdk
        from pkg.twelvelabs.embed import compute_text_query_embedding

        vec = compute_text_query_embedding(text="test query")

        assert isinstance(vec, list)
        assert len(vec) == 1024
        assert vec[0] == 0.2
        mock_cls.assert_called_once()

    def test_compute_text_query_embedding_api_error(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_text_query_embedding

        reset_client()
        mock_client = MagicMock()
        mock_client.embed.v_2.create.side_effect = ValueError("API failure")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(TwelveLabsError):
                compute_text_query_embedding(text="test")

    def test_compute_video_embedding_no_data(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_video_embedding

        reset_client()
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "task_xyz"
        mock_client.embed.v_2.tasks.create.return_value = mock_task
        mock_retrieve = MagicMock()
        mock_retrieve.status = "ready"
        mock_retrieve.data = None
        mock_client.embed.v_2.tasks.retrieve.return_value = mock_retrieve

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(
                TwelveLabsError, match="Video embedding task ready but no data"
            ):
                compute_video_embedding(video_url="https://example.com/v.mp4")

    def test_compute_text_query_embedding_no_data(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_text_query_embedding

        reset_client()
        mock_client = MagicMock()
        mock_embed_result = MagicMock()
        mock_embed_result.data = []
        mock_client.embed.v_2.create.return_value = mock_embed_result

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(
                TwelveLabsError, match="Text query embedding returned no data"
            ):
                compute_text_query_embedding(text="test")

    def test_compute_video_embedding_missing_key(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_video_embedding

        reset_client()
        original = CONFIG.twelve_labs_api_key
        CONFIG.twelve_labs_api_key = ""
        try:
            with pytest.raises(TwelveLabsError, match="API key is required"):
                compute_video_embedding(video_url="https://example.com/v.mp4")
        finally:
            CONFIG.twelve_labs_api_key = original

    def test_compute_text_query_embedding_missing_key(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.embed import compute_text_query_embedding

        reset_client()
        original = CONFIG.twelve_labs_api_key
        CONFIG.twelve_labs_api_key = ""
        try:
            with pytest.raises(TwelveLabsError, match="API key is required"):
                compute_text_query_embedding(text="test")
        finally:
            CONFIG.twelve_labs_api_key = original


class TestSegments:
    """Pegasus video segmentation — ingest_video_segments."""

    @pytest.fixture
    def mock_segment_sdk(self):
        """Mock the TwelveLabs SDK for segment analysis.

        Async segmentation fails (to trigger fallback path) and sync
        analyze returns paragraph-based text.
        """
        from pkg.twelvelabs.client import reset_client

        reset_client()
        mock_client = MagicMock()

        # Async segmentation raises (triggers fallback)
        mock_client.analyze_async.tasks.create.side_effect = ValueError("no async")

        # Sync fallback
        mock_result = MagicMock()
        mock_result.data = (
            "First paragraph about the video content.\n\n"
            "Second paragraph covering a different topic.\n\n"
            "Third and final paragraph with conclusions."
        )
        mock_client.analyze.return_value = mock_result

        with patch("twelvelabs.TwelveLabs", return_value=mock_client) as mock_cls:
            yield mock_cls, mock_client

    def test_ingest_video_segments_mocked(self, mock_segment_sdk, monkeypatch):
        """Fallback: each paragraph becomes a relay document."""
        mock_cls, _ = mock_segment_sdk
        from pkg.twelvelabs.segments import ingest_video_segments

        fake_results = [
            IngestResult(
                doc_id=f"seg_seg_{i * 10}s_abc",
                epoch_id=i + 1,
                content_hash="a" * 64,
                embedding_hash="b" * 64,
                merkle_root="c" * 64,
                source_file=f"seg_video_{i * 10}s.md",
            )
            for i in range(3)
        ]

        ingest_calls = []

        def fake_ingest_text(**kwargs):
            ingest_calls.append(kwargs)
            return fake_results[len(ingest_calls) - 1]

        with patch("pkg.twelvelabs.segments.ingest_text", side_effect=fake_ingest_text):
            results = ingest_video_segments(
                video_url="https://example.com/video.mp4",
                tenant_id="test_seg",
                semantic_tags=["demo"],
            )

        assert len(results) == 3
        assert "video" in ingest_calls[0]["semantic_tags"]
        assert "segment" in ingest_calls[0]["semantic_tags"]
        assert "demo" in ingest_calls[0]["semantic_tags"]
        assert ingest_calls[0]["valid_from"] == "0s"
        assert ingest_calls[1]["valid_from"] == "10s"
        assert ingest_calls[2]["valid_from"] == "20s"

    def test_ingest_video_segments_empty_transcript(self, mock_segment_sdk):
        """Empty transcript returns an empty list."""
        from pkg.twelvelabs.segments import ingest_video_segments

        mock_cls, mock_client = mock_segment_sdk
        mock_result = MagicMock()
        mock_result.data = ""
        mock_client.analyze.return_value = mock_result

        results = ingest_video_segments(
            video_url="https://example.com/video.mp4",
            tenant_id="test_empty",
        )
        assert results == []

    def test_ingest_video_segments_api_error(self):
        from pkg.twelvelabs.client import TwelveLabsError, reset_client
        from pkg.twelvelabs.segments import ingest_video_segments

        reset_client()
        mock_client = MagicMock()
        mock_client.analyze_async.tasks.create.side_effect = ValueError("no async")
        mock_client.analyze.side_effect = ValueError("API failure")

        with patch("twelvelabs.TwelveLabs", return_value=mock_client):
            with pytest.raises(TwelveLabsError, match="Video segmentation failed"):
                ingest_video_segments(
                    video_url="https://example.com/bad.mp4",
                    tenant_id="test_err",
                )


class TestIngestText:
    """ingest_text() — the new core ingestion primitive."""

    def test_ingest_text_basic(self, monkeypatch):
        """ingest_text returns a valid IngestResult when Qdrant is available."""
        from relay.ingest import ingest_text

        # Only run if Qdrant is reachable
        pytest.importorskip("qdrant_client")
        try:
            from relay.collections import ensure_collections

            ensure_collections()
        except Exception:
            pytest.skip("Qdrant not reachable")

        tenant = f"test_text_{uuid.uuid4().hex[:4]}"
        result = ingest_text(
            text="NATS is a cloud-native messaging system. "
            "It provides at-most-once and at-least-once delivery. "
            "JetStream adds persistence and stream processing.",
            tenant_id=tenant,
            valid_from="2025-01-01",
            semantic_tags=["nats", "messaging"],
            source_file="nats_summary.md",
        )

        assert isinstance(result, IngestResult)
        assert result.epoch_id == 1
        assert result.source_file == "nats_summary.md"
        assert len(result.content_hash) == 64
        assert len(result.merkle_root) == 64

    def test_ingest_text_with_custom_doc_id(self, monkeypatch):
        """ingest_text accepts an explicit doc_id."""
        from relay.ingest import ingest_text

        pytest.importorskip("qdrant_client")
        try:
            from relay.collections import ensure_collections

            ensure_collections()
        except Exception:
            pytest.skip("Qdrant not reachable")

        tenant = f"test_custom_{uuid.uuid4().hex[:4]}"
        custom_id = f"my_doc_{uuid.uuid4().hex[:8]}"
        result = ingest_text(
            text="Custom doc content for testing.",
            tenant_id=tenant,
            valid_from="2026-01-01",
            doc_id=custom_id,
        )

        assert result.doc_id == custom_id

    def test_ingest_text_with_supersedes(self, monkeypatch):
        """ingest_text passes supersedes through correctly."""
        from relay.ingest import ingest_text

        pytest.importorskip("qdrant_client")
        try:
            from relay.collections import ensure_collections

            ensure_collections()
        except Exception:
            pytest.skip("Qdrant not reachable")

        tenant = f"test_sup_{uuid.uuid4().hex[:4]}"
        result = ingest_text(
            text="New version of the doc.",
            tenant_id=tenant,
            valid_from="2026-06-01",
            supersedes=["old_doc_001"],
        )
        assert isinstance(result, IngestResult)
        assert result.epoch_id == 1


class TestVideoIntegration:
    """End-to-end: mocked TwelveLabs → real relay ingestion → query → verify.

    Requires a running Qdrant instance (skipped if not reachable).
    """

    TEST_TENANT = f"test_video_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(scope="class", autouse=True)
    def _setup(self, qdrant_client):
        from relay.collections import ensure_collections

        ensure_collections()

    def test_full_video_pipeline(self, mock_twelvelabs):
        """Mocked video analysis + real relay ingestion + query + verify."""
        from pkg.twelvelabs.ingest_video import ingest_video_url
        from relay.query import query
        from relay.verify import verify_retrieval

        # Step 1: Ingest a video
        result = ingest_video_url(
            video_url="https://example.com/demo_video.mp4",
            tenant_id=self.TEST_TENANT,
            valid_from="2026-01-01",
            prompt="Summarize this technical demo",
            semantic_tags=["demo", "architecture"],
        )

        assert isinstance(result, IngestResult)
        assert result.epoch_id == 1
        assert result.doc_id is not None

        # Step 2: Query the ingested video transcript
        qr = query(
            text="messaging system",
            tenant_id=self.TEST_TENANT,
        )
        assert qr.result_count > 0
        qr.request_id

        # Step 3: Verify the retrieval
        vr = verify_retrieval(
            request_id=qr.request_id,
            tenant_id=self.TEST_TENANT,
        )
        assert vr.status.value == "VERIFIED"

    def test_multiple_video_ingest_creates_sequential_epochs(self, mock_twelvelabs):
        """Ingesting two videos creates sequential epochs."""
        from pkg.twelvelabs.ingest_video import ingest_video_url

        tenant = f"test_seq_{uuid.uuid4().hex[:4]}"

        r1 = ingest_video_url(
            video_url="https://example.com/video1.mp4",
            tenant_id=tenant,
            valid_from="2026-01-01",
        )
        r2 = ingest_video_url(
            video_url="https://example.com/video2.mp4",
            tenant_id=tenant,
            valid_from="2026-06-01",
        )

        assert r1.epoch_id == 1
        assert r2.epoch_id == 2
        assert r2.epoch_id > r1.epoch_id
