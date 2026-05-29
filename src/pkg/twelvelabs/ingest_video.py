"""Video ingestion pipeline — analyze with TwelveLabs, ingest into relay."""

from pathlib import Path
from typing import Optional

from relay.ingest import ingest_text
from relay.models import IngestResult

from pkg.twelvelabs.client import analyze_video


def ingest_video_url(
    video_url: str,
    tenant_id: str,
    valid_from: str,
    valid_to: Optional[str] = None,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    semantic_tags: Optional[list[str]] = None,
) -> IngestResult:
    """Full pipeline: analyze a video URL with Pegasus, then ingest into relay.

    Steps:
        1. Analyze the video via TwelveLabs Pegasus → structured text.
        2. Ingest into relay with full epoch/Merkle commitment.

    Args:
        video_url: Publicly accessible video URL.
        tenant_id: relay tenant for the ingested document.
        valid_from: Temporal validity start (YYYY-MM-DD).
        valid_to: Optional temporal validity end.
        prompt: Pegasus analysis prompt.
        model_name: Pegasus model version.
        semantic_tags: Optional list of semantic tags.

    Returns:
        IngestResult from relay's ingestion pipeline.

    Raises:
        TwelveLabsError: If video analysis fails.
    """
    prompt = prompt or (
        "Provide a detailed technical summary with key decisions, "
        "timestamps, and architectural details."
    )

    transcript = analyze_video(
        video_url=video_url,
        prompt=prompt,
        model_name=model_name,
    )

    tags = ["video"]
    if semantic_tags:
        tags.extend(semantic_tags)

    return ingest_text(
        text=transcript,
        tenant_id=tenant_id,
        valid_from=valid_from,
        valid_to=valid_to,
        semantic_tags=tags,
        source_file=f"video_{_url_to_name(video_url)}.md",
    )


def _url_to_name(video_url: str) -> str:
    """Derive a short identifier from a video URL."""
    from urllib.parse import urlparse

    parsed = urlparse(video_url)
    path = Path(parsed.path)
    stem = path.stem or "video"
    return stem.replace(" ", "_")[:48]
