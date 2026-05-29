"""TwelveLabs video understanding integration for relay.

Provides video analysis via Pegasus models and ingestion into relay's
temporal semantic memory as epoch-versioned documents.

Usage::

    from pkg.twelvelabs import ingest_video_url

    result = ingest_video_url(
        video_url="https://example.com/demo.mp4",
        tenant_id="video_docs",
        valid_from="2026-01-01",
        prompt="Summarize this technical talk",
    )

Or from the CLI::

    relay video ingest --url <URL> --tenant <TENANT> --valid-from <DATE>
"""

from pkg.twelvelabs.client import (
    TwelveLabsClient,
    TwelveLabsError,
    analyze_video,
    analyze_video_stream,
    upload_asset,
)
from pkg.twelvelabs.ingest_video import ingest_video_url

__all__ = [
    "TwelveLabsClient",
    "TwelveLabsError",
    "analyze_video",
    "analyze_video_stream",
    "ingest_video_url",
    "upload_asset",
]
