"""Pegasus video segmentation via async analysis tasks.

Uses ``client.analyze_async.tasks.create()`` with
``analysis_mode="time_based_metadata"`` and ``response_format.type=
"segment_definitions"`` to extract timestamped segments from a video.

Each segment becomes a relay document with ``valid_from`` set to the
segment's start time for sub-second temporal queries.

Falls back to paragraph-splitting on the sync ``analyze()`` output when
async segmentation returns no segments (e.g., video too short or model
limitation).
"""

import time as _time
from pathlib import Path
from typing import Optional

from relay.ingest import ingest_text
from relay.models import IngestResult

from pkg.twelvelabs.errors import TwelveLabsError
from pkg.twelvelabs.session import get_client


def ingest_video_segments(
    video_url: str,
    tenant_id: str,
    model_name: Optional[str] = None,
    semantic_tags: Optional[list[str]] = None,
    min_segment_duration: float = 4.0,
    max_segment_duration: float = 60.0,
) -> list[IngestResult]:
    """Segment a video with Pegasus and ingest each segment into relay.

    Attempts async ``time_based_metadata`` segmentation first. If the
    video is too short or the API returns no segments, falls back to
    sync analysis with paragraph-based splitting.

    Args:
        video_url: Publicly accessible video URL.
        tenant_id: relay tenant for the ingested documents.
        model_name: Pegasus model version (default: ``pegasus1.5``).
        semantic_tags: Optional tags applied to every segment doc.
        min_segment_duration: Minimum segment length in seconds.
        max_segment_duration: Maximum segment length in seconds.

    Returns:
        List of IngestResult, one per segment.

    Raises:
        TwelveLabsError: If video analysis fails.
    """
    tags = ["video", "segment"]
    if semantic_tags:
        tags.extend(semantic_tags)

    source_name = Path(video_url).stem.replace(" ", "_")[:32] or "video"

    # Try async segmentation first
    segments = _async_segment(
        video_url, model_name, min_segment_duration, max_segment_duration
    )

    if segments:
        results: list[IngestResult] = []
        for start_sec, text in segments:
            results.append(
                ingest_text(
                    text=text,
                    tenant_id=tenant_id,
                    valid_from=f"{start_sec}s",
                    source_file=f"seg_{source_name}_{int(start_sec)}s.md",
                    semantic_tags=tags,
                )
            )
        return results

    # Fallback: sync analysis → paragraph split
    return _fallback_paragraph_split(
        video_url, tenant_id, model_name, tags, source_name
    )


def _async_segment(
    video_url: str,
    model_name: Optional[str],
    min_seg: float,
    max_seg: float,
) -> list[tuple[float, str]]:
    """Run async segmentation and return [(start_sec, text), ...]."""
    from twelvelabs.types import (
        AsyncResponseFormat,
        SegmentDefinition,
        VideoContext_Url,
    )

    client = get_client()
    model = model_name or "pegasus1.5"

    try:
        task_resp = client.analyze_async.tasks.create(
            model_name=model,
            video=VideoContext_Url(url=video_url),
            analysis_mode="time_based_metadata",
            response_format=AsyncResponseFormat(
                type="segment_definitions",
                segment_definitions=[
                    SegmentDefinition(
                        id="scene",
                        description="A distinct scene or topic in the video.",
                    ),
                ],
            ),
            min_segment_duration=min_seg,
            max_segment_duration=max_seg,
        )
        task_id = task_resp.task_id

        # Poll until ready
        for _ in range(360):
            status = client.analyze_async.tasks.retrieve(task_id=task_id)
            if status.status == "ready":
                return _parse_task_result(status)
            if status.status == "failed":
                return []  # fall back to sync
            _time.sleep(5)

        return []  # timed out, fall back
    except Exception:
        return []  # any error → fall back to sync


def _parse_task_result(task) -> list[tuple[float, str]]:
    """Extract (start_sec, text) pairs from a completed analysis task."""
    if not task.result or not task.result.data:
        return []

    # The result.data is a string — for time_based_metadata it contains
    # JSON with timestamped segments. Parse it.
    import json

    try:
        data = json.loads(task.result.data)
    except (json.JSONDecodeError, TypeError):
        # Not JSON — treat the whole thing as a single segment
        if task.result.data:
            return [(0.0, task.result.data)]
        return []

    segments: list[tuple[float, str]] = []
    # The response is typically {"segments": [...]} or a list directly
    items = data if isinstance(data, list) else data.get("segments", [])
    for item in items:
        start = float(item.get("start", item.get("start_sec", 0.0)))
        text = item.get("text", item.get("description", item.get("data", "")))
        if text:
            segments.append((start, str(text)))

    return segments


# Fallback: sync analysis → paragraph split
def _fallback_paragraph_split(
    video_url: str,
    tenant_id: str,
    model_name: Optional[str],
    tags: list[str],
    source_name: str,
) -> list[IngestResult]:
    """Sync analyze → split by paragraphs → ingest each."""
    from pkg.twelvelabs.client import analyze_video

    try:
        transcript = analyze_video(
            video_url=video_url,
            prompt="Provide a detailed technical summary. "
            "Use paragraph breaks for each logical section.",
            model_name=model_name,
        )
    except Exception as exc:
        raise TwelveLabsError(f"Video segmentation failed: {exc}") from exc

    if not transcript:
        return []

    paragraphs = [p.strip() for p in transcript.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [transcript]

    results: list[IngestResult] = []
    for i, para in enumerate(paragraphs):
        start_sec = i * 10
        results.append(
            ingest_text(
                text=para,
                tenant_id=tenant_id,
                valid_from=f"{start_sec}s",
                source_file=f"seg_{source_name}_{start_sec}s.md",
                semantic_tags=tags,
            )
        )
    return results
