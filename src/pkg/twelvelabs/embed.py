"""TwelveLabs Embed API v2 — multimodal video and text embeddings.

Uses the Embed v2 API (``client.embed.v_2``) which supersedes the
deprecated v1 endpoints. Produces embeddings in the Marengo 3.0 latent
space, enabling cross-modal (text ↔ video ↔ image ↔ audio) similarity.

  - ``compute_video_embedding()``: async task-based for video content
  - ``compute_text_query_embedding()``: sync for query text in the same space
"""

import time
from typing import Optional

from relay.config import CONFIG

from pkg.twelvelabs.errors import TwelveLabsError, _wrap_sdk_error
from pkg.twelvelabs.session import get_client


def compute_video_embedding(
    video_url: str,
    model_name: Optional[str] = None,
    embedding_scope: Optional[list[str]] = None,
) -> list[float]:
    """Embed a video into the Marengo multimodal vector space.

    Uses the Embed v2 async tasks API for video content (supports up
    to 4 hours). Blocks until the task completes (polling every 5s).

    Args:
        video_url: Publicly accessible URL to a video file.
        model_name: Embed model (default: ``marengo3.0`` from config).
        embedding_scope: ``["clip"]``, ``["asset"]``, or both.
            Default ``["asset"]`` for a single whole-video embedding.

    Returns:
        Fused-modality embedding vector.

    Raises:
        TwelveLabsError: If embedding fails.
    """
    from twelvelabs.types import MediaSource, VideoInputRequest

    client = get_client()
    model = model_name or CONFIG.twelve_labs_embed_model

    scope = embedding_scope or ["asset"]

    try:
        task = client.embed.v_2.tasks.create(
            input_type="video",
            model_name=model,
            video=VideoInputRequest(
                media_source=MediaSource(url=video_url),
                embedding_option=["visual", "audio"],
                embedding_scope=scope,
                embedding_type=["fused_embedding"],
            ),
        )
        if not task.id:
            raise TwelveLabsError("Video embedding task returned no ID")

        # Poll until ready
        for _ in range(360):  # up to 30 min
            result = client.embed.v_2.tasks.retrieve(task_id=task.id)
            if result.status == "ready":
                if result.data:
                    return result.data[0].embedding
                raise TwelveLabsError("Video embedding task ready but no data")
            if result.status == "failed":
                raise TwelveLabsError("Video embedding task failed")
            time.sleep(5)

        raise TwelveLabsError("Video embedding task timed out")
    except TwelveLabsError:
        raise
    except Exception as exc:
        raise _wrap_sdk_error(exc) from exc


def compute_text_query_embedding(
    text: str,
    model_name: Optional[str] = None,
) -> list[float]:
    """Embed a text query into the same space as video embeddings.

    Uses the Embed v2 sync API — no polling needed.

    Args:
        text: Query text to embed.
        model_name: Embed model (default: ``marengo3.0`` from config).

    Returns:
        Embedding vector comparable to video embeddings.

    Raises:
        TwelveLabsError: If embedding fails.
    """
    from twelvelabs.types import TextInputRequest

    client = get_client()
    model = model_name or CONFIG.twelve_labs_embed_model

    try:
        result = client.embed.v_2.create(
            input_type="text",
            model_name=model,
            text=TextInputRequest(input_text=text),
        )
        if result.data:
            return result.data[0].embedding
        raise TwelveLabsError("Text query embedding returned no data")
    except TwelveLabsError:
        raise
    except Exception as exc:
        raise _wrap_sdk_error(exc) from exc
