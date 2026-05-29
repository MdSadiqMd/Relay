"""TwelveLabs analysis and asset functions.

Provides:
  - :func:`analyze_video` — sync Pegasus analysis
  - :func:`analyze_video_stream` — streaming Pegasus analysis
  - :func:`upload_asset` — upload a video URL as a reusable asset
  - :class:`TwelveLabsClient` — reusable class wrapper

All error classes and the SDK singleton are defined in sibling modules
(:mod:`errors` and :mod:`session`) and re-exported here so existing
import paths (``from pkg.twelvelabs.client import ...``) keep working.
"""

from typing import Iterator, Optional

from relay.config import CONFIG

from pkg.twelvelabs.errors import (  # noqa: F401
    TwelveLabsError,
    _wrap_sdk_error,
)
from pkg.twelvelabs.session import get_client  # noqa: F401


def analyze_video(
    video_url: str,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """Analyze a video with Pegasus and return generated text.

    Args:
        video_url: Publicly accessible URL to a video file.
        prompt: Analysis prompt (default: technical summary).
        model_name: Pegasus model version (default from config).
        max_tokens: Maximum tokens to generate (Pegasus 1.5: up to 65536).
        start_time: Start of analysis window in seconds (Pegasus 1.5 only).
        end_time: End of analysis window in seconds (Pegasus 1.5 only).

    Returns:
        Generated text output from Pegasus.

    Raises:
        TwelveLabsError: On API failure.
    """
    from twelvelabs.types import VideoContext_Url

    client = get_client()
    prompt = prompt or (
        "Provide a detailed technical summary with key decisions, "
        "timestamps, and architectural details."
    )
    model = model_name or CONFIG.twelve_labs_model

    kwargs: dict = {
        "model_name": model,
        "video": VideoContext_Url(url=video_url),
        "prompt": prompt,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if start_time is not None:
        kwargs["start_time"] = start_time
    if end_time is not None:
        kwargs["end_time"] = end_time

    try:
        result = client.analyze(**kwargs)
        return result.data or ""
    except TwelveLabsError:
        raise
    except Exception as exc:
        raise _wrap_sdk_error(exc) from exc


def analyze_video_stream(
    video_url: str,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Iterator[str]:
    """Analyze a video with Pegasus and yield text fragments in real-time.

    Args:
        Same as :func:`analyze_video`.

    Yields:
        Text fragments as they are generated.

    Raises:
        TwelveLabsError: On API failure.
    """
    from twelvelabs.types import VideoContext_Url

    client = get_client()
    prompt = prompt or (
        "Provide a detailed technical summary with key decisions, "
        "timestamps, and architectural details."
    )
    model = model_name or CONFIG.twelve_labs_model

    kwargs: dict = {
        "model_name": model,
        "video": VideoContext_Url(url=video_url),
        "prompt": prompt,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if start_time is not None:
        kwargs["start_time"] = start_time
    if end_time is not None:
        kwargs["end_time"] = end_time

    try:
        stream = client.analyze_stream(**kwargs)
        for chunk in stream:
            if chunk.event_type == "text_generation" and chunk.text:
                yield chunk.text
    except TwelveLabsError:
        raise
    except Exception as exc:
        raise _wrap_sdk_error(exc) from exc


def upload_asset(video_url: str) -> str:
    """Upload a video URL as a reusable TwelveLabs asset.

    Args:
        video_url: Publicly accessible URL to a video file.

    Returns:
        The asset ID string.

    Raises:
        TwelveLabsError: On upload failure or timeout.
    """
    import time

    client = get_client()
    try:
        asset = client.assets.create(method="url", url=video_url)
        if not asset.id:
            raise TwelveLabsError("Asset creation returned no ID")

        for _ in range(120):  # up to 10 min
            detail = client.assets.retrieve(asset_id=asset.id)
            if detail.status == "ready":
                return asset.id
            if detail.status == "failed":
                raise TwelveLabsError(f"Asset processing failed: {asset.id}")
            time.sleep(5)

        raise TwelveLabsError(f"Asset processing timed out: {asset.id}")
    except TwelveLabsError:
        raise
    except Exception as exc:
        raise _wrap_sdk_error(exc) from exc


class TwelveLabsClient:
    """Reusable wrapper around the TwelveLabs SDK singleton.

    Use :func:`analyze_video` for the stateless convenience function,
    or instantiate this class when you need explicit configuration.
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            from twelvelabs import TwelveLabs

            self._client = TwelveLabs(api_key=api_key)
        else:
            self._client = get_client()
        self._model = CONFIG.twelve_labs_model

    def analyze(
        self,
        video_url: str,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Analyze a video and return generated text."""
        from twelvelabs.types import VideoContext_Url

        prompt = prompt or (
            "Provide a detailed technical summary with key decisions, "
            "timestamps, and architectural details."
        )
        model = model_name or self._model
        kwargs: dict = {
            "model_name": model,
            "video": VideoContext_Url(url=video_url),
            "prompt": prompt,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        try:
            result = self._client.analyze(**kwargs)
            return result.data or ""
        except Exception as exc:
            raise _wrap_sdk_error(exc) from exc
