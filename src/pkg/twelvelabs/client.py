"""TwelveLabs client wrapper with config-driven setup."""

from typing import Optional

from relay.config import CONFIG


class TwelveLabsError(Exception):
    """Raised when a TwelveLabs API call fails."""


def get_api_key() -> str:
    """Resolve the TwelveLabs API key from config.

    Raises:
        TwelveLabsError: If no API key is configured.
    """
    key = CONFIG.twelve_labs_api_key
    if not key:
        raise TwelveLabsError(
            "TwelveLabs API key is required. "
            "Set RELAY_TWELVE_LABS_API_KEY in your environment or .env file."
        )
    return key


def analyze_video(
    video_url: str,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Analyze a video with TwelveLabs Pegasus and return structured text.

    Args:
        video_url: Publicly accessible URL to a video file.
        prompt: Analysis prompt for Pegasus.
            Defaults to a technical summary prompt.
        model_name: Pegasus model version.
            Defaults to config ``twelve_labs_model``.

    Returns:
        Generated text output from Pegasus.

    Raises:
        TwelveLabsError: If the API call fails or no API key is configured.
    """
    from twelvelabs import TwelveLabs
    from twelvelabs.types import VideoContext_Url

    api_key = get_api_key()
    prompt = prompt or (
        "Provide a detailed technical summary with key decisions, "
        "timestamps, and architectural details."
    )
    model = model_name or CONFIG.twelve_labs_model

    client = TwelveLabs(api_key=api_key)

    try:
        result = client.analyze(
            model_name=model,
            video=VideoContext_Url(url=video_url),
            prompt=prompt,
        )
        return result.data or ""
    except Exception as exc:
        raise TwelveLabsError(f"Video analysis failed: {exc}") from exc


class TwelveLabsClient:
    """Reusable wrapper around the TwelveLabs SDK.

    Use :func:`analyze_video` for the stateless convenience function,
    or instantiate this class when you need to share a client across
    multiple calls.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or get_api_key()
        from twelvelabs import TwelveLabs

        self._client = TwelveLabs(api_key=self._api_key)
        self._model = CONFIG.twelve_labs_model

    def analyze(
        self,
        video_url: str,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Analyze a video and return structured text output."""
        from twelvelabs.types import VideoContext_Url

        prompt = prompt or (
            "Provide a detailed technical summary with key decisions, "
            "timestamps, and architectural details."
        )
        model = model_name or self._model
        try:
            result = self._client.analyze(
                model_name=model,
                video=VideoContext_Url(url=video_url),
                prompt=prompt,
            )
            return result.data or ""
        except Exception as exc:
            raise TwelveLabsError(f"Video analysis failed: {exc}") from exc
