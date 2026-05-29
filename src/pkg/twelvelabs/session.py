"""TwelveLabs SDK client singleton.

Provides a single shared ``twelvelabs.TwelveLabs`` instance per process.
Call :func:`reset_client` to force re-creation (key rotation, tests).
"""

from relay.config import CONFIG

from pkg.twelvelabs.errors import TwelveLabsError

_sdk_client = None


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


def get_client():
    """Get or create the TwelveLabs SDK client singleton.

    Returns:
        A ``twelvelabs.TwelveLabs`` instance.

    Raises:
        TwelveLabsError: If no API key is configured.
    """
    global _sdk_client
    if _sdk_client is None:
        from twelvelabs import TwelveLabs

        _sdk_client = TwelveLabs(api_key=get_api_key())
    return _sdk_client


def reset_client() -> None:
    """Reset the singleton — forces re-creation on next :func:`get_client` call.

    Use in tests (between test cases) or after rotating the API key.
    """
    global _sdk_client
    _sdk_client = None
