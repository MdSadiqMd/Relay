"""TwelveLabs typed error hierarchy.

Maps SDK HTTP status codes to specific exception types so callers can
handle rate limits, auth failures, and validation errors distinctly.
"""


class TwelveLabsError(Exception):
    """Base error for all TwelveLabs API failures."""


class TwelveLabsAuthError(TwelveLabsError):
    """Authentication or permission failure (401/403)."""


class TwelveLabsRateLimitError(TwelveLabsError):
    """Rate limit exceeded (429). Caller should back off and retry."""


class TwelveLabsValidationError(TwelveLabsError):
    """Bad request or unprocessable entity (400/422)."""


class TwelveLabsNotFoundError(TwelveLabsError):
    """Resource not found (404)."""


def _wrap_sdk_error(exc: Exception) -> TwelveLabsError:
    """Map a TwelveLabs SDK HTTP exception to the appropriate typed error.

    Falls back to the base :class:`TwelveLabsError` for unknown cases.
    """
    from twelvelabs.errors import (
        BadRequestError,
        ForbiddenError,
        NotFoundError,
        TooManyRequestsError,
    )

    cls_name = type(exc).__name__

    if isinstance(exc, BadRequestError):
        return TwelveLabsValidationError(str(exc))
    if isinstance(exc, TooManyRequestsError):
        return TwelveLabsRateLimitError(str(exc))
    if isinstance(exc, NotFoundError):
        return TwelveLabsNotFoundError(str(exc))
    if isinstance(exc, ForbiddenError) or cls_name == "AuthenticationError":
        return TwelveLabsAuthError(str(exc))
    return TwelveLabsError(str(exc))
