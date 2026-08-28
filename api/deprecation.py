"""Compatibility responses for retired product APIs."""

from __future__ import annotations

from fastapi.responses import JSONResponse

DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "2026-12-31",
}


def gone_response(*, replacement: str, message: str) -> JSONResponse:
    """Return a machine-readable, versioned retirement response."""

    headers = {
        **DEPRECATION_HEADERS,
        "Link": f'<{replacement}>; rel="successor-version"',
    }
    return JSONResponse(
        status_code=410,
        headers=headers,
        content={
            "status": "error",
            "code": "legacy_endpoint_deprecated",
            "message": message,
            "replacement": replacement,
        },
    )
