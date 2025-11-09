from __future__ import annotations

from fastapi import HTTPException, Request, status

from inviol_image_analyser_assignment.config.settings import settings


def require_api_key(request: Request) -> None:

    expected = settings.api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "server_misconfigured",
                "message": "API key is not configured on the server.",
            },
        )

    provided = request.headers.get("x-api-key")
    if not provided or provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "unauthorized",
                "message": "Invalid or missing API key.",
            },
        )
