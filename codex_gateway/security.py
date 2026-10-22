from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import Header, HTTPException, status


def require_bearer_token(
    authorization: str | None,
    expected_token: str,
) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    provided = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        )


def validate_workspace(requested: str, allowed_root: Path) -> str:
    candidate = Path(requested).expanduser().resolve()
    allowed = allowed_root.resolve()
    if candidate != allowed and allowed not in candidate.parents:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="workspace is outside the allowed root",
        )
    return str(candidate)
