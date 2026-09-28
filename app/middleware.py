import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

# Declaring the header as a security scheme is what makes the "Authorize"
# button appear in Swagger UI. auto_error=False so a missing header doesn't
# 403 on its own — we decide below (and skip entirely when no key is set).
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(x_api_key: str | None = Security(api_key_header)):
    # Auth disabled when no key configured.
    if not settings.api_key:
        return

    # Constant-time compare to avoid leaking the key via timing.
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
