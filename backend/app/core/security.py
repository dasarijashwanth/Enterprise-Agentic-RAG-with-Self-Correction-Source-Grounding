"""Day 5: API-key auth + in-memory rate limiting.

Auth: X-API-Key header compared to VERIRAG_API_KEY. When the env var is
unset, auth is disabled entirely so local dev keeps working with no
header and no changes. Rate limiting: sliding 60s window at
VERIRAG_RATE_LIMIT_PER_MIN (default 60) requests per key — or per
client IP when auth is disabled. Exceeding it returns HTTP 429 with a
Retry-After header. Applied to the ask, ask/stream, search, and
feedback endpoints.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from .config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided: str | None = Depends(_api_key_header)) -> str | None:
    """Return the key when valid; None when auth is disabled.

    Raises 401 only when VERIRAG_API_KEY is set and the presented key
    does not match it.
    """
    expected = settings.VERIRAG_API_KEY
    if not expected:
        return None  # auth disabled: local dev keeps working
    if provided != expected:
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key"
        )
    return provided


class RateLimiter:
    """In-memory sliding-window limiter (per process)."""

    def __init__(self, per_minute: int = 60, window_s: float = 60.0):
        self.per_minute = per_minute
        self.window_s = window_s
        self._hits: dict[str, Deque[float]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def _key(self, request: Request, api_key: str | None) -> str:
        if api_key:
            return f"key:{api_key}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def check(self, request: Request, api_key: str | None = None) -> None:
        now = time.monotonic()
        key = self._key(request, api_key)
        dq = self._hits.setdefault(key, deque())
        cutoff = now - self.window_s
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= self.per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        dq.append(now)


rate_limiter = RateLimiter(per_minute=60)


def rate_limit(
    request: Request, api_key: str | None = Depends(verify_api_key)
) -> str | None:
    """Combined auth + rate-limit dependency for protected endpoints."""
    # Re-read the configured limit on each call so env changes and
    # tests take effect without restarting the process.
    rate_limiter.per_minute = settings.VERIRAG_RATE_LIMIT_PER_MIN
    rate_limiter.check(request, api_key)
    return api_key
