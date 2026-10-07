"""Token-bucket rate limiting per client IP (default 100 requests/s, burst 200)."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class TokenBucket:
    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst = rate, burst
        self.tokens, self.updated = burst, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, rate: float = 100.0, burst: float | None = None) -> None:
        super().__init__(app)
        self.rate = rate
        self.burst = burst or 2 * rate
        self.buckets: dict[str, TokenBucket] = {}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith("/v1/"):
            key = request.client.host if request.client else "unknown"
            bucket = self.buckets.get(key)
            if bucket is None:
                if len(self.buckets) > 10_000:
                    self.buckets.clear()
                bucket = self.buckets[key] = TokenBucket(self.rate, self.burst)
            if not bucket.take():
                return JSONResponse(
                    {"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "1"}
                )
        return await call_next(request)
