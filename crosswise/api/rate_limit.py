"""In-memory sliding-window rate limiting for endpoints that spend API money.

State lives in this process: fine for the single-process server this app runs
as, but limits reset on restart and aren't shared across workers.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Hashable, List, Optional, Tuple

from fastapi import HTTPException, Request

from crosswise.config import settings

HOUR = 3600.0
DAY = 86400.0


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._hits: Dict[Hashable, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(self, rules: List[Tuple[Hashable, int, float]]) -> Optional[float]:
        """Record one hit against every rule, or none if any rule is full.

        Each rule is (key, limit, window_seconds). Returns None when allowed,
        otherwise the seconds until the fullest rule frees a slot. Checking all
        rules before recording means a rejected request uses up no quota.
        """
        now = self._clock()
        with self._lock:
            retry_after = 0.0
            for key, limit, window in rules:
                hits = self._hits[key]
                while hits and hits[0] <= now - window:
                    hits.popleft()
                if len(hits) >= limit:
                    retry_after = max(retry_after, hits[0] + window - now)
            if retry_after > 0:
                return retry_after
            for key, _, _ in rules:
                self._hits[key].append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def _rules(kind: str, client: str) -> List[Tuple[Hashable, int, float]]:
    if kind == "solve":
        return [
            (("solve", client), settings.RATE_LIMIT_SOLVES_PER_HOUR, HOUR),
            # Global cap: the real denial-of-wallet guard, since per-client keys
            # are trivially varied behind proxies.
            (("solve", "*"), settings.RATE_LIMIT_SOLVES_PER_DAY, DAY),
        ]
    if kind == "upload":
        return [(("upload", client), settings.RATE_LIMIT_UPLOADS_PER_HOUR, HOUR)]
    raise ValueError(f"Unknown rate-limit kind: {kind}")


def rate_limited(kind: str):
    """FastAPI dependency limiting one kind of request ("solve" or "upload")."""
    async def dependency(request: Request) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        client = request.client.host if request.client else "unknown"
        retry_after = limiter.acquire(_rules(kind, client))
        if retry_after is not None:
            raise HTTPException(
                429,
                f"Too many {kind} requests; try again in {int(retry_after) + 1}s",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
    return dependency
