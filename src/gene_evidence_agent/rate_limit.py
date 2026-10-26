"""Async request pacing for shared upstream API limits."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Awaitable, Callable


class AsyncRateLimiter:
    """Serialize calls and enforce a minimum interval between requests."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._next_allowed_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until another request is allowed to start."""

        async with self._lock:
            now = self._clock()
            delay = self._next_allowed_at - now
            if delay > 0:
                await self._sleeper(delay)
                now = self._clock()
            self._next_allowed_at = max(now, self._next_allowed_at) + self.interval_seconds
