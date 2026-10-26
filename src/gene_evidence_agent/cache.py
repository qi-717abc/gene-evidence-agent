"""Small thread-safe in-memory TTL cache for immutable-style API results."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from time import monotonic
from typing import Callable, Generic, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


class TTLCache(Generic[Key, Value]):
    """Bounded least-recently-used cache that expires entries after a TTL."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 900,
        max_entries: int = 256,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[Key, tuple[float, Value]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Key) -> Value | None:
        """Return a copy of an unexpired value, or ``None`` on a cache miss."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return deepcopy(value)

    def set(self, key: Key, value: Value) -> None:
        """Store an isolated copy and evict least-recently-used entries as needed."""

        if self._ttl_seconds == 0:
            return
        with self._lock:
            self._entries[key] = (self._clock() + self._ttl_seconds, deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove all cache entries."""

        with self._lock:
            self._entries.clear()
