# catley/util/caching.py

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .live_vars import live_variable_registry

# Define generic types for keys and values
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


@dataclass
class CacheStats:
    """Statistics for a ResourceCache instance."""

    hits: int = 0
    misses: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return (self.hits / self.total_lookups) * 100.0

    def __repr__(self) -> str:
        return f"{self.hits} hits, {self.misses} misses ({self.hit_rate:.1f}% hit rate)"


class ResourceCache[KeyType, ValueType]:
    """
    A generic, size-limited, Least Recently Used (LRU) cache.

    This cache stores expensive-to-create resources and retrieves them with a
    key derived from game state. It automatically handles LRU eviction and
    tracks performance stats.

    An optional `on_evict` callback can be provided to handle cleanup of
    evicted resources, such as destroying textures if needed.
    """

    def __init__(
        self,
        name: str,
        max_size: int = 16,
        on_evict: Callable[[ValueType], None] | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("Cache max_size must be a positive integer.")
        self.name = name
        self.max_size = max_size
        self._cache: OrderedDict[KeyType, ValueType] = OrderedDict()
        self.stats = CacheStats()
        self.on_evict = on_evict

        live_variable_registry.register(
            name=f"cache.{self.name}.stats",
            getter=lambda: str(self.stats),
            setter=None,
            description=f"Live stats for the {self.name} cache.",
        )

    def get(self, key: KeyType) -> ValueType | None:
        """
        Retrieve an item from the cache.

        If the item is found, it's marked as recently used.
        Returns the item if found, otherwise None.
        """
        if key not in self._cache:
            self.stats.misses += 1
            return None

        # Hit! Move the key to the end to mark it as recently used.
        self._cache.move_to_end(key)
        self.stats.hits += 1
        return self._cache[key]

    def store(self, key: KeyType, value: ValueType) -> None:
        """
        Store an item in the cache.

        If the cache is full, the least recently used item is evicted.
        """
        self._cache[key] = value
        self._cache.move_to_end(key)

        # Use a while-loop for defensive correctness. This ensures the cache
        # size is enforced even if a future change adds multiple items at once.
        while len(self._cache) > self.max_size:
            # Evict the least recently used item
            _evicted_key, evicted_value = self._cache.popitem(last=False)
            if self.on_evict:
                self.on_evict(evicted_value)

    def clear(self) -> None:
        """Clear all items from the cache and reset stats."""
        if self.on_evict:
            for value in self._cache.values():
                self.on_evict(value)

        self._cache.clear()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._cache)

    def __str__(self) -> str:
        """User-friendly string representation for reports."""
        return (
            f"{self.name} Cache: {self.stats.hits} hits, {self.stats.misses} misses "
            f"({self.stats.hit_rate:.1f}% hit rate)"
        )

    def __repr__(self) -> str:
        """More detailed representation for debugging."""
        return (
            f"<{self.__class__.__name__} '{self.name}' "
            f"size={len(self)}/{self.max_size}, stats={self.stats!r}>"
        )
