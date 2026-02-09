"""Deterministic random number generation with isolated streams.

This module provides a centralized RNG system where each subsystem (map generation,
combat, NPC placement, etc.) gets its own independent random stream derived from
a master seed. This ensures that:

1. The game is fully deterministic from the same master seed
2. Changes to one system's random consumption don't cascade to others
3. Adding/removing systems doesn't shift other systems' random sequences

Usage:
    # At game startup
    from catley.util import rng
    rng.init(config.RANDOM_SEED)

    # In any module - cache the stream reference
    _rng = rng.get("combat.dice")

    def roll_d20() -> int:
        return _rng.randint(1, 20)

    # After rng.reset(), cached references automatically use the new stream

Domain naming convention (hierarchical):
    - "map.terrain", "map.buildings", "map.wfc"
    - "world.npc_placement", "world.containers"
    - "combat.dice", "combat.damage"
    - "audio.variation"
    - "effects.particles"
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from random import Random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from catley.types import RandomSeed

T = TypeVar("T")


class RNGStream:
    """Proxy that delegates to the current RNG for a domain.

    This wrapper allows callers to cache a reference that survives rng.reset().
    All method calls are forwarded to the underlying Random instance,
    which is looked up fresh each time from the provider.
    """

    def __init__(self, provider: RNGProvider, domain: str) -> None:
        self._provider = provider
        self._domain = domain

    def _rng(self) -> Random:
        """Get the current underlying RNG."""
        return self._provider._get_raw(self._domain)

    # -------------------------------------------------------------------------
    # Random method proxies
    # -------------------------------------------------------------------------

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        return self._rng().random()

    def randint(self, a: int, b: int) -> int:
        """Return random integer N such that a <= N <= b."""
        return self._rng().randint(a, b)

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        """Return randomly selected element from range(start, stop, step)."""
        return self._rng().randrange(start, stop, step)

    def choice(self, seq: Sequence[T]) -> T:
        """Return random element from non-empty sequence."""
        return self._rng().choice(seq)

    def choices(
        self,
        population: Sequence[T],
        weights: Sequence[float] | None = None,
        *,
        cum_weights: Sequence[float] | None = None,
        k: int = 1,
    ) -> list[T]:
        """Return k-sized list of elements chosen with replacement."""
        return self._rng().choices(
            population, weights=weights, cum_weights=cum_weights, k=k
        )

    def shuffle(self, x: list) -> None:
        """Shuffle list x in place."""
        self._rng().shuffle(x)

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        """Return k unique elements from population."""
        return self._rng().sample(population, k)

    def uniform(self, a: float, b: float) -> float:
        """Return random float N such that a <= N <= b."""
        return self._rng().uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Return Gaussian distribution with mean mu and standard deviation sigma."""
        return self._rng().gauss(mu, sigma)

    def getrandbits(self, k: int) -> int:
        """Return an integer with k random bits."""
        return self._rng().getrandbits(k)


# Type alias for functions that accept either Random or RNGStream.
# Use this in type hints: `def foo(rng: RNG) -> int:`
type RNG = Random | RNGStream


class RNGProvider:
    """Provides isolated RNG streams for different game subsystems.

    Each domain gets its own Random instance derived deterministically
    from the master seed. Domains are identified by string names.
    """

    def __init__(self, master_seed: RandomSeed = None) -> None:
        self._master_seed = master_seed
        self._streams: dict[str, Random] = {}
        self._proxies: dict[str, RNGStream] = {}

    def get(self, domain: str) -> RNGStream:
        """Get an RNG stream for the named domain.

        Returns a proxy object that can be cached. The proxy automatically
        uses the current underlying RNG, even after reset().

        Args:
            domain: Hierarchical name like "map.terrain" or "combat.dice"

        Returns:
            An RNGStream proxy with the same interface as Random
        """
        if domain not in self._proxies:
            self._proxies[domain] = RNGStream(self, domain)
        return self._proxies[domain]

    def _get_raw(self, domain: str) -> Random:
        """Get the raw Random instance for a domain (internal use)."""
        if domain not in self._streams:
            if self._master_seed is None:
                # No seed: use system entropy for non-deterministic behavior
                self._streams[domain] = Random()
            else:
                # Use crc32 instead of hash() - hash() is randomized per Python
                # session via PYTHONHASHSEED, which would break cross-session
                # determinism
                derived_seed = zlib.crc32(f"{self._master_seed}:{domain}".encode())
                self._streams[domain] = Random(derived_seed)
        return self._streams[domain]

    def reset(self, master_seed: RandomSeed = None) -> None:
        """Reset all streams with a new master seed.

        Existing RNGStream proxies remain valid and will use the new streams.

        Args:
            master_seed: New master seed for all streams
        """
        self._master_seed = master_seed
        self._streams.clear()
        # Note: _proxies are kept - they'll get fresh RNGs on next access


# =============================================================================
# Module-level API
# =============================================================================

_provider: RNGProvider | None = None


def init(master_seed: RandomSeed = None) -> None:
    """Initialize the global RNG provider with a master seed.

    If a provider already exists, resets it instead of creating a new one.
    This ensures cached RNGStream proxies continue to work after init().

    Args:
        master_seed: The master seed for all random streams.
            Can be int, str, or None for non-deterministic behavior.
    """
    global _provider
    if _provider is not None:
        # Reset existing provider so cached proxies keep working
        _provider.reset(master_seed)
    else:
        _provider = RNGProvider(master_seed)


def get(domain: str) -> RNGStream:
    """Get an RNG stream for the named domain.

    The returned RNGStream can be cached at module or instance level.
    It will automatically use the current stream even after reset().

    If the RNG provider hasn't been initialized yet, it will be auto-initialized
    with a default seed (None, which gives non-deterministic behavior).
    Call init() explicitly at game startup to set a deterministic seed.

    Args:
        domain: Hierarchical name like "map.terrain" or "combat.dice"

    Returns:
        An RNGStream proxy with the same interface as Random
    """
    global _provider
    if _provider is None:
        # Auto-initialize with default seed for module-level usage
        _provider = RNGProvider(None)
    return _provider.get(domain)


def reset(master_seed: RandomSeed = None) -> None:
    """Reset all RNG streams with a new master seed.

    Use this when regenerating the world or starting a new game.
    Existing cached RNGStream references remain valid.

    Args:
        master_seed: New master seed for all streams
    """
    if _provider is None:
        raise RuntimeError("RNG not initialized - call rng.init() first")
    _provider.reset(master_seed)
