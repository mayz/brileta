"""Unit tests for the RNG stream system."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from catley.util import rng
from catley.util.rng import RNGProvider, RNGStream


class TestRNGStream:
    """Tests for RNGStream proxy behavior."""

    def test_stream_proxies_random_methods(self) -> None:
        """RNGStream exposes standard Random methods."""
        provider = RNGProvider(master_seed=42)
        stream = provider.get("test.domain")

        # All these should work without error
        _ = stream.random()
        _ = stream.randint(1, 10)
        _ = stream.randrange(0, 100)
        _ = stream.choice([1, 2, 3])
        _ = stream.choices([1, 2, 3], k=2)
        _ = stream.uniform(0.0, 1.0)
        _ = stream.gauss(0.0, 1.0)
        _ = stream.sample([1, 2, 3, 4, 5], k=2)
        _ = stream.getrandbits(8)

        items = [1, 2, 3]
        stream.shuffle(items)

    def test_cached_proxy_works_after_reset(self) -> None:
        """Cached RNGStream references continue to work after reset()."""
        provider = RNGProvider(master_seed=42)
        stream = provider.get("test.domain")

        # Get some values with seed 42
        val1 = stream.randint(0, 1000)

        # Reset with a different seed
        provider.reset(master_seed=99)

        # The cached stream should now use the new seed - consume a value
        _ = stream.randint(0, 1000)

        # Reset back to 42 and verify we get the same first value
        provider.reset(master_seed=42)
        val2 = stream.randint(0, 1000)

        assert val1 == val2


class TestRNGProvider:
    """Tests for RNGProvider seed derivation and isolation."""

    def test_same_seed_produces_same_sequence(self) -> None:
        """Same master seed + domain produces identical sequence."""
        provider1 = RNGProvider(master_seed=12345)
        provider2 = RNGProvider(master_seed=12345)

        stream1 = provider1.get("combat.dice")
        stream2 = provider2.get("combat.dice")

        values1 = [stream1.randint(1, 20) for _ in range(10)]
        values2 = [stream2.randint(1, 20) for _ in range(10)]

        assert values1 == values2

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Different master seeds produce different sequences."""
        provider1 = RNGProvider(master_seed=111)
        provider2 = RNGProvider(master_seed=222)

        stream1 = provider1.get("combat.dice")
        stream2 = provider2.get("combat.dice")

        values1 = [stream1.randint(1, 1000) for _ in range(10)]
        values2 = [stream2.randint(1, 1000) for _ in range(10)]

        assert values1 != values2

    def test_different_domains_are_isolated(self) -> None:
        """Different domains produce independent sequences."""
        provider = RNGProvider(master_seed=42)

        stream_a = provider.get("domain.a")
        stream_b = provider.get("domain.b")

        # Get values from domain A
        values_a = [stream_a.randint(1, 1000) for _ in range(5)]

        # Reset and get values from A again, but consume some from B first
        provider.reset(master_seed=42)
        stream_a = provider.get("domain.a")
        stream_b = provider.get("domain.b")

        # Consume values from B - this should NOT affect A
        _ = [stream_b.randint(1, 1000) for _ in range(100)]

        # A should still produce the same sequence
        values_a_again = [stream_a.randint(1, 1000) for _ in range(5)]

        assert values_a == values_a_again

    def test_domain_access_order_does_not_matter(self) -> None:
        """Accessing domains in different order produces same results."""
        # First: access A then B
        provider1 = RNGProvider(master_seed=42)
        stream1_a = provider1.get("domain.a")
        stream1_b = provider1.get("domain.b")
        val1_a = stream1_a.randint(1, 1000)
        val1_b = stream1_b.randint(1, 1000)

        # Second: access B then A
        provider2 = RNGProvider(master_seed=42)
        stream2_b = provider2.get("domain.b")
        stream2_a = provider2.get("domain.a")
        val2_b = stream2_b.randint(1, 1000)
        val2_a = stream2_a.randint(1, 1000)

        assert val1_a == val2_a
        assert val1_b == val2_b


class TestModuleLevelAPI:
    """Tests for the module-level init/get/reset functions."""

    def test_reset_without_init_raises(self) -> None:
        """reset() raises RuntimeError if called before init()."""
        # Save current provider state
        import catley.util.rng as rng_module

        saved_provider = rng_module._provider

        try:
            # Force provider to None
            rng_module._provider = None

            with pytest.raises(RuntimeError, match="RNG not initialized"):
                rng.reset(0)
        finally:
            # Restore provider
            rng_module._provider = saved_provider

    def test_get_auto_initializes(self) -> None:
        """get() auto-initializes if provider doesn't exist."""
        import catley.util.rng as rng_module

        saved_provider = rng_module._provider

        try:
            rng_module._provider = None
            stream = rng.get("test.auto")
            # Should not raise, and should return a valid stream
            assert isinstance(stream, RNGStream)
            _ = stream.randint(1, 10)
        finally:
            rng_module._provider = saved_provider

    def test_init_resets_existing_provider(self) -> None:
        """init() resets existing provider rather than replacing it."""
        # Get a stream reference
        stream = rng.get("test.init")
        rng.init(42)
        val1 = stream.randint(0, 1000)

        # Re-init with same seed - stream should still work
        rng.init(42)
        val2 = stream.randint(0, 1000)

        assert val1 == val2


class TestCrossSessionDeterminism:
    """Tests that verify determinism across Python sessions.

    These tests spawn subprocess to verify that the RNG system produces
    identical results in separate Python processes (which would fail if
    we used hash() instead of crc32).
    """

    def test_seed_derivation_is_deterministic_across_processes(self) -> None:
        """Same seed produces same sequence in different Python processes."""
        # Script that prints RNG values
        script = """
import sys
sys.path.insert(0, '.')
from catley.util.rng import RNGProvider
provider = RNGProvider(master_seed=12345)
stream = provider.get("test.cross_session")
values = [stream.randint(1, 10000) for _ in range(5)]
print(",".join(map(str, values)))
"""
        # Run twice in separate processes
        result1 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        result2 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

        assert result1.returncode == 0, f"Process 1 failed: {result1.stderr}"
        assert result2.returncode == 0, f"Process 2 failed: {result2.stderr}"

        values1 = result1.stdout.strip()
        values2 = result2.stdout.strip()

        assert values1 == values2, (
            f"Cross-session determinism failed!\n"
            f"Process 1: {values1}\n"
            f"Process 2: {values2}"
        )
