"""Tests for the native FastNoiseLite noise extension."""

from __future__ import annotations

import pytest

from brileta.util._native import _NoiseState
from brileta.util.noise import (
    FractalType,
    NoiseGenerator,
    NoiseType,
)

# ---------------------------------------------------------------------------
# Low-level _NoiseState tests
# ---------------------------------------------------------------------------


class TestNoiseStateLowLevel:
    """Verify the raw C extension type behaves correctly."""

    def test_default_construction(self) -> None:
        """Default _NoiseState should be constructable with no args."""
        state = _NoiseState()
        val = state.sample_2d(10.0, 20.0)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_sample_2d_returns_float_in_range(self) -> None:
        """2D sampling must return values in [-1, 1]."""
        state = _NoiseState(seed=42, frequency=0.05)
        for x in range(20):
            for y in range(20):
                val = state.sample_2d(float(x), float(y))
                assert -1.0 <= val <= 1.0

    def test_sample_3d_returns_float_in_range(self) -> None:
        """3D sampling must return values in [-1, 1]."""
        state = _NoiseState(seed=42, frequency=0.05)
        val = state.sample_3d(1.0, 2.0, 3.0)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_different_seeds_produce_different_output(self) -> None:
        """Two generators with different seeds should diverge."""
        a = _NoiseState(seed=1)
        b = _NoiseState(seed=9999)
        val_a = a.sample_2d(50.0, 50.0)
        val_b = b.sample_2d(50.0, 50.0)
        assert val_a != val_b

    def test_same_seed_is_deterministic(self) -> None:
        """Same seed + same coordinates must produce identical results."""
        a = _NoiseState(seed=42, frequency=0.1)
        b = _NoiseState(seed=42, frequency=0.1)
        assert a.sample_2d(7.5, 3.2) == b.sample_2d(7.5, 3.2)

    def test_seed_property_read_write(self) -> None:
        """The seed property should be readable and writable."""
        state = _NoiseState(seed=10)
        assert state.seed == 10
        state.seed = 999
        assert state.seed == 999

    def test_frequency_property_read_write(self) -> None:
        """The frequency property should be readable and writable."""
        state = _NoiseState(frequency=0.05)
        assert state.frequency == pytest.approx(0.05)
        state.frequency = 0.1
        assert state.frequency == pytest.approx(0.1)

    def test_domain_warp_2d_returns_tuple(self) -> None:
        """domain_warp_2d should return a (float, float) tuple."""
        state = _NoiseState(domain_warp_amp=10.0)
        result = state.domain_warp_2d(5.0, 5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_domain_warp_3d_returns_tuple(self) -> None:
        """domain_warp_3d should return a (float, float, float) tuple."""
        state = _NoiseState(domain_warp_amp=10.0)
        result = state.domain_warp_3d(5.0, 5.0, 5.0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# High-level NoiseGenerator tests
# ---------------------------------------------------------------------------


class TestNoiseGenerator:
    """Verify the Pythonic wrapper API."""

    def test_basic_construction_and_sampling(self) -> None:
        """NoiseGenerator should produce noise values in [-1, 1]."""
        gen = NoiseGenerator(seed=42, noise_type=NoiseType.PERLIN, frequency=0.05)
        val = gen.sample(10.0, 20.0)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_all_noise_types_produce_output(self) -> None:
        """Every NoiseType variant should produce valid noise."""
        for nt in NoiseType:
            gen = NoiseGenerator(seed=1, noise_type=nt, frequency=0.02)
            val = gen.sample(5.0, 5.0)
            assert -1.0 <= val <= 1.0, f"{nt.name} out of range: {val}"

    def test_fractal_fbm(self) -> None:
        """FBM fractal layering should produce different output than no fractal."""
        plain = NoiseGenerator(seed=42, frequency=0.05)
        fbm = NoiseGenerator(
            seed=42, frequency=0.05, fractal_type=FractalType.FBM, octaves=4
        )
        val_plain = plain.sample(10.0, 10.0)
        val_fbm = fbm.sample(10.0, 10.0)
        # FBM and non-fractal should generally differ
        assert val_plain != val_fbm

    def test_sample_3d(self) -> None:
        """3D sampling through the high-level API should work."""
        gen = NoiseGenerator(seed=7)
        val = gen.sample_3d(1.0, 2.0, 3.0)
        assert -1.0 <= val <= 1.0

    def test_domain_warp(self) -> None:
        """Domain warping should shift coordinates."""
        gen = NoiseGenerator(seed=42, domain_warp_amp=50.0)
        wx, wy = gen.domain_warp(100.0, 100.0)
        # With amp=50, the warped position should differ from the input
        assert (wx, wy) != (100.0, 100.0)

    def test_seed_mutation(self) -> None:
        """Changing seed after construction should alter output."""
        gen = NoiseGenerator(seed=1, frequency=0.05)
        val_before = gen.sample(10.0, 10.0)
        gen.seed = 9999
        val_after = gen.sample(10.0, 10.0)
        assert val_before != val_after

    def test_frequency_mutation(self) -> None:
        """Changing frequency after construction should alter output."""
        gen = NoiseGenerator(seed=42, frequency=0.01)
        val_before = gen.sample(10.0, 10.0)
        gen.frequency = 0.5
        val_after = gen.sample(10.0, 10.0)
        assert val_before != val_after

    def test_determinism_across_instances(self) -> None:
        """Two identically configured generators must return the same values."""
        a = NoiseGenerator(seed=42, noise_type=NoiseType.PERLIN, frequency=0.05)
        b = NoiseGenerator(seed=42, noise_type=NoiseType.PERLIN, frequency=0.05)
        assert a.sample(7.5, 3.2) == b.sample(7.5, 3.2)
