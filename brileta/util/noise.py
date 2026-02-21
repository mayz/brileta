"""Procedural noise generation powered by FastNoiseLite.

Provides a ``NoiseGenerator`` class that wraps the native C extension for
high-performance Perlin, Simplex, Cellular, and Value noise with optional
fractal layering and domain warping.

Example usage::

    from brileta.util.noise import NoiseGenerator, NoiseType

    gen = NoiseGenerator(seed=42, noise_type=NoiseType.PERLIN, frequency=0.05)
    value = gen.sample(10.5, 20.3)  # returns float in [-1, 1]
"""

from __future__ import annotations

from enum import IntEnum

try:
    from brileta.util._native import _NoiseState
except ImportError as exc:
    raise ImportError(
        "brileta.util._native is required. "
        "Build native extensions with `make` (or `uv pip install -e .`)."
    ) from exc


# ---------------------------------------------------------------------------
# Enum wrappers matching FastNoiseLite C enums
# ---------------------------------------------------------------------------


class NoiseType(IntEnum):
    """Noise algorithm selection - maps to fnl_noise_type."""

    OPENSIMPLEX2 = 0
    OPENSIMPLEX2S = 1
    CELLULAR = 2
    PERLIN = 3
    VALUE_CUBIC = 4
    VALUE = 5


class FractalType(IntEnum):
    """Fractal layering mode - maps to fnl_fractal_type."""

    NONE = 0
    FBM = 1
    RIDGED = 2
    PINGPONG = 3
    DOMAIN_WARP_PROGRESSIVE = 4
    DOMAIN_WARP_INDEPENDENT = 5


class CellularDistanceFunc(IntEnum):
    """Distance function for cellular noise - maps to fnl_cellular_distance_func."""

    EUCLIDEAN = 0
    EUCLIDEAN_SQ = 1
    MANHATTAN = 2
    HYBRID = 3


class CellularReturnType(IntEnum):
    """Return value for cellular noise - maps to fnl_cellular_return_type."""

    CELL_VALUE = 0
    DISTANCE = 1
    DISTANCE2 = 2
    DISTANCE2_ADD = 3
    DISTANCE2_SUB = 4
    DISTANCE2_MUL = 5
    DISTANCE2_DIV = 6


class DomainWarpType(IntEnum):
    """Domain warp algorithm - maps to fnl_domain_warp_type."""

    OPENSIMPLEX2 = 0
    OPENSIMPLEX2_REDUCED = 1
    BASIC_GRID = 2


# ---------------------------------------------------------------------------
# High-level generator
# ---------------------------------------------------------------------------


class NoiseGenerator:
    """High-level noise generator wrapping FastNoiseLite.

    All noise output is bounded to [-1, 1].

    Args:
        seed: Integer seed for deterministic generation.
        noise_type: Algorithm to use (default OpenSimplex2).
        frequency: Base sampling frequency (default 0.01).
        fractal_type: Fractal layering mode (default none).
        octaves: Number of fractal octaves (default 3).
        lacunarity: Frequency multiplier per octave (default 2.0).
        gain: Amplitude multiplier per octave (default 0.5).
        weighted_strength: Fractal weighted strength (default 0.0).
        ping_pong_strength: Ping-pong fractal strength (default 2.0).
        cellular_distance_func: Distance function for cellular noise.
        cellular_return_type: Return type for cellular noise.
        cellular_jitter_mod: Cellular point jitter (default 1.0).
        domain_warp_type: Domain warp algorithm.
        domain_warp_amp: Domain warp amplitude (default 1.0).
    """

    __slots__ = ("_state",)

    def __init__(
        self,
        seed: int = 1337,
        noise_type: NoiseType = NoiseType.OPENSIMPLEX2,
        frequency: float = 0.01,
        *,
        fractal_type: FractalType = FractalType.NONE,
        octaves: int = 3,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        cellular_distance_func: CellularDistanceFunc = CellularDistanceFunc.EUCLIDEAN_SQ,
        cellular_return_type: CellularReturnType = CellularReturnType.DISTANCE,
        cellular_jitter_mod: float = 1.0,
        domain_warp_type: DomainWarpType = DomainWarpType.OPENSIMPLEX2,
        domain_warp_amp: float = 1.0,
    ) -> None:
        # Truncate to signed 32-bit range. The C extension (FastNoiseLite)
        # stores the seed as a C int. Python ints are arbitrary precision,
        # so callers using XOR salts or getrandbits(32) can exceed INT_MAX.
        seed_i32 = int(seed) & 0xFFFF_FFFF
        if seed_i32 >= 0x8000_0000:
            seed_i32 -= 0x1_0000_0000

        self._state = _NoiseState(
            seed=seed_i32,
            noise_type=int(noise_type),
            frequency=float(frequency),
            fractal_type=int(fractal_type),
            octaves=int(octaves),
            lacunarity=float(lacunarity),
            gain=float(gain),
            weighted_strength=float(weighted_strength),
            ping_pong_strength=float(ping_pong_strength),
            cellular_distance_func=int(cellular_distance_func),
            cellular_return_type=int(cellular_return_type),
            cellular_jitter_mod=float(cellular_jitter_mod),
            domain_warp_type=int(domain_warp_type),
            domain_warp_amp=float(domain_warp_amp),
        )

    # -- Sampling ----------------------------------------------------------

    def sample(self, x: float, y: float) -> float:
        """Sample 2D noise at *(x, y)*. Returns a value in [-1, 1]."""
        return self._state.sample_2d(x, y)

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """Sample 3D noise at *(x, y, z)*. Returns a value in [-1, 1]."""
        return self._state.sample_3d(x, y, z)

    # -- Domain warping ----------------------------------------------------

    def domain_warp(self, x: float, y: float) -> tuple[float, float]:
        """Apply 2D domain warp. Returns warped *(x, y)*."""
        return self._state.domain_warp_2d(x, y)

    def domain_warp_3d(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Apply 3D domain warp. Returns warped *(x, y, z)*."""
        return self._state.domain_warp_3d(x, y, z)

    # -- Mutable properties ------------------------------------------------

    @property
    def seed(self) -> int:
        """Current seed (read-write)."""
        return self._state.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._state.seed = int(value)

    @property
    def frequency(self) -> float:
        """Current frequency (read-write)."""
        return self._state.frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._state.frequency = float(value)
