"""Generate a tileable Perlin noise texture for atmospheric layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _perlin_tileable(
    size: int,
    grid_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single-octave, perfectly tileable Perlin noise layer."""
    if size % grid_size != 0:
        msg = f"size {size} must be divisible by grid_size {grid_size}"
        raise ValueError(msg)

    # Random gradient vectors on a wrapping grid.
    angles = rng.random((grid_size, grid_size)) * (2.0 * np.pi)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Coordinate grid in "cell space".
    coords = np.linspace(0.0, grid_size, num=size, endpoint=False)
    gx, gy = np.meshgrid(coords, coords, indexing="xy")

    x0 = np.floor(gx).astype(int) % grid_size
    y0 = np.floor(gy).astype(int) % grid_size
    x1 = (x0 + 1) % grid_size
    y1 = (y0 + 1) % grid_size

    # Local coordinates within each grid cell.
    xf = gx - np.floor(gx)
    yf = gy - np.floor(gy)

    # Gradient vectors at corners.
    g00 = gradients[x0, y0]
    g10 = gradients[x1, y0]
    g01 = gradients[x0, y1]
    g11 = gradients[x1, y1]

    # Vectors from corners to point.
    d00 = np.stack((xf, yf), axis=-1)
    d10 = np.stack((xf - 1.0, yf), axis=-1)
    d01 = np.stack((xf, yf - 1.0), axis=-1)
    d11 = np.stack((xf - 1.0, yf - 1.0), axis=-1)

    # Dot products.
    n00 = np.sum(g00 * d00, axis=-1)
    n10 = np.sum(g10 * d10, axis=-1)
    n01 = np.sum(g01 * d01, axis=-1)
    n11 = np.sum(g11 * d11, axis=-1)

    # Interpolate.
    u = _fade(xf)
    v = _fade(yf)
    nx0 = _lerp(n00, n10, u)
    nx1 = _lerp(n01, n11, u)
    return _lerp(nx0, nx1, v)


def generate_tileable_fractal_perlin(
    size: int = 256,
    base_grid: int = 8,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate multi-octave tileable Perlin noise."""
    rng = np.random.default_rng(seed)
    noise = np.zeros((size, size), dtype=np.float32)
    max_amp = 0.0

    for octave in range(octaves):
        grid_size = base_grid * (2**octave)
        amp = persistence**octave
        layer = _perlin_tileable(size, grid_size, rng)
        noise += layer * amp
        max_amp += amp

    noise /= max_amp
    return (noise + 1.0) * 0.5


def main() -> None:
    output_path = Path("assets/textures/cloud_noise.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noise = generate_tileable_fractal_perlin()
    noise_uint8 = (noise * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(noise_uint8, mode="L").save(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
