"""Field-of-view computation using native symmetric shadowcasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from brileta.util._native import fov as _c_fov
except ImportError as exc:  # pragma: no cover - fails fast by design
    raise ImportError(
        "brileta.util._native is required. "
        "Build native extensions with `make` (or `uv pip install -e .`)."
    ) from exc


def compute_fov(
    transparent: NDArray[np.bool_],
    origin: tuple[int, int],
    radius: int,
) -> NDArray[np.bool_]:
    """Compute visible tiles from *origin* using native symmetric shadowcasting."""
    # empty_like (not zeros_like) - the C extension zeroes the array itself.
    visible = np.empty_like(transparent, dtype=np.bool_)
    ox, oy = origin
    _c_fov(transparent, visible, ox, oy, radius)
    return visible
