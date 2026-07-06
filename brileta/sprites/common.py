"""Shared helpers for procedural sprite modules."""

from __future__ import annotations

import numpy as np

from brileta.game.enums import CreatureSize


def sprite_visual_scale_for_shadow_height(
    sprite: np.ndarray,
    shadow_height: int,
) -> float:
    """Return visual scale so rendered sprite height matches physical shadow height.

    Calibration rule:
    - CreatureSize.MEDIUM (shadow_height=2) is treated as roughly one tile tall.
    - Therefore target height in tiles is ``shadow_height / 2``.

    The function measures the sprite's non-transparent alpha silhouette height
    and solves the scale factor that yields the target on-screen height.
    """
    if shadow_height <= 0:
        return 0.0

    alpha = sprite[:, :, 3]
    occupied_rows = np.where(np.any(alpha > 0, axis=1))[0]
    if occupied_rows.size == 0:
        return 1.0

    silhouette_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    if silhouette_height <= 0:
        return 1.0

    sprite_height = int(sprite.shape[0])
    medium_shadow_height = float(CreatureSize.MEDIUM.shadow_height)
    target_height_tiles = float(shadow_height) / medium_shadow_height
    scale = target_height_tiles * float(sprite_height) / float(silhouette_height)
    return max(0.3, min(3.0, scale))


def sprite_content_bbox(
    sprite: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Return the normalized opaque bounding box within the sprite canvas.

    Returns ``(u0, v0, u1, v1)`` in 0..1 with (0, 0) at the canvas top-left, or
    None if the sprite is fully transparent. Sprites are drawn into a square,
    ground-anchored quad that is often mostly transparent padding (a flat boulder
    fills only the bottom sliver). Pointer hit testing uses this box to match the
    sprite's actual silhouette instead of the padded quad.
    """
    alpha = sprite[:, :, 3]
    occupied_rows = np.where(np.any(alpha > 0, axis=1))[0]
    occupied_cols = np.where(np.any(alpha > 0, axis=0))[0]
    if occupied_rows.size == 0 or occupied_cols.size == 0:
        return None

    height, width = alpha.shape
    return (
        float(occupied_cols[0]) / width,
        float(occupied_rows[0]) / height,
        float(occupied_cols[-1] + 1) / width,
        float(occupied_rows[-1] + 1) / height,
    )
