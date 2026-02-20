"""Dynamic sprite atlas for procedurally generated actor sprites.

Manages a single GPU texture that holds variable-size RGBA sprites packed
via a skyline bin-packing algorithm.  Sprites are uploaded at map load time,
not per-frame.  Each allocation returns UV coordinates that the actor
renderer uses to draw from this atlas instead of the CP437 glyph atlas.

The atlas exposes deallocate() and clear() as escape hatches for future
multi-map transitions - not as part of a per-frame paging system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from brileta.types import SpriteUV

if TYPE_CHECKING:
    import wgpu

    from .resource_manager import WGPUResourceManager

# Default atlas dimensions.  A 2048x2048 RGBA texture uses 16 MB of VRAM
# and holds roughly 10,000 20x20 sprites - enough for ~40 maps at current
# tree density.
ATLAS_WIDTH = 2048
ATLAS_HEIGHT = 2048

# 1 pixel of padding between packed sprites prevents texture-filtering bleed.
PADDING = 1


@dataclass
class _SkylineNode:
    """One segment of the skyline.  The skyline is a monotonic staircase of
    (x, y, width) segments that tracks the topmost occupied row at each
    horizontal span.
    """

    x: int
    y: int
    width: int


@dataclass
class _SpriteRegion:
    """Bookkeeping for one allocated sprite inside the atlas."""

    x: int
    y: int
    width: int
    height: int


class SpriteAtlas:
    """Dynamic texture atlas packed via skyline bin-packing.

    Usage::

        atlas = SpriteAtlas(resource_manager)
        uv = atlas.allocate(my_rgba_array)
        # ... later ...
        atlas.deallocate(uv)
        atlas.clear()

    The GPU texture is created lazily on the first allocation so that
    constructing a SpriteAtlas is cheap (no GPU work until needed).
    """

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        width: int = ATLAS_WIDTH,
        height: int = ATLAS_HEIGHT,
    ) -> None:
        self._resource_manager = resource_manager
        self.width = width
        self.height = height

        # GPU texture - created lazily on first allocate().
        self._texture: wgpu.GPUTexture | None = None

        # Skyline: list of segments sorted left-to-right, covering [0, width).
        self._skyline: list[_SkylineNode] = [_SkylineNode(x=0, y=0, width=width)]

        # Track allocated regions keyed by the SpriteUV returned to the caller.
        self._regions: dict[SpriteUV, _SpriteRegion] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def texture(self) -> wgpu.GPUTexture | None:
        """The underlying GPU texture, or None if no sprites have been allocated."""
        return self._texture

    def allocate(self, pixels: np.ndarray) -> SpriteUV | None:
        """Pack an RGBA sprite into the atlas and upload it to the GPU.

        Args:
            pixels: uint8 numpy array with shape (height, width, 4).

        Returns:
            A SpriteUV with the allocated UV coordinates, or None if the
            atlas is full.
        """
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError(
                f"Expected RGBA array with shape (H, W, 4), got {pixels.shape}"
            )

        sprite_h, sprite_w = pixels.shape[:2]
        if sprite_w <= 0 or sprite_h <= 0:
            raise ValueError(
                f"Sprite dimensions must be positive, got {sprite_w}x{sprite_h}"
            )

        # Padded dimensions for packing (padding prevents filtering bleed).
        padded_w = sprite_w + PADDING
        padded_h = sprite_h + PADDING

        # Find the best skyline position for this sprite.
        best_idx, best_y = self._find_best_position(padded_w, padded_h)
        if best_idx is None or best_y is None:
            return None  # Atlas is full.

        # Compute the x position from the chosen skyline node.
        best_x = self._skyline[best_idx].x

        # Ensure the GPU texture exists.
        if self._texture is None:
            self._texture = self._create_texture()

        # Upload sprite pixels to the atlas region.
        self._upload_region(best_x, best_y, sprite_w, sprite_h, pixels)

        # Update the skyline to reflect the new allocation.
        self._insert_skyline_node(best_idx, best_x, best_y + padded_h, padded_w)

        # Compute normalized UV coordinates.
        u1 = best_x / self.width
        v1 = best_y / self.height
        u2 = (best_x + sprite_w) / self.width
        v2 = (best_y + sprite_h) / self.height
        uv = SpriteUV(u1=u1, v1=v1, u2=u2, v2=v2)

        self._regions[uv] = _SpriteRegion(
            x=best_x, y=best_y, width=padded_w, height=padded_h
        )
        return uv

    def deallocate(self, uv: SpriteUV) -> None:
        """Free a previously allocated sprite region.

        The freed space becomes available for future skyline packing but may
        not be perfectly reclaimed without a full clear().  This is an escape
        hatch for multi-map transitions, not a per-frame operation.

        Warning: may cause overlapping allocations if other sprites were packed
        above the freed region.  Use clear() for a guaranteed-safe full reset.
        """
        region = self._regions.pop(uv, None)
        if region is None:
            return

        # Insert a skyline node at the freed region's base y so subsequent
        # allocations can reuse the vertical space above it.  This is a
        # best-effort reclamation - fragmentation is possible but acceptable
        # given the expected usage pattern (clear() between maps).
        self._insert_skyline_node_at(region.x, region.y, region.width)

    def clear(self) -> None:
        """Reset the atlas entirely, freeing all sprite regions.

        Call this on map transitions to reclaim all atlas space.  The GPU
        texture is kept alive to avoid reallocation.
        """
        self._skyline = [_SkylineNode(x=0, y=0, width=self.width)]
        self._regions.clear()

    @property
    def allocated_count(self) -> int:
        """Number of sprites currently in the atlas."""
        return len(self._regions)

    # ------------------------------------------------------------------
    # Skyline bin-packing internals
    # ------------------------------------------------------------------

    def _find_best_position(
        self, padded_w: int, padded_h: int
    ) -> tuple[int | None, int | None]:
        """Find the skyline index and y-coordinate that wastes the least space.

        Uses the "bottom-left" heuristic: among all positions where the sprite
        fits, choose the one with the lowest resulting y (and leftmost on ties).
        """
        best_idx: int | None = None
        best_y: int | None = None

        for i in range(len(self._skyline)):
            y = self._check_fit(i, padded_w, padded_h)
            if y is not None and (best_y is None or y < best_y):
                best_y = y
                best_idx = i

        return best_idx, best_y

    def _check_fit(self, index: int, padded_w: int, padded_h: int) -> int | None:
        """Check whether a sprite fits starting at skyline node *index*.

        The sprite spans horizontally from the node's x to x + padded_w.  Its
        y position is the maximum y of all skyline nodes it overlaps with.

        Returns the required y, or None if the sprite doesn't fit.
        """
        x = self._skyline[index].x
        if x + padded_w > self.width:
            return None

        remaining_width = padded_w
        y = 0
        i = index

        while remaining_width > 0:
            if i >= len(self._skyline):
                return None
            node = self._skyline[i]
            y = max(y, node.y)
            if y + padded_h > self.height:
                return None
            remaining_width -= node.width
            i += 1

        return y

    def _insert_skyline_node(self, index: int, x: int, y: int, padded_w: int) -> None:
        """Insert a new skyline node and merge/trim overlapping segments."""
        new_node = _SkylineNode(x=x, y=y, width=padded_w)
        self._skyline.insert(index, new_node)

        # Trim or remove nodes that are now covered by the new node.
        i = index + 1
        right_edge = x + padded_w
        while i < len(self._skyline):
            node = self._skyline[i]
            node_right = node.x + node.width
            if node.x >= right_edge:
                break
            if node_right > right_edge:
                # Partially covered - shrink it.
                self._skyline[i] = _SkylineNode(
                    x=right_edge, y=node.y, width=node_right - right_edge
                )
                break
            # Fully covered - remove it.
            self._skyline.pop(i)

        self._merge_skyline()

    def _insert_skyline_node_at(self, x: int, y: int, width: int) -> None:
        """Insert a freed region back into the skyline for potential reuse."""
        # Find where this region falls in the skyline.
        insert_idx = 0
        for i, node in enumerate(self._skyline):
            if node.x + node.width > x:
                insert_idx = i
                break
        else:
            insert_idx = len(self._skyline)

        new_node = _SkylineNode(x=x, y=y, width=width)
        self._skyline.insert(insert_idx, new_node)

        # Trim overlapping nodes.
        right_edge = x + width
        i = insert_idx + 1
        while i < len(self._skyline):
            node = self._skyline[i]
            node_right = node.x + node.width
            if node.x >= right_edge:
                break
            if node_right > right_edge:
                self._skyline[i] = _SkylineNode(
                    x=right_edge, y=node.y, width=node_right - right_edge
                )
                break
            self._skyline.pop(i)

        self._merge_skyline()

    def _merge_skyline(self) -> None:
        """Merge adjacent skyline nodes that share the same y value."""
        i = 0
        while i < len(self._skyline) - 1:
            current = self._skyline[i]
            next_node = self._skyline[i + 1]
            if current.y == next_node.y:
                self._skyline[i] = _SkylineNode(
                    x=current.x,
                    y=current.y,
                    width=current.width + next_node.width,
                )
                self._skyline.pop(i + 1)
            else:
                i += 1

    # ------------------------------------------------------------------
    # GPU upload helpers
    # ------------------------------------------------------------------

    def _create_texture(self) -> wgpu.GPUTexture:
        """Create the atlas GPU texture (RGBA8, 2048x2048 by default)."""
        import wgpu as wgpu_mod

        texture = self._resource_manager.device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu_mod.TextureFormat.rgba8unorm,
            usage=wgpu_mod.TextureUsage.TEXTURE_BINDING
            | wgpu_mod.TextureUsage.COPY_DST,
            label="sprite_atlas",
        )

        # Clear the texture to fully transparent black so unoccupied regions
        # don't bleed visible garbage into neighboring sprites.
        clear_data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._resource_manager.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            memoryview(clear_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": self.width * 4,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )
        return texture

    def _upload_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        pixels: np.ndarray,
    ) -> None:
        """Upload pixel data to a sub-region of the atlas texture."""
        assert self._texture is not None
        # Ensure contiguous C-order layout for the GPU upload.
        contiguous = np.ascontiguousarray(pixels, dtype=np.uint8)
        self._resource_manager.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (x, y, 0)},
            memoryview(contiguous.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": width * 4,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
