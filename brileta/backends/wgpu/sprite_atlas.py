"""Dynamic sprite atlas for procedurally generated actor sprites.

Manages a single GPU texture that holds variable-size RGBA sprites packed
via a skyline bin-packing algorithm.  Sprites are uploaded at map load time,
not per-frame.  Each allocation returns UV coordinates that the actor
renderer uses to draw from this atlas instead of the CP437 glyph atlas.

The atlas is auto-sized at map load time to fit all sprites for the current
map.  On map transitions, the old atlas is dropped and a fresh one is
created at the appropriate size for the new map.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from brileta.types import SpriteUV

if TYPE_CHECKING:
    import wgpu

    from .resource_manager import WGPUResourceManager

logger = logging.getLogger(__name__)

# Minimum atlas dimension (no point going smaller than this).
_MIN_ATLAS_SIZE = 256

# 1 pixel of padding between packed sprites prevents texture-filtering bleed.
PADDING = 1

# Packing overhead factor.  Skyline bin-packing wastes some space in the gaps
# between variable-height sprites.  1.3 means we budget 30% more area than the
# raw sum of sprite areas, which is conservative enough to avoid a too-small
# atlas while keeping VRAM usage reasonable.
_PACKING_OVERHEAD = 1.3


def compute_atlas_size(
    sprites: list[np.ndarray],
    gpu_max_texture_dim: int,
    extra_area: int = 0,
) -> int:
    """Compute the smallest power-of-two atlas side that fits all sprites.

    Estimates the total padded area, adds packing overhead, and rounds up to
    the next power of two.  The result is clamped between ``_MIN_ATLAS_SIZE``
    and ``gpu_max_texture_dim`` (the hardware limit queried from the GPU).

    ``extra_area`` reserves headroom (in padded px^2) beyond the given sprites
    so late-added actors can pack into the live atlas without a rebuild. It is
    counted toward the area budget only, not the per-sprite max-dimension
    constraint.

    Returns a single int because the atlas is always square.
    """
    if not sprites:
        return _MIN_ATLAS_SIZE

    # Sum padded pixel areas and track the widest/tallest sprite (the atlas
    # must be at least that large on each axis or the sprite can never fit).
    total_area = 0
    max_dim = 0
    for px in sprites:
        h, w = px.shape[:2]
        padded_w = w + PADDING
        padded_h = h + PADDING
        total_area += padded_w * padded_h
        max_dim = max(max_dim, padded_w, padded_h)

    total_area += extra_area
    target_area = total_area * _PACKING_OVERHEAD
    # Side length from area, then round up to next power of two.
    side = max(max_dim, math.ceil(math.sqrt(target_area)))
    side = _next_power_of_two(side)
    side = max(_MIN_ATLAS_SIZE, min(side, gpu_max_texture_dim))

    if side * side < total_area:
        logger.warning(
            "Sprite atlas clamped to GPU max (%dx%d). %d sprites totalling "
            "%d px^2 may not all fit - some actors will use glyph fallback.",
            side,
            side,
            len(sprites),
            total_area,
        )

    return side


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


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

    Preferred usage (batched - one GPU upload for the whole atlas)::

        sprites = [generate_sprite(...) for actor in actors]
        size = compute_atlas_size(sprites, gpu_max)
        atlas = SpriteAtlas(resource_manager, size, size)
        for sprite in sprites:
            uv = atlas.pack(sprite)     # CPU-only skyline placement
        atlas.flush()                   # single GPU upload

    Legacy usage (per-sprite GPU upload, slower for many sprites)::

        uv = atlas.allocate(sprite)     # skyline + immediate GPU upload

    The GPU texture is created lazily on the first ``allocate()`` or
    ``flush()`` call so that constructing a SpriteAtlas is cheap.
    """

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        width: int,
        height: int,
    ) -> None:
        self._resource_manager = resource_manager
        self.width = width
        self.height = height

        # GPU texture - created lazily on first allocate() or flush().
        self._texture: wgpu.GPUTexture | None = None

        # CPU-side staging buffer for batched pack()+flush() workflow.
        # Allocated lazily on first pack() call, freed after flush().
        self._cpu_buffer: np.ndarray | None = None

        # Skyline: list of segments sorted left-to-right, covering [0, width).
        self._skyline: list[_SkylineNode] = [_SkylineNode(x=0, y=0, width=width)]

        # Track allocated regions keyed by the SpriteUV returned to the caller.
        self._regions: dict[SpriteUV, _SpriteRegion] = {}

        # Bulk pack count (set by pack_all, not tracked per-region).
        self._bulk_count: int = 0

        # Shelf cursor left by pack_all so pack_incremental can continue packing
        # late sprites into the free space below the last batch row.
        self._shelf_x: int = 0
        self._shelf_y: int = 0
        self._shelf_h: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def texture(self) -> wgpu.GPUTexture | None:
        """The underlying GPU texture, or None if no sprites have been uploaded."""
        return self._texture

    def pack(self, pixels: np.ndarray) -> SpriteUV | None:
        """Place a sprite in the atlas (CPU only, no GPU upload).

        Performs skyline bin-packing and blits the sprite into a CPU-side
        staging buffer.  Call :meth:`flush` after packing all sprites to
        upload the entire atlas to the GPU in a single ``write_texture``.

        Args:
            pixels: uint8 numpy array with shape (height, width, 4).

        Returns:
            A SpriteUV with the allocated UV coordinates, or None if the
            atlas is full.
        """
        uv, best_x, best_y = self._place(pixels)
        if uv is None:
            return None

        sprite_h, sprite_w = pixels.shape[:2]

        # Lazily create the CPU staging buffer (zeros = transparent black).
        if self._cpu_buffer is None:
            self._cpu_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Blit sprite pixels into the staging buffer.
        self._cpu_buffer[best_y : best_y + sprite_h, best_x : best_x + sprite_w] = (
            pixels
        )

        # Duplicate the bottom row into the 1px padding below the sprite.
        # The shadow render pass samples sprite silhouettes with a linear-
        # filtering sampler that approaches the bottom UV boundary. Without
        # edge extension, it blends opaque pixels with transparent padding,
        # creating a visible seam.
        if sprite_h > 0 and best_y + sprite_h < self.height:
            self._cpu_buffer[best_y + sprite_h, best_x : best_x + sprite_w] = pixels[
                sprite_h - 1
            ]

        return uv

    def pack_all(self, sprites: list[np.ndarray]) -> list[SpriteUV | None]:
        """Pack all sprites at once using fast shelf packing.

        Sorts sprites by height descending and packs them left-to-right in
        fixed-height rows (shelves).  This is O(1) per sprite - pure integer
        arithmetic, no list manipulation - and much faster than per-sprite
        skyline packing for large batches.

        Returns UVs in the **same order** as the input list.  Sprites that
        don't fit return None in their slot.

        Also builds the CPU staging buffer.  Call :meth:`flush` afterwards
        to upload to the GPU.
        """
        if not sprites:
            return []

        atlas_h = self.height
        atlas_w = self.width

        # A fresh batch build owns the whole atlas: reset the shelf cursor so
        # placement starts at the origin (and so a later pack_incremental resumes
        # from wherever this batch leaves off).
        self._shelf_x = self._shelf_y = self._shelf_h = 0

        # Allocate the CPU staging buffer.
        buf = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)

        results: list[SpriteUV | None] = [None] * len(sprites)
        placed = 0
        for i in self._tallest_first(sprites):
            px = sprites[i]
            sp_h, sp_w = px.shape[0], px.shape[1]
            pos = self._shelf_place(sp_w + PADDING, sp_h + PADDING)
            if pos is None:
                continue  # Atlas full for this sprite.
            x, y = pos

            # Blit sprite into CPU buffer.
            buf[y : y + sp_h, x : x + sp_w] = px
            # Bottom-row edge extension for shadow linear filtering.
            if sp_h > 0 and y + sp_h < atlas_h:
                buf[y + sp_h, x : x + sp_w] = px[sp_h - 1]

            results[i] = self._uv_for(x, y, sp_w, sp_h)
            placed += 1

        self._cpu_buffer = buf
        # Update region count to match what we placed (for allocated_count).
        # We skip individual _regions tracking for pack_all since the UVs are
        # returned directly and deallocate() is not used in the bulk path.
        self._bulk_count = placed
        return results

    def pack_incremental(self, sprites: list[np.ndarray]) -> list[SpriteUV | None]:
        """Pack late-added sprites into the atlas and upload each region now.

        Continues the shelf cursor left by :meth:`pack_all` and uploads each
        sprite with an immediate partial ``write_texture`` - the batch
        :meth:`flush` already ran, so there is no CPU staging buffer to reuse.
        This lets an actor created after the batch build join the live atlas
        without re-flushing the whole texture.

        Returns UVs in the **same order** as the input list; a sprite that no
        longer fits returns None in its slot (the caller treats a partial pack
        as a full atlas and rebuilds).
        """
        if not sprites:
            return []

        # The batch build flushes and drops the CPU buffer but keeps the
        # texture. If this atlas was never batch-built, create a cleared texture
        # so unoccupied regions read as transparent under the partial uploads.
        if self._texture is None:
            self._texture = self._create_texture()

        results: list[SpriteUV | None] = [None] * len(sprites)
        for i in self._tallest_first(sprites):
            px = sprites[i]
            sp_h, sp_w = px.shape[0], px.shape[1]
            pos = self._shelf_place(sp_w + PADDING, sp_h + PADDING)
            if pos is None:
                continue  # Atlas full for this sprite; slot stays None.
            x, y = pos

            self._upload_region(x, y, sp_w, sp_h, px)
            results[i] = self._uv_for(x, y, sp_w, sp_h)
            self._bulk_count += 1

        return results

    @staticmethod
    def _tallest_first(sprites: list[np.ndarray]) -> list[int]:
        """Input indices ordered by sprite height descending.

        Shelf packing sizes each shelf by its first sprite, so feeding sprites
        tallest-first makes every later sprite on a shelf no taller than it -
        which keeps :meth:`_shelf_place`'s height guard from tripping mid-shelf.
        """
        return sorted(
            range(len(sprites)), key=lambda i: sprites[i].shape[0], reverse=True
        )

    def _shelf_place(self, padded_w: int, padded_h: int) -> tuple[int, int] | None:
        """Advance the shelf cursor for one sprite; return its (x, y) or None.

        The single placement primitive shared by :meth:`pack_all` and
        :meth:`pack_incremental`. It mutates the persistent ``_shelf_*`` cursor,
        so pack_all resets it first (fresh atlas) and pack_incremental resumes it
        to append below the last batch row. Callers must feed sprites
        tallest-first (see :meth:`_tallest_first`). Returns None when the sprite
        no longer fits vertically (atlas full).
        """
        # Start a new shelf when the sprite would overflow the current row's
        # width, or is taller than the current shelf (its band is not reserved
        # for us, so placing it would overlap the row above). The height guard is
        # a no-op for tallest-first input except at a genuine shelf start.
        if (
            self._shelf_h == 0
            or self._shelf_x + padded_w > self.width
            or padded_h > self._shelf_h
        ):
            self._shelf_y += self._shelf_h
            self._shelf_x = 0
            self._shelf_h = padded_h

        if self._shelf_y + padded_h > self.height:
            return None  # Out of vertical space.

        x = self._shelf_x
        y = self._shelf_y
        self._shelf_x += padded_w
        return x, y

    def _uv_for(self, x: int, y: int, sp_w: int, sp_h: int) -> SpriteUV:
        """Normalized UV rectangle for a sprite placed at pixel (x, y).

        Multiplies by the precomputed reciprocal (rather than dividing) so the
        UVs stay bit-identical to the pre-refactor pack_all output.
        """
        inv_w = 1.0 / self.width
        inv_h = 1.0 / self.height
        return SpriteUV(
            u1=x * inv_w,
            v1=y * inv_h,
            u2=(x + sp_w) * inv_w,
            v2=(y + sp_h) * inv_h,
        )

    def flush(self) -> None:
        """Upload the CPU staging buffer to the GPU in a single transfer.

        Creates the GPU texture if it doesn't exist, uploads the entire
        staging buffer, then frees the CPU buffer.  After this call,
        :attr:`texture` is ready for rendering.

        Does nothing if no sprites have been packed via :meth:`pack`.
        """
        if self._cpu_buffer is None:
            return

        if self._texture is None:
            self._texture = self._create_texture_bare()

        data = np.ascontiguousarray(self._cpu_buffer)
        self._resource_manager.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            memoryview(data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": self.width * 4,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )

        # Free the CPU staging buffer now that it's on the GPU.
        self._cpu_buffer = None

    def allocate(self, pixels: np.ndarray) -> SpriteUV | None:
        """Pack an RGBA sprite and upload it to the GPU immediately.

        This is the legacy per-sprite path.  Prefer :meth:`pack` +
        :meth:`flush` for bulk loading - it avoids thousands of
        individual ``write_texture`` calls.

        Args:
            pixels: uint8 numpy array with shape (height, width, 4).

        Returns:
            A SpriteUV with the allocated UV coordinates, or None if the
            atlas is full.
        """
        uv, best_x, best_y = self._place(pixels)
        if uv is None:
            return None

        sprite_h, sprite_w = pixels.shape[:2]

        # Ensure the GPU texture exists.
        if self._texture is None:
            self._texture = self._create_texture()

        # Upload sprite pixels to the atlas region.
        self._upload_region(best_x, best_y, sprite_w, sprite_h, pixels)

        return uv

    def _place(self, pixels: np.ndarray) -> tuple[SpriteUV | None, int, int]:
        """Validate, find a skyline position, and record the region.

        Shared by both :meth:`pack` and :meth:`allocate`.  Returns
        ``(uv, x, y)`` on success or ``(None, 0, 0)`` if the atlas is full.
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

        padded_w = sprite_w + PADDING
        padded_h = sprite_h + PADDING

        best_idx, best_y = self._find_best_position(padded_w, padded_h)
        if best_idx is None or best_y is None:
            return None, 0, 0

        best_x = self._skyline[best_idx].x
        self._insert_skyline_node(best_idx, best_x, best_y + padded_h, padded_w)

        u1 = best_x / self.width
        v1 = best_y / self.height
        u2 = (best_x + sprite_w) / self.width
        v2 = (best_y + sprite_h) / self.height
        uv = SpriteUV(u1=u1, v1=v1, u2=u2, v2=v2)

        self._regions[uv] = _SpriteRegion(
            x=best_x, y=best_y, width=padded_w, height=padded_h
        )
        return uv, best_x, best_y

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
        return len(self._regions) + self._bulk_count

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

    def _create_texture_bare(self) -> wgpu.GPUTexture:
        """Create the atlas GPU texture without clearing it.

        Used by :meth:`flush`, which writes the entire buffer in one call
        so a separate clear would be wasted work.
        """
        import wgpu as wgpu_mod

        return self._resource_manager.device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu_mod.TextureFormat.rgba8unorm,
            usage=wgpu_mod.TextureUsage.TEXTURE_BINDING
            | wgpu_mod.TextureUsage.COPY_DST,
            label="sprite_atlas",
        )

    def _create_texture(self) -> wgpu.GPUTexture:
        """Create the atlas GPU texture and clear it to transparent black.

        Used by the per-sprite :meth:`allocate` path, where unoccupied
        regions need to be pre-cleared so partial uploads don't leave
        visible garbage.
        """
        texture = self._create_texture_bare()

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
        """Upload pixel data to a sub-region of the atlas texture.

        Also duplicates the bottom row into the 1px padding below the
        sprite.  The shadow render pass samples sprite silhouettes with a
        linear-filtering sampler.  Without edge extension, linear filtering
        at the bottom UV boundary blends opaque sprite pixels with the
        transparent atlas padding, creating a semi-transparent base edge
        that shows terrain through as a visible seam.  Duplicating the
        bottom row into the padding gives the filter the correct color to
        blend with.
        """
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

        # Extend bottom row into the PADDING region below the sprite.
        # Only the bottom edge needs this treatment: the shadow render pass
        # samples sprite silhouettes with a linear-filtering sampler that
        # approaches the bottom UV boundary, not the left, right, or top
        # edges.  If a future render pass samples those edges under linear
        # filtering, similar extensions will be needed on the other sides.
        if height > 0 and y + height < self.height:
            bottom_row = np.ascontiguousarray(
                pixels[height - 1 : height, :, :], dtype=np.uint8
            )
            self._resource_manager.queue.write_texture(
                {
                    "texture": self._texture,
                    "mip_level": 0,
                    "origin": (x, y + height, 0),
                },
                memoryview(bottom_row.tobytes()),
                {"offset": 0, "bytes_per_row": width * 4, "rows_per_image": 1},
                (width, 1, 1),
            )
