"""Atlas texture loading and management for the WGPU backend.

Loads the tileset PNG, derives outlined and blurred variants, and exposes
the resulting GPU textures. Owned by WGPUGraphicsContext.
"""

from __future__ import annotations

import numpy as np
import wgpu
from PIL import Image as PILImage
from PIL import ImageFilter

from brileta import config
from brileta.util.tilesets import derive_outlined_atlas

from .resource_manager import WGPUResourceManager


class WGPUAtlasManager:
    """Loads and owns the tileset atlas textures.

    Creates three GPU textures from the tileset PNG:
    - atlas_texture: main tileset with magenta transparency keyed out
    - outlined_atlas_texture: white 1-pixel outlines of each glyph (for shimmer effects)
    - blurred_atlas_texture: Gaussian-blurred copy (for soft actor shadows)
    """

    def __init__(self, resource_manager: WGPUResourceManager) -> None:
        self._resource_manager = resource_manager

        # Load the main atlas and keep raw pixels for derived textures
        pixels = self._load_atlas_pixels()
        self.atlas_texture: wgpu.GPUTexture = self._create_atlas_texture(pixels)
        self.outlined_atlas_texture: wgpu.GPUTexture | None = (
            self._create_outlined_texture(pixels)
        )
        self.blurred_atlas_texture: wgpu.GPUTexture = self._create_blurred_texture(
            pixels
        )

    def _load_atlas_pixels(self) -> np.ndarray:
        """Load tileset PNG and apply magenta transparency keying."""
        img = PILImage.open(str(config.TILESET_PATH)).convert("RGBA")
        pixels = np.array(img, dtype="u1")

        # Key out magenta (255, 0, 255) as fully transparent
        magenta_mask = (
            (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 255)
        )
        pixels[magenta_mask, 3] = 0
        return pixels

    def _create_atlas_texture(self, pixels: np.ndarray) -> wgpu.GPUTexture:
        """Create the main tileset GPU texture."""
        height, width = pixels.shape[:2]
        return self._resource_manager.create_atlas_texture(
            width=width,
            height=height,
            data=pixels.tobytes(),
            texture_format="rgba8unorm",
        )

    def _create_outlined_texture(self, pixels: np.ndarray) -> wgpu.GPUTexture | None:
        """Create a white-outlined variant of each glyph for tinting at render time."""
        height, width = pixels.shape[:2]
        tile_width = width // config.TILESET_COLUMNS
        tile_height = height // config.TILESET_ROWS

        outlined_pixels = derive_outlined_atlas(
            pixels,
            tile_width,
            tile_height,
            config.TILESET_COLUMNS,
            config.TILESET_ROWS,
            color=(255, 255, 255, 255),
        )

        return self._resource_manager.create_atlas_texture(
            width=width,
            height=height,
            data=outlined_pixels.tobytes(),
            texture_format="rgba8unorm",
        )

    def _create_blurred_texture(self, pixels: np.ndarray) -> wgpu.GPUTexture:
        """Create a Gaussian-blurred copy of the atlas for soft actor shadows."""
        height, width = pixels.shape[:2]
        atlas_image = PILImage.fromarray(pixels, mode="RGBA")
        blurred_image = atlas_image.filter(
            ImageFilter.GaussianBlur(radius=config.ACTOR_SHADOW_BLUR_RADIUS)
        )
        blurred_pixels = np.array(blurred_image, dtype="u1")

        return self._resource_manager.create_atlas_texture(
            width=width,
            height=height,
            data=blurred_pixels.tobytes(),
            texture_format="rgba8unorm",
        )
