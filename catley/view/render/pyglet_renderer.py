"""
A concrete implementation of the Renderer interface using the Pyglet library.

This module provides the `PygletRenderer`, which handles all low-level drawing
operations by interfacing directly with Pyglet and OpenGL. It translates the
abstract drawing commands from the `RendererBase` interface into specific,
high-performance Pyglet API calls.

Key Implementation Details:
- It manages a Pyglet `Window` and a set of `Batch` objects for efficient drawing.
- It loads the game's tileset into a `pyglet.image.ImageGrid` for fast texture access.
- It uses `pyglet.graphics.OrderedGroup` to ensure correct rendering layers for
  actors, particles, and environmental effects.
- It is the single source of truth for screen dimensions and coordinate-space
  transformations within the Pyglet backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyglet.image
from pyglet.graphics import Batch, Group
from pyglet.image import AbstractImage, ImageData, ImageGrid
from pyglet.shapes import BorderedRectangle, Circle, Rectangle
from pyglet.sprite import Sprite
from pyglet.window import Window

from catley import colors, config
from catley.game.enums import BlendMode
from catley.util.coordinates import Rect, RootConsoleTilePos, TileDimensions
from catley.view.render.effects.particles import ParticleLayer

from .base_renderer import Renderer

if TYPE_CHECKING:
    from catley.util.coordinates import PixelCoord, PixelPos
    from catley.view.ui.cursor_manager import CursorManager

    from .effects.particles import SubTileParticleSystem


class PygletRenderer(Renderer):
    """Low-level graphics primitives and Pyglet/OpenGL operations."""

    def __init__(self, window: Window):
        self.window = window

        # --- Batching and Layering Setup ---
        # A dedicated batch for the static map tiles. Drawn first.
        self.map_batch = Batch()

        # A single batch for all dynamic world objects, with layering
        # controlled by OrderedGroups to ensure correct draw order.
        self.world_objects_batch = Batch()

        # Groups define the Z-order. Lower order numbers are drawn first.
        self.particle_under_group = Group(0)
        self.actor_group = Group(1)
        self.particle_over_group = Group(2)
        self.environmental_effects_group = Group(3)

        # A dedicated batch for all UI elements, drawn last (on top).
        self.ui_batch = Batch()

        # --- Tileset and Sprite Atlas ---
        self.tile_width_px, self.tile_height_px = self._get_tile_pixel_size()
        self.tile_atlas = self._load_tile_atlas()

        # A pre-allocated pool of sprites for rendering the map efficiently.
        self.map_sprites: list[Sprite] = self._initialize_map_sprites()

    def _get_tile_pixel_size(self) -> tuple[int, int]:
        """Calculates the pixel dimensions of a single tile from the tileset image."""
        # This is a temporary way to get tile size before loading.
        # It assumes the tileset image dimensions are a multiple of the columns/rows.
        img = pyglet.image.load(str(config.TILESET_PATH))
        return (img.width // config.TILESET_COLUMNS, img.height // config.TILESET_ROWS)

    def _load_tile_atlas(self) -> ImageGrid:
        """Loads the tileset image and slices it into a grid of textures."""
        tileset_image = pyglet.image.load(str(config.TILESET_PATH))
        return ImageGrid(
            tileset_image,
            rows=config.TILESET_ROWS,
            columns=config.TILESET_COLUMNS,
        )

    def _initialize_map_sprites(self) -> list[Sprite]:
        """Creates the initial pool of Sprite objects for the map."""
        max_width = config.SCREEN_WIDTH
        max_height = config.SCREEN_HEIGHT

        placeholder_texture = self.tile_atlas[0]
        sprites = []
        for y in range(max_height):
            for x in range(max_width):
                sprite = Sprite(
                    img=placeholder_texture,
                    x=x * self.tile_width_px,
                    y=y * self.tile_height_px,
                    batch=self.map_batch,
                )
                sprite.visible = False
                sprites.append(sprite)
        return sprites

    # --- Implementation of RendererBase ABC ---

    @property
    def tile_dimensions(self) -> TileDimensions:
        # In Pyglet, tile dimensions can change if the window is resized.
        # This should eventually be updated based on window size.
        return (self.tile_width_px, self.tile_height_px)

    @property
    def console_width_tiles(self) -> int:
        return config.SCREEN_WIDTH

    @property
    def console_height_tiles(self) -> int:
        return config.SCREEN_HEIGHT

    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Draw an actor by creating a temporary Sprite."""
        char_code = ord(char)
        texture = self.tile_atlas[char_code]

        final_color = (
            int(color[0] * light_intensity[0]),
            int(color[1] * light_intensity[1]),
            int(color[2] * light_intensity[2]),
        )

        # For one-off draws, creating a sprite is fine.
        Sprite(
            img=texture,
            x=screen_x,
            y=screen_y,
            batch=self.world_objects_batch,
            group=self.actor_group,
        ).color = final_color

    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draws the active cursor using data from the cursor manager."""
        # This will be implemented fully in Phase 1.
        # It will involve getting the cursor data, checking for a cached
        # pyglet texture, creating one if it doesn't exist, and then
        # drawing it.
        pass

    def draw_tile_highlight(
        self,
        root_x: int,
        root_y: int,
        color: colors.Color,
        alpha: float,
    ) -> None:
        """Draws a semi-transparent highlight over a single tile."""
        px_x, px_y = self.console_to_screen_coords(root_x, root_y)

        Rectangle(
            x=px_x,
            y=px_y,
            width=self.tile_width_px,
            height=self.tile_height_px,
            color=(*color, int(alpha * 255)),
            batch=self.ui_batch,  # Highlights are UI elements
        ).opacity = int(alpha * 255)

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        """Renders particles using temporary Sprites."""
        # For a more optimized version, you would manage a pool of sprites
        # or use a single VertexList. For the initial port, this is fine.
        group = (
            self.particle_under_group
            if layer == ParticleLayer.UNDER_ACTORS
            else self.particle_over_group
        )

        for i in range(particle_system.active_count):
            if particle_system.layers[i] != layer.value:
                continue

            # This logic will need to be part of the renderer.
            # Simplified for this example.
            px, py = 0, 0  # FIXME: Implement particle_to_screen_coords

            char_code = (
                ord(particle_system.chars[i]) if particle_system.chars[i] else ord(" ")
            )
            sprite = Sprite(
                img=self.tile_atlas[char_code],
                x=px,
                y=py,
                batch=self.world_objects_batch,
                group=group,
            )
            sprite.color = particle_system.colors[i]
            # ... set opacity based on lifetime ...

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        # This would typically be a shader or a pre-rendered circular texture.
        # For a simple start, we can use pyglet.shapes.Circle
        px_x, px_y = self.console_to_screen_coords(*position)
        px_radius = radius * self.tile_width_px

        circle = Circle(
            x=px_x,
            y=px_y,
            radius=px_radius,
            color=tint_color,
            batch=self.world_objects_batch,
            group=self.environmental_effects_group,
        )
        circle.opacity = int(intensity * 255)

    def prepare_to_present(self) -> None:
        """Clears the window in preparation for drawing the batches."""
        self.window.clear()

    def finalize_present(self) -> None:
        """Draws all the batches to the screen."""
        self.map_batch.draw()
        self.world_objects_batch.draw()
        self.ui_batch.draw()

    def update_dimensions(self) -> None:
        """Pyglet handles this automatically on window resize."""
        pass

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        # FIXME:
        # This is a simplified conversion. A real implementation needs to account
        # for letterboxing if the window aspect ratio doesn't match the console
        # aspect ratio. For now, we assume a direct mapping.

        # Pyglet's origin is bottom-left, the existing console's is top-left.
        # We must flip Y.
        screen_y = self.window.height - ((console_y + 1) * self.tile_height_px)
        screen_x = console_x * self.tile_width_px
        return (screen_x, screen_y)

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        # Flip Y-coordinate back to console space (origin top-left)
        flipped_y = self.window.height - pixel_y

        tile_x = int(pixel_x // self.tile_width_px)
        tile_y = int(flipped_y // self.tile_height_px)
        return (tile_x, tile_y)

    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Creates a Pyglet texture from a raw NumPy RGBA pixel array."""
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("Numpy array must be in RGBA format (height, width, 4)")

        # The 'RGBA' format string tells pyglet how to interpret the byte data.
        return ImageData(
            pixels.shape[1],
            pixels.shape[0],
            "RGBA",
            pixels.tobytes(),
            -pixels.shape[1] * 4,
        )

    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Presents a texture, assumed to be a pyglet.image.AbstractImage."""
        if not isinstance(texture, AbstractImage):
            return  # Or raise an error

        px_x, px_y = self.console_to_screen_coords(x_tile, y_tile)
        texture.blit(int(px_x), int(px_y))

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        self.window.set_fullscreen(not self.window.fullscreen)

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draws a raw, unfilled rectangle directly to the screen for debugging."""
        # Note: This draws in UI space, on top of everything.
        BorderedRectangle(
            x=px_x,
            y=px_y,
            width=px_w,
            height=px_h,
            border=1,
            color=(0, 0, 0, 0),  # Transparent fill
            border_color=color,
            batch=self.ui_batch,
        )
