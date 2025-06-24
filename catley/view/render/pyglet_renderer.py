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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

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


class HasVisible(Protocol):
    @property
    def visible(self) -> bool: ...

    @visible.setter
    def visible(self, __value: bool, /) -> None: ...

    # The '/' tells the type checker:
    # "This parameter must be passed by position, not by keyword."
    # This makes the parameter's name irrelevant for compatibility checks.


class PygletObjectPool[T: HasVisible]:
    """Generic object pool for reusing UI objects."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self.factory = factory
        self.pooled_objects: list[T] = []
        self.active_objects: list[T] = []

    def get_or_create(self) -> T:
        """Get object from pool or create new one."""
        obj = self.pooled_objects.pop() if self.pooled_objects else self.factory()
        obj.visible = True  # Ensure the object is visible when reused
        self.active_objects.append(obj)
        return obj

    def return_all_to_pool(self) -> None:
        """Return all active objects to pool."""
        for obj in self.active_objects:
            obj.visible = False
            self.pooled_objects.append(obj)
        self.active_objects.clear()


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

        self.actor_sprite_pool: PygletObjectPool[Sprite] = PygletObjectPool(
            lambda: Sprite(
                img=self.tile_atlas[ord(" ")],
                batch=self.world_objects_batch,
                group=self.actor_group,
            )
        )

        self.highlight_pool: PygletObjectPool[Rectangle] = PygletObjectPool(
            lambda: Rectangle(0, 0, 0, 0, batch=self.ui_batch)
        )

        self.particle_under_pool: PygletObjectPool[Sprite] = PygletObjectPool(
            lambda: Sprite(
                img=self.tile_atlas[ord(" ")],
                batch=self.world_objects_batch,
                group=self.particle_under_group,
            )
        )

        self.particle_over_pool: PygletObjectPool[Sprite] = PygletObjectPool(
            lambda: Sprite(
                img=self.tile_atlas[ord(" ")],
                batch=self.world_objects_batch,
                group=self.particle_over_group,
            )
        )

        self.environmental_effect_pool: PygletObjectPool[Circle] = PygletObjectPool(
            lambda: Circle(
                0,
                0,
                1,
                batch=self.world_objects_batch,
                group=self.environmental_effects_group,
            )
        )

    def _get_tile_pixel_size(self) -> tuple[int, int]:
        """Calculates the pixel dimensions of a single tile from the tileset image."""
        # This is a temporary way to get tile size before loading.
        # It assumes the tileset image dimensions are a multiple of the columns/rows.
        img = pyglet.image.load(str(config.TILESET_PATH))
        return (img.width // config.TILESET_COLUMNS, img.height // config.TILESET_ROWS)

    def _load_tile_atlas(self) -> ImageGrid:
        """Loads the tileset, makes magenta pixels transparent, and slices it."""
        # Load the original image, and get the raw pixel data
        tileset_image = pyglet.image.load(str(config.TILESET_PATH))
        raw_data = tileset_image.get_image_data()
        pixels = raw_data.get_data("RGBA", raw_data.width * 4)

        # Convert to a mutable NumPy array to manipulate it
        pixel_array = (
            np.frombuffer(pixels, dtype=np.uint8)
            .reshape((raw_data.height, raw_data.width, 4))
            .copy()
        )

        # Find all magenta pixels (R=255, G=0, B=255)
        magenta_mask = (
            (pixel_array[:, :, 0] == 255)
            & (pixel_array[:, :, 1] == 0)
            & (pixel_array[:, :, 2] == 255)
        )

        # Set the alpha channel of these pixels to 0 (fully transparent)
        pixel_array[magenta_mask, 3] = 0

        # Create a new, transparency-fixed ImageData object from our modified array
        fixed_image_data = ImageData(
            raw_data.width,
            raw_data.height,
            "RGBA",
            pixel_array.tobytes(),
            raw_data.width * -4,
        )

        # Create the ImageGrid from the corrected image data
        return ImageGrid(
            fixed_image_data,
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
        sprite = self.actor_sprite_pool.get_or_create()

        char_code = ord(char)
        sprite.image = self.tile_atlas[char_code]
        sprite.x = screen_x
        sprite.y = screen_y

        final_color = (
            int(color[0] * light_intensity[0]),
            int(color[1] * light_intensity[1]),
            int(color[2] * light_intensity[2]),
        )
        sprite.color = final_color

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

        highlight = self.highlight_pool.get_or_create()
        highlight.x = px_x
        highlight.y = px_y
        highlight.width = self.tile_width_px
        highlight.height = self.tile_height_px
        highlight.color = (*color, int(alpha * 255))
        highlight.opacity = int(alpha * 255)

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        # Choose the right pool based on layer
        particle_pool = (
            self.particle_under_pool
            if layer == ParticleLayer.UNDER_ACTORS
            else self.particle_over_pool
        )

        for i in range(particle_system.active_count):
            if particle_system.layers[i] != layer.value:
                continue
            coords = particle_system._convert_particle_to_screen_coords(
                i, viewport_bounds, view_offset, self
            )
            if coords is None:
                continue

            px, py = coords
            char_code = (
                ord(particle_system.chars[i]) if particle_system.chars[i] else ord(" ")
            )

            sprite = particle_pool.get_or_create()
            sprite.image = self.tile_atlas[char_code]
            sprite.x = px
            sprite.y = py
            sprite.color = tuple(particle_system.colors[i])

            lifetime_ratio = (
                particle_system.lifetimes[i] / particle_system.max_lifetimes[i]
            )
            sprite.opacity = int(255 * lifetime_ratio)

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

        circle = self.environmental_effect_pool.get_or_create()

        # Update its properties
        circle.x = px_x
        circle.y = px_y
        circle.radius = px_radius
        circle.color = tint_color
        circle.opacity = int(intensity * 255)

    def prepare_to_present(self) -> None:
        """Clears the window in preparation for drawing the batches."""
        self.window.clear()

        self.actor_sprite_pool.return_all_to_pool()
        self.highlight_pool.return_all_to_pool()
        self.particle_under_pool.return_all_to_pool()
        self.particle_over_pool.return_all_to_pool()
        self.environmental_effect_pool.return_all_to_pool()

    def finalize_present(self) -> None:
        """Draws all the batches to the screen."""
        self.map_batch.draw()
        self.world_objects_batch.draw()
        self.ui_batch.draw()

    def update_dimensions(self) -> None:
        """Pyglet handles this automatically on window resize."""
        pass

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """
        Convert console coordinates to screen coordinates with proper
        letterboxing/pillarboxing. This matches the logic from TCODRenderer to
        ensure consistent coordinate mapping.
        """
        window_width = self.window.width
        window_height = self.window.height
        console_width = self.console_width_tiles
        console_height = self.console_height_tiles

        console_aspect = console_width / console_height
        window_aspect = window_width / window_height

        if console_aspect > window_aspect:
            # Console is letterboxed vertically (black bars on top/bottom)
            scaled_height = int(window_width / console_aspect)
            offset_y = (window_height - scaled_height) // 2
            screen_x = console_x * (window_width / console_width)
            # Pyglet uses bottom-left origin, so we need to account for that
            screen_y = (
                window_height
                - offset_y
                - (console_y + 1) * (scaled_height / console_height)
            )
        else:
            # Console is pillarboxed horizontally (black bars on left/right)
            scaled_width = int(window_height * console_aspect)
            offset_x = (window_width - scaled_width) // 2
            screen_x = offset_x + console_x * (scaled_width / console_width)
            # Pyglet uses bottom-left origin, so we need to flip Y
            screen_y = window_height - (console_y + 1) * (
                window_height / console_height
            )

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
