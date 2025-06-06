from __future__ import annotations

import tcod.render
import tcod.sdl.render
from tcod.console import Console

from catley import colors
from catley.util.coordinates import CoordinateConverter


class Renderer:
    """Low-level graphics primitives and SDL/TCOD operations."""

    def __init__(
        self,
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: tuple[int, int],
    ) -> None:
        # Extract SDL components from context
        sdl_renderer = context.sdl_renderer
        assert sdl_renderer is not None

        sdl_atlas = context.sdl_atlas
        assert sdl_atlas is not None

        self.context = context
        self.sdl_renderer: tcod.sdl.render.Renderer = sdl_renderer
        self.console_render = tcod.render.SDLConsoleRender(sdl_atlas)
        self.root_console = root_console
        self.tile_dimensions = tile_dimensions

        # Set up coordinate conversion
        renderer_width, renderer_height = self.sdl_renderer.output_size
        self.coordinate_converter = CoordinateConverter(
            console_width=root_console.width,
            console_height=root_console.height,
            tile_width=self.tile_dimensions[0],
            tile_height=self.tile_dimensions[1],
            renderer_width=renderer_width,
            renderer_height=renderer_height,
        )

    def clear_console(self, console: Console) -> None:
        """Clear a console."""
        console.clear()

    def draw_text(
        self, x: int, y: int, text: str, fg: colors.Color = colors.WHITE
    ) -> None:
        """Draw text at a specific position with a given color."""
        self.root_console.print(x=x, y=y, text=text, fg=fg)

    def blit_console(
        self,
        source: Console,
        dest: Console,
        dest_x: int = 0,
        dest_y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Blit one console onto another."""
        source.blit(
            dest=dest,
            dest_x=dest_x,
            dest_y=dest_y,
            width=width or source.width,
            height=height or source.height,
        )

    def prepare_to_present(self) -> None:
        """Converts the root console to a texture and copies it to the backbuffer."""
        # Convert TCOD console to SDL texture
        console_texture = self.console_render.render(self.root_console)

        renderer_width, renderer_height = self.sdl_renderer.output_size

        # Copy texture to screen, scaling to match the window size
        self.sdl_renderer.clear()
        self.sdl_renderer.copy(
            console_texture,
            dest=(0, 0, renderer_width, renderer_height),
        )

    def finalize_present(self) -> None:
        """Presents the backbuffer to the screen."""
        self.sdl_renderer.present()

    def present_frame(self) -> None:
        """Presents the final composited frame via SDL. (Now a convenience method)"""
        self.prepare_to_present()
        self.finalize_present()

    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        renderer_width, renderer_height = self.sdl_renderer.output_size

        # Determine tile scaling based on current window size
        tile_width = max(1, renderer_width // self.root_console.width)
        tile_height = max(1, renderer_height // self.root_console.height)
        self.tile_dimensions = (tile_width, tile_height)

        self.coordinate_converter = CoordinateConverter(
            console_width=self.root_console.width,
            console_height=self.root_console.height,
            tile_width=tile_width,
            tile_height=tile_height,
            renderer_width=renderer_width,
            renderer_height=renderer_height,
        )
