"""Main entry point for the game."""

import random

import tcod
from tcod.console import Console
from tcod.sdl.video import WindowFlags

from . import config
from .controller import Controller


def main() -> None:
    random.seed(config.RANDOM_SEED)

    screen_width = config.SCREEN_WIDTH
    screen_height = config.SCREEN_HEIGHT
    title = config.WINDOW_TITLE

    tileset = tcod.tileset.load_tilesheet(
        config.TILESET_PATH,
        columns=config.TILESET_COLUMNS,
        rows=config.TILESET_ROWS,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    root_console = Console(screen_width, screen_height, order="F")

    sdl_window_flags: int = (
        # Requests an OpenGL-accelerated rendering context. Essential for compatibility
        # with the future GPU-based lighting system using moderngl.
        WindowFlags.OPENGL
        # Enables high-DPI awareness. On displays like Retina or scaled 4K monitors,
        # this allows the game to render at the full native resolution, preventing blur.
        #
        # FIXME: This currently breaks the mouse cursor scaling. Figure out why,
        #        then re-enable.
        # | WindowFlags.ALLOW_HIGHDPI
        | WindowFlags.RESIZABLE
        | WindowFlags.MAXIMIZED
    )

    # Create TCOD context with proper resource management
    with tcod.context.new(
        console=root_console,
        tileset=tileset,
        title=title,
        vsync=config.VSYNC,
        sdl_window_flags=sdl_window_flags,
    ) as context:
        controller = Controller(context, root_console, tileset.tile_shape)
        try:
            controller.run_game_loop()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    main()
