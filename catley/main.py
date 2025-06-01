"""Main entry point for the game."""

import tcod
import tcod.render
from tcod.console import Console

from . import config
from .controller import Controller


def main() -> None:
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

    # Create TCOD context with proper resource management
    with tcod.context.new(
        console=root_console,
        tileset=tileset,
        title=title,
        vsync=config.VSYNC,
    ) as context:
        # Extract SDL components from the context
        sdl_renderer = context.sdl_renderer
        sdl_atlas = context.sdl_atlas

        # Create console renderer from the atlas
        console_render = tcod.render.SDLConsoleRender(sdl_atlas)

        controller = Controller(
            context, sdl_renderer, console_render, root_console, tileset
        )
        try:
            controller.run_game_loop()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    main()
