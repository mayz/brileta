"""Main entry point for the game."""

import tcod
from controller import Controller
from tcod.console import Console


def main() -> None:
    screen_width = 80
    screen_height = 50
    title = "Catley Prototype"

    tileset = tcod.tileset.load_tilesheet(
        "Taffer_20x20.png",
        columns=16,
        rows=16,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    root_console = Console(screen_width, screen_height, order="F")

    with tcod.context.new(
        console=root_console,
        tileset=tileset,
        title=title,
        vsync=True,
    ) as context:
        controller = Controller(context, root_console)
        try:
            controller.run_game_loop()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


if __name__ == "__main__":
    main()
