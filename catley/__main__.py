"""Main entry point for the game."""

import random

from . import config
from .app import App, AppConfig


def main() -> None:
    random.seed(config.RANDOM_SEED)

    app_config = AppConfig(
        title=config.WINDOW_TITLE,
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
        vsync=config.VSYNC,
    )

    if True:
        from .backends.tcod.app import TCODApp

        app: App = TCODApp(app_config)
    else:
        from .backends.pyglet.app import PygletApp

        app: App = PygletApp(app_config)

    app.run()


if __name__ == "__main__":
    main()
