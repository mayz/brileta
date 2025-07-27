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

    match config.APP_BACKEND:
        case "tcod":
            from catley.backends.tcod.app import TCODApp

            _APP_CLASS = TCODApp
        case "glfw":
            from catley.backends.glfw.app import GlfwApp

            _APP_CLASS = GlfwApp
        case _:
            raise ValueError(f"Unknown app backend: {config.APP_BACKEND}")

    app: App = _APP_CLASS(app_config)
    app.run()


if __name__ == "__main__":
    main()
