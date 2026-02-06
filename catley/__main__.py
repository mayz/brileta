"""Main entry point for the game."""

import logging

from . import config
from .app import App, AppConfig
from .util import rng


def main() -> None:
    # Suppress SDL3 startup info messages from tcod
    logging.getLogger("tcod.sdl").setLevel(logging.ERROR)

    # Initialize the RNG stream system for deterministic randomness
    rng.init(config.RANDOM_SEED)

    app_config = AppConfig(
        title=config.WINDOW_TITLE,
        width=config.SCREEN_WIDTH,
        height=config.SCREEN_HEIGHT,
        vsync=config.VSYNC,
    )

    match config.BACKEND.app:
        case "glfw":
            from catley.backends.glfw.app import GlfwApp

            _APP_CLASS = GlfwApp

    app: App = _APP_CLASS(app_config)
    app.run()


if __name__ == "__main__":
    main()
