from __future__ import annotations

import tcod
import tcod.sdl.mouse
import tcod.sdl.render
from tcod.console import Console
from tcod.sdl.video import WindowFlags

from catley import config
from catley.app import App, AppConfig
from catley.types import DeltaTime

from .graphics import TCODGraphicsContext


class TCODApp(App):
    """
    The TCOD implementation of the application driver.

    Uses TCOD's polling-based architecture to implement the shared fixed
    timestep game loop pattern.
    """

    def __init__(self, app_config: AppConfig) -> None:
        super().__init__(app_config)

        # Create TCOD-specific resources
        tileset = tcod.tileset.load_tilesheet(
            config.TILESET_PATH,
            columns=config.TILESET_COLUMNS,
            rows=config.TILESET_ROWS,
            charmap=tcod.tileset.CHARMAP_CP437,
        )
        self.root_console = Console(
            config.SCREEN_WIDTH, config.SCREEN_HEIGHT, order="F"
        )
        sdl_flags = (
            WindowFlags.OPENGL
            # Enables high-DPI awareness. On displays like Retina or scaled 4K
            # monitors, this allows the game to render at the full native resolution,
            # preventing blur.
            | WindowFlags.ALLOW_HIGHDPI
            | WindowFlags.RESIZABLE
            | WindowFlags.MAXIMIZED
        )
        self.tcod_context = tcod.context.new(
            console=self.root_console,
            tileset=tileset,
            title=app_config.title,
            vsync=app_config.vsync,
            sdl_window_flags=sdl_flags,
        )

        self.sdl_renderer: tcod.sdl.render.Renderer = self.tcod_context.sdl_renderer  # type: ignore[assignment]

        # Create the graphics context
        self.graphics = TCODGraphicsContext(
            self.tcod_context, self.root_console, tileset.tile_shape
        )

        # Initialize shared controller
        self._initialize_controller(self.graphics)

    def run(self) -> None:
        """Starts the main application loop and runs the game."""
        assert self.controller is not None
        assert self.controller.input_handler is not None

        try:
            tcod.sdl.mouse.show(False)
            while True:
                # --- Input ---
                for event in tcod.event.get():
                    # Check for App-level commands like fullscreen
                    if isinstance(event, tcod.event.WindowResized):
                        self.handle_resize(event)
                    elif (
                        isinstance(event, tcod.event.KeyDown)
                        and event.sym == tcod.event.KeySym.RETURN
                        and (event.mod & tcod.event.Modifier.ALT)
                    ):
                        self.toggle_fullscreen()
                    else:
                        # Otherwise, dispatch to the game's input handler
                        self.controller.input_handler.dispatch(event)

                # --- Time and Logic Loop ---
                delta_time: DeltaTime = self.controller.clock.sync(
                    fps=config.TARGET_FPS
                )
                self.update_game_logic(delta_time)

                # --- Rendering ---
                self.render_frame()
        finally:
            tcod.sdl.mouse.show(True)

    def handle_resize(self, event: tcod.event.WindowResized) -> None:
        """Handles a window resize event by enforcing aspect ratio and
        notifying subsystems."""
        # 1. Perform the resize logic using objects the App owns.
        sdl_window = self.tcod_context.sdl_window
        if sdl_window is not None:
            # This logic can be more sophisticated, but for now, this works.
            # A better approach might be to not force a size but to let letterboxing
            # handle it.
            ASPECT_RATIO = 8 / 5
            new_width = event.width
            new_height = int(new_width / ASPECT_RATIO)
            # Might want to remove this line to allow free resizing
            # and let letterboxing handle the aspect ratio.
            sdl_window.size = new_width, new_height

        # 2. Notify the other components that need to know about the change.
        # The App owns the graphics context and the controller, so it can call them.
        self.graphics.update_dimensions()
        assert self.controller is not None
        assert self.controller.frame_manager is not None
        self.controller.frame_manager.on_window_resized()

    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        # It's the App's responsibility to manage the main render target.
        self.root_console.clear()  # Clear the console that views will draw ON.
        console_texture = self.graphics._texture_from_console(
            self.root_console, transparent=False
        )
        self.sdl_renderer.clear()  # Clear the window backbuffer.

        # Get the destination rect from the graphics context
        dest_rect = self.graphics.get_render_destination_rect()

        # Blit the (now cleared) root console texture to its letterboxed position.
        # This sets up the black background for the frame.
        self.sdl_renderer.copy(console_texture, dest=dest_rect)

    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen."""
        self.sdl_renderer.present()

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        window = self.tcod_context.sdl_window
        if not window:
            return
        if window.fullscreen:
            window.fullscreen = False
        else:
            window.fullscreen = WindowFlags.FULLSCREEN_DESKTOP

    def _exit_backend(self) -> None:
        """Performs backend-specific exit procedures for TCOD."""
        raise SystemExit()
