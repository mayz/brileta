from __future__ import annotations

import tcod
import tcod.sdl.mouse
from tcod.console import Console
from tcod.sdl.video import WindowFlags

from catley import config
from catley.app import AppConfig
from catley.controller import Controller
from catley.types import DeltaTime, InterpolationAlpha
from catley.util.live_vars import live_variable_registry, record_time_live_variable

from .graphics import TCODGraphicsContext


class TCODApp:
    """
    The TCOD implementation of the application driver.

    This class manages the main game loop using a fixed timestep with
    interpolation, ensuring a deterministic simulation and smooth visuals.

    Architecture Overview:
    - FIXED TIMESTEP: Game logic runs at a consistent rate (e.g., 60Hz)
      for a deterministic simulation, handled by the accumulator pattern.
    - VARIABLE RENDERING: Visual frames are rendered as fast as the hardware
      allows (up to a configured cap), independent of logic speed.
    - INTERPOLATION: Smooth movement and animations are achieved by rendering
      objects at an interpolated state between the previous and current
      logic steps, using an 'alpha' value.

    The "accumulator pattern" in the `run` method ensures:
    1. Deterministic game state (the same inputs will always produce the
       same outputs).
    2. Smooth visuals even when logic and rendering rates differ.
    3. Responsive input, as events are processed immediately every visual
       frame, not queued for a logic step.
    """

    def __init__(self, app_config: AppConfig) -> None:
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

        # Create the graphics context
        self.graphics = TCODGraphicsContext(
            self.tcod_context, self.root_console, tileset.tile_shape
        )

        # Create and wire up the core components
        self.controller = Controller(self, self.graphics)

        # The controller is now fully initialized. We can compute the first FOV.
        self.controller.update_fov()

        # Register live variables for performance monitoring
        self.register_metrics()
        self.controller.register_metrics()

    def register_metrics(self) -> None:
        """Registers live variables specific to the App layer."""
        live_variable_registry.register_metric(
            "cpu.total_frame_ms",
            description="Total CPU time for one visual frame",
            num_samples=1000,
        )

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

                # --- Player Action Processing (once per frame) ---
                # This ensures player actions are processed with zero latency,
                # independent of the fixed logic timestep.
                self.controller.process_player_input()

                # --- Time and Logic Loop ---
                # FRAME TIMING: Get elapsed time and add to accumulator
                # This caps visual framerate while accumulating "debt" for logic updates
                delta_time: DeltaTime = self.controller.clock.sync(
                    fps=config.TARGET_FPS
                )
                self.controller.accumulator += delta_time

                # FIXED TIMESTEP LOOP: Run logic updates at exactly 60Hz
                # May run 0, 1, or multiple times per visual frame for consistency
                # Death spiral protection: limit logic steps per frame
                logic_steps_this_frame = 0
                while (
                    self.controller.accumulator >= self.controller.fixed_timestep
                    and logic_steps_this_frame
                    < self.controller.max_logic_steps_per_frame
                ):
                    self.controller.update_logic_step()

                    # CONSUME TIME: Remove one fixed timestep from accumulator
                    # Continue loop if more time needs to be processed
                    self.controller.accumulator -= self.controller.fixed_timestep
                    logic_steps_this_frame += 1

                # RECOVERY: If death spiral protection kicked in, gradually reduce
                # This prevents permanent slowdown when performance recovers
                if logic_steps_this_frame >= self.controller.max_logic_steps_per_frame:
                    # Gradually bleed off excess time to recover from spikes
                    self.controller.accumulator *= 0.8

                # --- Rendering ---
                with record_time_live_variable("cpu.total_frame_ms"):
                    # INTERPOLATION ALPHA: Calculate how far between logic steps we are
                    # alpha=0.0 means "exactly at previous step", alpha=1.0 is "current"
                    # alpha=0.5 means "halfway between steps" - creates smooth movement
                    # Clamp to 1.0 to prevent overshoot when death spiral hits
                    alpha = InterpolationAlpha(
                        min(
                            1.0,
                            self.controller.accumulator
                            / self.controller.fixed_timestep,
                        )
                    )
                    self.prepare_for_new_frame()
                    self.controller.render_visual_frame(alpha)
                    self.present_frame()
        finally:
            tcod.sdl.mouse.show(True)

    def handle_resize(self, event: tcod.event.WindowResized) -> None:
        """Handles a window resize event by enforcing aspect ratio and
        notifying subsystems."""
        # 1. Perform the resize logic using objects the App owns.
        sdl_window = self.tcod_context.sdl_window
        if sdl_window is not None:
            ASPECT_RATIO = 8 / 5
            new_width = event.width
            new_height = int(new_width / ASPECT_RATIO)
            sdl_window.size = new_width, new_height

        # 2. Notify the other components that need to know about the change.
        # The App owns the graphics context and the controller, so it can call them.
        self.graphics.update_dimensions()
        assert self.controller.frame_manager is not None
        self.controller.frame_manager.on_window_resized()

    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        # FIXME: Get rid of the prepare_to_present() method in TCODGraphicsContext and
        #        move that code here.
        self.graphics.prepare_to_present()

    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen."""
        # FIXME: Get rid of the finalize_present() method in TCODGraphicsContext and
        #        move that code here.
        self.graphics.finalize_present()

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        window = self.tcod_context.sdl_window
        if not window:
            return
        if window.fullscreen:
            window.fullscreen = False
        else:
            window.fullscreen = WindowFlags.FULLSCREEN_DESKTOP
