from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from catley.types import DeltaTime, InterpolationAlpha
from catley.util.live_vars import (
    MetricSpec,
    live_variable_registry,
    record_time_live_variable,
)
from catley.view.render.graphics import GraphicsContext

# App-level metrics.
_APP_METRICS: list[MetricSpec] = [
    MetricSpec("time.total_ms", "Total time for one visual frame", 1000),
]
live_variable_registry.register_metrics(_APP_METRICS)

if TYPE_CHECKING:
    from catley.controller import Controller


@dataclass
class AppConfig:
    """Configuration for an App implementation."""

    width: int  # Console width in tiles
    height: int  # Console height in tiles
    title: str
    vsync: bool

    # Window behavior
    resizable: bool = True
    maximized: bool = True
    fullscreen: bool = False


class App[TGraphics: GraphicsContext](ABC):
    """
    Abstract base class for application drivers.

    An App is the top-level component responsible for bridging the abstract game
    logic with a concrete windowing and event-handling backend (e.g., TCOD,
    GLFW). It owns the main loop and drives the entire application.

    This class implements the fixed timestep with interpolation pattern that
    ensures a deterministic simulation and smooth visuals across all backends.

    Architecture Overview:
    ----------------------
    - FIXED TIMESTEP: Game logic runs at a consistent rate (e.g., 60Hz)
      for a deterministic simulation, handled by the accumulator pattern.
    - VARIABLE RENDERING: Visual frames are rendered as fast as the hardware
      allows (up to a configured cap), independent of logic speed.
    - INTERPOLATION: Smooth movement and animations are achieved by rendering
      objects at an interpolated state between the previous and current
      logic steps, using an 'alpha' value.

    The "accumulator pattern" ensures:
    1. Deterministic game state (the same inputs will always produce the
       same outputs).
    2. Smooth visuals even when logic and rendering rates differ.
    3. Responsive input, as events are processed immediately every visual
       frame, not queued for a logic step.

    Backend Responsibilities:
    -------------------------
    Subclasses must implement backend-specific functionality:
    - Window and Context Management: Creates and manages the main application window
      and its underlying graphics context (e.g., OpenGL).
    - Input Event Translation: Captures backend-specific input events and translates
      them into the game's internal, backend-agnostic event format (tcod.event objects).
    - Frame Presentation: Handles the final step of presenting rendered frames to
      screen.
    """

    def __init__(self, app_config: AppConfig) -> None:
        """Initializes the application with shared components."""
        self.app_config = app_config

        # Will be set by subclasses after they create their graphics context
        self.graphics: TGraphics  # Subclasses must initialize this
        self.controller: Controller | None = None

    def _initialize_controller(self) -> None:
        """Initializes the controller after graphics context is ready."""
        from catley.controller import Controller

        self.controller = Controller(self, self.graphics)
        self.controller.update_fov()

    @abstractmethod
    def run(self) -> None:
        """Starts the main application loop and runs the game."""
        pass

    def update_game_logic(self, delta_time: DeltaTime) -> None:
        """
        Updates game logic using fixed timestep with accumulator pattern.

        This method implements the core timing logic shared by all backends:
        - Accumulates frame time for consistent logic updates
        - Runs fixed timestep logic updates (may be 0, 1, or multiple per frame)
        - Implements death spiral protection to prevent permanent slowdown
        """
        assert self.controller is not None

        # --- Player Action Processing (once per frame) ---
        # This ensures player actions are processed with zero latency,
        # independent of the fixed logic timestep.
        self.controller.process_player_input()

        # --- Time and Logic Loop ---
        # FRAME TIMING: Get elapsed time and add to accumulator
        # This caps visual framerate while accumulating "debt" for logic updates
        self.controller.accumulator += delta_time

        # FIXED TIMESTEP LOOP: Run logic updates at exactly 60Hz
        # May run 0, 1, or multiple times per visual frame for consistency
        # Death spiral protection: limit logic steps per frame
        logic_steps_this_frame = 0
        while (
            self.controller.accumulator >= self.controller.fixed_timestep
            and logic_steps_this_frame < self.controller.max_logic_steps_per_frame
        ):
            # Call shared logic execution method
            self.execute_fixed_logic_step()

            # CONSUME TIME: Remove one fixed timestep from accumulator
            # Continue loop if more time needs to be processed
            self.controller.accumulator -= self.controller.fixed_timestep
            logic_steps_this_frame += 1

        # RECOVERY: If death spiral protection kicked in, gradually reduce
        # This prevents permanent slowdown when performance recovers
        if logic_steps_this_frame >= self.controller.max_logic_steps_per_frame:
            self.controller.accumulator *= 0.8

    def execute_fixed_logic_step(self) -> None:
        """
        Execute exactly one logic step at fixed 60Hz timing.

        This method is called by timing strategies:
        - Accumulator-based (TCOD, GLFW):
            0-N times per visual frame (accumulator catches up to maintain 60Hz)
        - Event-driven:
            exactly once per 1/60 second (scheduler guarantees 60Hz)

        Shared execution ensures identical game logic across all backends.
        """
        assert self.controller is not None
        # Note: process_player_input() is handled differently by each backend
        # Accumulator-based (TCOD, GLFW):
        #   calls it once per visual frame in update_game_logic()
        # Event-driven:
        #   will call it in its own timing
        self.controller.update_logic_step()

    def render_frame(self) -> None:
        """
        Renders a visual frame with interpolation (accumulator-based timing).

        This method is used by polling backends (TCOD) that use the accumulator
        pattern. It calculates interpolation alpha from the accumulator state.
        """
        assert self.controller is not None

        # INTERPOLATION ALPHA: Calculate how far between logic steps we are
        # alpha=0.0 means "exactly at previous step", alpha=1.0 is "current"
        # alpha=0.5 means "halfway between steps" - creates smooth movement
        # Clamp to 1.0 to prevent overshoot when death spiral hits
        alpha = InterpolationAlpha(
            min(
                1.0,
                self.controller.accumulator / self.controller.fixed_timestep,
            )
        )

        self.render_frame_with_alpha(alpha)

    def render_frame_with_alpha(self, alpha: InterpolationAlpha) -> None:
        """
        Renders a visual frame with explicit interpolation alpha.

        This method is called by rendering strategies:
        - Accumulator-based (TCOD, GLFW): via render_frame() with calculated alpha
        - Event-driven: directly with manually-calculated alpha

        Shared rendering ensures identical visual output across all backends.
        """
        assert self.controller is not None

        with record_time_live_variable("time.total_ms"):
            self.prepare_for_new_frame()
            self.controller.render_visual_frame(alpha)
            self.present_frame()

    @abstractmethod
    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        pass

    @abstractmethod
    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen."""
        pass

    @abstractmethod
    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        pass

    @abstractmethod
    def _exit_backend(self) -> None:
        """Performs backend-specific exit procedures."""
        pass

    def quit(self) -> None:
        """Initiates the application shutdown process."""
        if self.controller:
            self.controller.cleanup()
        self._exit_backend()
