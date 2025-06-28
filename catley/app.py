from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AppConfig:
    """Configuration for an App implementation."""

    width: int
    height: int
    title: str
    vsync: bool


class App(Protocol):
    """
    Defines the structural interface for an application driver.

    An App is the top-level component responsible for bridging the abstract game
    logic with a concrete windowing and event-handling backend (e.g., TCOD,
    Pyglet, SDL). It owns the main loop and drives the entire application.

    Responsibilities:
    -----------------
    - Window and Context Management: Creates and manages the main application window
      and its underlying graphics context (e.g., OpenGL).

    - Main Loop Execution: Implements the run() method, which starts and manages the
      application's main event loop. The nature of this loop (e.g., polling vs.
      event-driven) is specific to the backend.

    - Frame Tick Triggering: In each iteration of its main loop, the App must calculate
      the delta_time (time elapsed since the last iteration) and call the Controller's
      main update method. This drives the game's clock.

    - Input Event Translation: Captures backend-specific input events (like keyboard
      presses or mouse movement) and translates them into the game's internal,
      backend-agnostic event format (tcod.event objects).  It then dispatches these
      standardized events to the Controller.

    Why a Protocol and not an ABC?
    -------------------------------
    This interface is defined as a Protocol rather than an Abstract Base Class for a
    crucial architectural reason: different backends have fundamentally different
    programming paradigms for their main loops.

    - A TCOD-based implementation uses an active, polling `while True:` loop.
    - A Pyglet-based implementation uses a passive, callback-driven event loop
      (`pyglet.app.run()`).

    Forcing these two disparate models to inherit from a common base class
    would result in a leaky and awkward abstraction. One implementation would
    have to contort its natural control flow to fit the abstract methods.

    By using a Protocol, we define a contract based on structure and behavior
    ("if it has a `run()` method, it's a valid App") rather than inheritance.
    This allows each implementation to use its backend's idiomatic approach
    while guaranteeing that they are interchangeable from the perspective of
    __main__.py. It provides strong static type-checking without imposing a rigid,
    inappropriate class hierarchy.
    """

    def __init__(self, app_config: AppConfig) -> None:
        """Initializes the host with the controller and configuration."""
        ...

    def run(self) -> None:
        """Starts the main application loop and runs the game."""
        ...

    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        ...

    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen."""
        ...

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        ...
