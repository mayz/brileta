"""Game Mode System - Behavioral overlays that change how the game works.

Modes are a powerful architectural component for temporarily altering game
behavior while keeping the core game logic running. They are distinct from
Overlays (like menus), which primarily display information.

Use a Mode when you need to:
- Fundamentally change how player input is interpreted (e.g., mapping keys
  to target cycling instead of movement).
- Provide visual feedback directly within the game world (e.g., highlighting
  targets or drawing an area-of-effect preview).
- Create a temporary, stateful context for a complex action (e.g., a
  multi-step lockpicking sequence).

The input handling priority is always: Active Mode -> UI Overlays -> Game Actions.
The lifecycle is managed by the Controller: enter() -> handle input/render -> exit_..._mode().
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import tcod.event
from tcod.console import Console

from catley.game.actors import Character

if TYPE_CHECKING:
    from catley.controller import Controller


class Mode(abc.ABC):
    """Base class for all game behavioral modes.

    A Mode defines a temporary change in how the game interprets input and
    what is rendered on screen. It has direct hooks to render effects both
    within the world view and on the main UI console.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller: Controller = controller
        self.active: bool = False

    @abc.abstractmethod
    def enter(self) -> None:
        """Initialize and activate the mode.

        Subclasses should always call super().enter() to set the active flag.
        """
        self.active = True

    @abc.abstractmethod
    def _exit(self) -> None:
        """Perform internal cleanup and deactivate the mode.

        This method should only be called by the Controller's exit_*_mode()
        methods, not directly. Subclasses should always call super()._exit().
        """
        self.active = False

    @abc.abstractmethod
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events with priority over other systems.

        Return True if the event was consumed, preventing it from being
        processed by overlays or the main input handler.
        """
        pass

    @abc.abstractmethod
    def render_ui(self, console: Console) -> None:
        """Render UI elements on the root console, outside the world view.

        Use this for status text or instructions related to the mode.
        If a mode's UI needs become complex or interactive, consider using a
        dedicated Overlay instead.
        """
        pass

    def render_world(self) -> None:  # noqa: B027
        """Render visual effects inside the game world's viewport.

        Use this for effects like highlighting actors, drawing targeting lines,
        or showing area-of-effect previews on the game map itself.
        """
        pass

    def update(self) -> None:  # noqa: B027
        """Perform per-frame updates.

        Useful for animations, state validation, or other logic that needs
        to run on every frame while the mode is active.
        """
        pass

    def on_actor_death(self, actor: Character) -> None:  # noqa: B027
        """Hook to react to an actor's death.

        Override this if the mode needs to track specific actors and must
        react if one of them is killed (e.g., by removing them from a list
        of valid targets).
        """
        pass