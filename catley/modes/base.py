from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import tcod.event
from tcod.console import Console

from catley.game.actors import Character

if TYPE_CHECKING:
    from catley.controller import Controller


class Mode(abc.ABC):
    """Abstract base for all game modes (e.g., targeting, inventory, etc.)."""

    def __init__(self, controller: Controller) -> None:
        self.controller: Controller = controller
        self.active: bool = False

    @abc.abstractmethod
    def enter(self) -> None:
        """Called when mode is activated."""
        self.active = True

    @abc.abstractmethod
    def exit(self) -> None:
        """Called when mode is deactivated."""
        self.active = False

    @abc.abstractmethod
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input for this mode. Return True if event was consumed."""
        pass

    @abc.abstractmethod
    def render_ui(self, console: Console) -> None:
        """Render mode-specific UI overlays (text, menus, etc.)"""
        pass

    def render_world(self) -> None:  # noqa: B027
        """Render mode-specific effects in world space (tile highlights, effects, etc.).

        Default implementation does nothing - modes only override if they need
        world rendering.
        """
        pass

    def update(self) -> None:  # noqa: B027
        """Called each frame for mode-specific updates"""
        pass

    def on_actor_death(self, actor: Character) -> None:  # noqa: B027
        """Called when an actor dies. Override in subclasses that care."""
        pass
