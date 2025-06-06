"""Game Mode System - Behavioral overlays that change how the game works.

Modes temporarily alter game behavior while keeping the core game running:
- Targeting (select enemies/locations)
- Pickpocketing
- Inventory management

Use modes for behavioral changes. Use overlays for temporary UI.

Modes get first input priority and can render world effects.
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
    """Base class for game behavioral modes.

    Modes change how the game interprets input and add specialized
    visual effects while keeping the core game running.

    Input priority: Mode -> UI Commands -> Game Actions
    Lifecycle: enter() -> handle input/render -> exit()
    """

    def __init__(self, controller: Controller) -> None:
        self.controller: Controller = controller
        self.active: bool = False

    @abc.abstractmethod
    def enter(self) -> None:
        """Initialize mode. Always call super().enter()."""
        self.active = True

    @abc.abstractmethod
    def exit(self) -> None:
        """Clean up mode. Always call super().exit()."""
        self.active = False

    @abc.abstractmethod
    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input with priority over game. Return True if consumed."""
        pass

    @abc.abstractmethod
    def render_ui(self, console: Console) -> None:
        """Render mode instructions and status over the game world."""
        pass

    def render_world(self) -> None:  # noqa: B027
        """Render world effects like highlights and targeting indicators."""
        pass

    def update(self) -> None:  # noqa: B027
        """Per-frame updates for animations and state validation."""
        pass

    def on_actor_death(self, actor: Character) -> None:  # noqa: B027
        """Handle actor death events. Override if mode tracks actors."""
        pass
