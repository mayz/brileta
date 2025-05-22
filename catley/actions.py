"""
This module contains the Action class and its subclasses that represent game actions.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import tcod

if TYPE_CHECKING:
    from engine import Controller
    from model import Entity


class Action(abc.ABC):
    """Base class for all game actions."""

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute the action."""
        raise NotImplementedError()


class MoveAction(Action):
    """Action for moving an entity on the game map."""

    def __init__(
        self, controller: Controller, entity: Entity, dx: int, dy: int
    ) -> None:
        self.controller = controller
        self.game_map = controller.model.game_map
        self.entity = entity

        self.dx = dx
        self.dy = dy
        self.newx = self.entity.x + self.dx
        self.newy = self.entity.y + self.dy

    def execute(self) -> None:
        if self.game_map.tiles[self.newx][self.newy].blocked:
            return

        # Check for blocking entities
        for entity in self.controller.model.entities:
            if (
                entity.blocks_movement
                and entity.x == self.newx
                and entity.y == self.newy
            ):
                return  # Cannot move into blocking entity

        self.entity.move(self.dx, self.dy)
        self.controller.fov.fov_needs_recomputing = True


class ToggleFullscreenAction(Action):
    """Action for toggling fullscreen mode."""

    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitAction(Action):
    """Action for quitting the game."""

    def execute(self) -> None:
        raise SystemExit()
