"""
Movement actions for actors in the game world.

Handles actor movement including collision detection and automatic ramming
when attempting to move into occupied spaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class MoveIntent(GameIntent):
    """Intent for moving an actor on the game map."""

    def __init__(
        self, controller: Controller, actor: Character, dx: int, dy: int
    ) -> None:
        super().__init__(controller, actor)

        # Type narrowing.
        self.actor: Character

        self.dx = dx
        self.dy = dy
        self.newx = self.actor.x + self.dx
        self.newy = self.actor.y + self.dy
