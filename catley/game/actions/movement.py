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
    """Intent for moving an actor on the game map.

    The optional `duration_ms` parameter allows callers to specify custom timing
    for movement. This duration controls both animation speed and pacing before
    the next action. If not provided, the executor uses AUTOPILOT_MOVE_DURATION_MS.
    """

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        dx: int,
        dy: int,
        duration_ms: int | None = None,
    ) -> None:
        super().__init__(controller, actor)

        # Type narrowing.
        self.actor: Character

        self.dx = dx
        self.dy = dy

        # Optional duration override. Controls both animation and pacing.
        # If None, executor uses config.AUTOPILOT_MOVE_DURATION_MS.
        self.duration_ms = duration_ms

    @property
    def newx(self) -> int:
        """Return the actor's prospective x-coordinate when this intent executes."""
        return self.actor.x + self.dx

    @property
    def newy(self) -> int:
        """Return the actor's prospective y-coordinate when this intent executes."""
        return self.actor.y + self.dy
