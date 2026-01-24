"""
Combat stunts - physical maneuvers that manipulate enemy positioning.

Stunts are non-weapon combat actions like Push, Trip, Disarm, and Grapple.
They use opposed stat checks and can create environmental interactions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class PushIntent(GameIntent):
    """Intent to push an adjacent enemy one tile away.

    Resolution uses Strength vs Strength opposed check. On success, the
    defender is moved one tile directly away from the attacker. Environmental
    interactions occur if pushed into hazards, walls, or other actors.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.PushExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a push intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the push.
            defender: The target character being pushed.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender


class TripIntent(GameIntent):
    """Intent to trip an adjacent enemy, causing them to fall prone.

    Resolution uses Agility vs Agility opposed check. Trip is the reliable
    knockdown option - any success applies TrippedEffect (skips 2 turns),
    unlike Push which only causes knockdown on critical success.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.TripExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a trip intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the trip.
            defender: The target character being tripped.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender
