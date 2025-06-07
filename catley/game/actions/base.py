"""
Base classes for game actions.

Defines the core GameAction interface that all in-world actions inherit from.
These represent meaningful decisions made by actors that change the game world state.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor


class GameAction(abc.ABC):
    """An action that represents a game turn, such as moving an actor or
    performing an attack. This is distinct from UI actions and is meant to
    be executed within the game loop.

    An action performed by an actor (player or NPC) that directly affects the
    game world's state and typically consumes that actor's turn. Examples include
    moving, attacking, performing stunts or tricks, etc.
    """

    def __init__(self, controller: Controller, actor: Actor) -> None:
        self.controller = controller
        self.actor = actor
        self.frame_manager = self.controller.frame_manager

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute this action."""
        pass

    def can_execute(self) -> tuple[bool, str | None]:
        """Check if this action can be executed. Returns (can_execute, reason_if_not)"""
        return True, None

    def get_success_probability(self) -> float | None:
        """Get probability of success for this action, if applicable"""
        return None
