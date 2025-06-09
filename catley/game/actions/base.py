"""
Base classes for game actions.

Defines the core GameAction interface that all in-world actions inherit from.
These represent meaningful decisions made by actors that change the game world state.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catley.game.consequences import Consequence

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor


@dataclass
class GameActionResult:
    """Result returned by :meth:`GameAction.execute`.

    When ``should_update_fov`` is ``True`` the player's field of view will be
    recomputed after the action is processed by
    :class:`~catley.turn_manager.TurnManager`.
    """

    consequences: list[Consequence] = field(default_factory=list)
    should_update_fov: bool = False


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

    @abc.abstractmethod
    def execute(self) -> GameActionResult | None:
        """Execute this action.

        Returning ``None`` means no special post-processing is required by the
        caller. A :class:`GameActionResult` can be returned to request additional
        handling such as a field-of-view update.
        """
        pass
