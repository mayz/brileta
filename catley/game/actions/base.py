"""
Base classes for game intents.

Defines the core GameIntent interface that all in-world intents inherit from.
These represent meaningful decisions made by actors that will be executed
by the TurnManager through the Intent/Executor pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from catley.game.consequences import Consequence

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor


@dataclass
class GameActionResult:
    """Result returned by action executors.

    When ``should_update_fov`` is ``True`` the player's field of view will be
    recomputed after the action is processed by
    :class:`~catley.turn_manager.TurnManager`.
    """

    succeeded: bool = True
    should_update_fov: bool = False
    blocked_by: Any | None = None
    block_reason: str | None = None
    consequences: list[Consequence] = field(default_factory=list)


class GameIntent:
    """Data-only object describing an intended game action.

    Intents are routed through dedicated executors by
    :class:`~catley.game.turn_manager.TurnManager`. They contain only
    the data needed to describe what the actor wants to do, with no
    execution logic.
    """

    def __init__(self, controller: Controller, actor: Actor) -> None:
        self.controller = controller
        self.actor = actor
