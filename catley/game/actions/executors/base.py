from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from catley.game.actions.base import GameActionResult

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameAction


class ActionExecutor(abc.ABC):
    """Base class for action executors.

    ActionExecutors are specialist classes responsible for orchestrating
    the entire resolution of a single type of Intent. Each executor contains
    all the high-level coordination logic for an action type, using various
    low-level systems to apply the final results to the GameWorld.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller

    @abc.abstractmethod
    def execute(self, intent: GameAction) -> GameActionResult | None:
        """Execute the given intent.

        Args:
            intent: The intent object specifying the desired action

        Returns:
            GameActionResult with any post-processing requirements, or None
        """
        pass
