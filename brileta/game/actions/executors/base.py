from __future__ import annotations

import abc

from brileta.game.actions.base import GameActionResult, GameIntent


class ActionExecutor[IntentT: GameIntent](abc.ABC):
    """
    Base class for action executors.

    ActionExecutors are specialist classes responsible for orchestrating
    the entire resolution of a single type of Intent. Each executor contains
    all the high-level coordination logic for an action, using various
    low-level systems to apply the final results to the GameWorld.

    Executors are stateless workers - they access the Controller through
    ``intent.controller`` rather than storing a reference. This keeps
    executors decoupled from specific game instances.

    Executors are the "private implementation" of an action's logic and should
    only ever be called by the ``ActionRouter``. They should never be called
    directly from UI or AI code.
    """

    @abc.abstractmethod
    def execute(self, intent: IntentT) -> GameActionResult | None:
        """Execute the given intent.

        Args:
            intent: The intent object specifying the desired action

        Returns:
            GameActionResult with any post-processing requirements, or None
        """
        pass
