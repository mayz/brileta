"""Social interaction actions.

Intent classes for interacting with NPCs through dialogue and other social means.
Executors are in catley/game/actions/executors/social.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class TalkIntent(GameIntent):
    """Intent to initiate conversation with an NPC.

    This is a data class describing the intent to talk. The actual execution
    logic is in TalkExecutor.
    """

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
    ) -> None:
        """Create a talk intent.

        Args:
            controller: The game controller.
            actor: The character initiating conversation (usually player).
            target: The NPC to talk to.
        """
        super().__init__(controller, actor)
        self.target = target
