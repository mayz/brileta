"""Social action executors.

Executors for social interactions like talking to NPCs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from catley.game.actions.social import TalkIntent


class TalkExecutor(ActionExecutor):
    """Executes talk intents.

    Currently shows a message indicating conversation with the target.
    Future versions may open a dialogue interface.
    """

    def execute(self, intent: TalkIntent) -> GameActionResult | None:  # type: ignore[override]
        """Execute the talk action.

        Args:
            intent: The TalkIntent containing the target to talk to.

        Returns:
            A successful result. Talk actions cannot fail.
        """
        publish_event(MessageEvent(f"You approach {intent.target.name}.", colors.CYAN))
        return GameActionResult(succeeded=True, should_update_fov=False)
