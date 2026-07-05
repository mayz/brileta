"""Social action executors.

Executors for social interactions like talking to NPCs.
"""

from __future__ import annotations

from brileta import colors
from brileta.events import MessageEvent, publish_event
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actions.social import TalkIntent
from brileta.game.actors import NPC


class TalkExecutor(ActionExecutor[TalkIntent]):
    """Executes talk intents by opening the conversation UI.

    Talking to an NPC opens a ConversationMenu (NUBS 7): the menu derives the
    NPC's stance and offers social verbs. If the NPC is mid-request, the menu
    opens on that ask (Accept/Decline); otherwise it opens on a greeting.
    """

    def execute(self, intent: TalkIntent) -> GameActionResult | None:
        """Open a conversation with the target NPC.

        Args:
            intent: The TalkIntent containing the target to talk to.

        Returns:
            A successful result. Talk actions cannot fail.
        """
        target = intent.target

        # Only NPCs have the state (stance, goals, offers) a conversation reads.
        # Talking to a non-NPC character (should not normally happen) just notes
        # the approach rather than opening an empty menu.
        if not isinstance(target, NPC):
            publish_event(MessageEvent(f"You approach {target.name}.", colors.CYAN))
            return GameActionResult(succeeded=True, should_update_fov=False)

        # Local import: the conversation menu pulls in view/UI modules, and
        # importing those at module load would cycle back through the action
        # layer that registers this executor.
        from brileta.view.ui.conversation_menu import open_conversation

        open_conversation(intent.controller, target)
        return GameActionResult(succeeded=True, should_update_fov=False)
