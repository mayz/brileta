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
from brileta.game.actors.ai.behaviors.request_help import RequestHelpGoal


class TalkExecutor(ActionExecutor[TalkIntent]):
    """Executes talk intents.

    Currently shows a message indicating conversation with the target.
    Future versions may open a dialogue interface.
    """

    def execute(self, intent: TalkIntent) -> GameActionResult | None:
        """Execute the talk action.

        Args:
            intent: The TalkIntent containing the target to talk to.

        Returns:
            A successful result. Talk actions cannot fail.
        """
        target = intent.target

        # NUBS 6 minimal help hook: if the target is asking for help, engaging it
        # completes the request (no failed attempt) and voices the need. The real
        # conversation UI is Phase 7; this just clears the goal and its bubble.
        goal = target.current_goal if isinstance(target, NPC) else None
        if (
            isinstance(target, NPC)
            and isinstance(goal, RequestHelpGoal)
            and not goal.is_complete
            and target.needs
        ):
            need = max(target.needs, key=lambda n: n.urgency)
            goal.fulfill(target)  # type: ignore[arg-type]
            target.current_goal = None
            # Resolving the need stands in for Phase 7's real help; without this
            # the NPC would immediately re-adopt the request and re-approach.
            target.needs.remove(need)
            publish_event(
                MessageEvent(
                    f"{target.name} needs help: {need.describe()}.", colors.CYAN
                )
            )
            return GameActionResult(succeeded=True, should_update_fov=False)

        publish_event(MessageEvent(f"You approach {target.name}.", colors.CYAN))
        return GameActionResult(succeeded=True, should_update_fov=False)
