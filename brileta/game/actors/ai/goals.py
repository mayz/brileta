"""
Goal system for multi-turn NPC behaviors.

Most NPC behaviors are single-tick UtilityActions (see utility.py) that need
no memory between ticks. A Goal is only needed when the behavior requires
state across ticks - tracking which waypoint comes next (Patrol), how far
from a threat (Flee), or whether the NPC is approaching vs. signaling vs.
waiting (RequestHelp). If each tick can stand alone, it should be an action,
not a goal.

When an action does need a goal, it creates one on its first tick. After that,
ContinueGoalAction competes in scoring each tick to keep the goal alive. Goals
are a behavioral layer above ActionPlans: a Goal decides *what to achieve* and
produces ActionPlans to handle the mechanical *how* (pathfinding, step
sequencing).

Continuous re-evaluation: goals do not bypass the utility brain. Every tick, the
brain scores all available actions including ContinueGoalAction. The current
goal's continuation gets a persistence bonus based on progress - an NPC 80%
through fleeing is less likely to abandon than one 10% in. If something scores
dramatically higher, the goal is naturally interrupted.

Key classes:
    Goal: Abstract base for multi-turn objectives.
    GoalState: Enum tracking goal lifecycle.
    ContinueGoalAction: Utility action that competes in scoring to continue
        the current goal, with a persistence bonus scaled by goal.progress.

Concrete goal implementations live in the behaviors/ subpackage, colocated
with the actions that create them.
"""

from __future__ import annotations

import abc
from enum import Enum, auto
from typing import TYPE_CHECKING

from .utility import Consideration, Precondition, UtilityAction, UtilityContext

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC

# Small baseline bonus so that ContinueGoalAction always beats the raw
# action it mirrors in a tie. Without this, a goal at progress 0.0 ties
# with the atomic action and gets recreated every tick (progress never
# advances).
PERSISTENCE_MINIMUM = 0.01

# How much weight a fully-progressed goal gets as persistence bonus.
# This is added to the ContinueGoalAction's base score so that goals
# already in progress aren't easily interrupted by marginal score differences.
PERSISTENCE_WEIGHT = 0.3


class GoalState(Enum):
    """Lifecycle states for a Goal.

    These cover the high-level lifecycle. Subclasses that need finer-grained
    states (e.g., APPROACHING/SIGNALING/WAITING for Phase 3 RequestHelpGoal)
    should track those internally rather than expanding this enum.
    """

    ACTIVE = auto()  # Goal is actively being pursued
    COMPLETED = auto()  # Goal achieved its objective
    FAILED = auto()  # Goal could not be achieved
    ABANDONED = auto()  # Goal was interrupted by a higher-scoring action


class Goal(abc.ABC):
    """Abstract base class for multi-turn NPC objectives.

    A Goal manages behavioral state transitions while delegating mechanical
    execution (pathfinding, step sequencing) to the ActionPlan system. Each
    tick, the goal produces the next appropriate intent for the NPC.

    Subclasses must implement:
        get_next_action: Return the next GameIntent for this tick.
        get_considerations: Return considerations for scoring continuation.
        evaluate_completion: Check if the goal should complete or fail.
    """

    def __init__(
        self,
        goal_id: str,
        preconditions: list[Precondition] | None = None,
    ) -> None:
        self.goal_id = goal_id
        self._state = GoalState.ACTIVE
        self._ticks_active: int = 0
        # Eligibility preconditions inherited from the originating action.
        # ContinueGoalAction checks these every tick so that a goal whose
        # preconditions no longer hold (e.g., wander after turning hostile)
        # scores 0 and gets replaced by whatever action wins instead.
        self.preconditions: list[Precondition] = preconditions or []

    @property
    def state(self) -> GoalState:
        """Current lifecycle state for this goal."""
        return self._state

    @state.setter
    def state(self, new_state: GoalState) -> None:
        """Set lifecycle state and run subclass transition hooks."""
        previous_state = self._state
        self._state = new_state
        if previous_state is not new_state:
            self._on_state_changed(previous_state, new_state)

    def _on_state_changed(
        self, previous_state: GoalState, new_state: GoalState
    ) -> None:
        """Hook for subclasses that need state-transition cleanup."""
        _ = previous_state, new_state
        return

    @property
    def progress(self) -> float:
        """How far along this goal is, from 0.0 to 1.0.

        Used by ContinueGoalAction to scale the persistence bonus. An NPC
        80% through fleeing is less likely to abandon than one 10% in.
        Subclasses should override this with meaningful progress tracking.
        """
        return 0.0

    @property
    def is_complete(self) -> bool:
        """Whether this goal has reached a terminal state."""
        return self.state in (
            GoalState.COMPLETED,
            GoalState.FAILED,
            GoalState.ABANDONED,
        )

    def abandon(self) -> None:
        """Mark this goal as abandoned (interrupted by higher-scoring action)."""
        self.state = GoalState.ABANDONED

    def tick(self) -> None:
        """Called each turn the goal remains active. Tracks lifetime."""
        self._ticks_active += 1

    @abc.abstractmethod
    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Return the appropriate intent for the current goal state.

        Called when ContinueGoalAction wins scoring. The goal should return
        the next mechanical action (move, idle, etc.) for the NPC to take.

        Returns:
            A GameIntent to execute, or None to do nothing this tick.
        """
        ...

    @abc.abstractmethod
    def get_considerations(self) -> list[Consideration]:
        """Return the considerations used to score continuing this goal.

        These are evaluated by ContinueGoalAction to determine how strongly
        the NPC wants to keep pursuing this goal vs switching to something else.
        """
        ...

    @abc.abstractmethod
    def get_base_score(self) -> float:
        """Return the base utility score for continuing this goal."""
        ...

    @abc.abstractmethod
    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Check whether this goal should complete or fail.

        Called each tick before get_next_action. Subclasses should set
        self.state to COMPLETED or FAILED as appropriate.
        """
        ...


class ContinueGoalAction(UtilityAction):
    """Utility action representing 'keep doing what I'm doing'.

    Competes in scoring alongside all other actions every tick. Its score
    is the goal's underlying considerations plus a persistence bonus that
    scales with goal.progress. This means goals that are nearly complete
    are harder to interrupt than freshly started ones.
    """

    def __init__(self, goal: Goal) -> None:
        super().__init__(
            action_id=f"continue_{goal.goal_id}",
            base_score=goal.get_base_score(),
            considerations=goal.get_considerations(),
            preconditions=goal.preconditions,
        )
        self.goal = goal

    def score(self, context: UtilityContext) -> float:
        """Score this action with persistence bonus added on top.

        When preconditions fail the goal is ineligible and scores 0 with no
        persistence bonus. When preconditions pass but considerations multiply
        to 0 (e.g., cornered NPC with no escape route), the bonus is still
        applied so the goal survives one more tick and can detect its own
        failure via get_next_action → GoalState.FAILED.
        """
        if any(not pre(context) for pre in self.preconditions):
            return 0.0
        base = super().score(context)
        persistence_bonus = (
            PERSISTENCE_MINIMUM + self.goal.progress * PERSISTENCE_WEIGHT
        )
        return base + persistence_bonus

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled by the goal itself, not this action.
        # The UtilityBrain checks for ContinueGoalAction winning and delegates
        # to goal.get_next_action() instead. This method should not be called
        # directly in normal flow.
        return None
