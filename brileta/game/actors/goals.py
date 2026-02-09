"""
Goal system for multi-turn NPC behaviors.

Most NPC behaviors are single-tick Actions (see utility.py) that need no memory
between ticks. A Goal is only needed when the behavior requires state across
ticks - tracking which waypoint comes next (Patrol), how far from a threat
(Flee), or whether the NPC is approaching vs. signaling vs. waiting
(RequestHelp). If each tick can stand alone, it should be an action, not a goal.

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
    FleeGoal: Flee from a threat to a safe distance, then reassess.
    PatrolGoal: Cycle between waypoints.
"""

from __future__ import annotations

import abc
from enum import Enum, auto
from typing import TYPE_CHECKING

from brileta.game.actors.utility import (
    Action as UtilityAction,
)
from brileta.game.actors.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    UtilityContext,
)

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

    def __init__(self, goal_id: str) -> None:
        self.goal_id = goal_id
        self.state = GoalState.ACTIVE
        self._ticks_active: int = 0

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
        )
        self.goal = goal

    def score(self, context: UtilityContext) -> float:
        """Score this action with persistence bonus added on top."""
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


# ---------------------------------------------------------------------------
# Concrete Goal Implementations
# ---------------------------------------------------------------------------

# How far the NPC should be from the threat before considering itself safe.
_FLEE_SAFE_DISTANCE = 8


class FleeGoal(Goal):
    """Flee from a threat until safely distant, then reassess.

    The FleeGoal moves the NPC away from a threat actor. Each tick it picks
    the best adjacent tile that increases distance from the threat. When the
    NPC reaches a safe distance or the threat is no longer present/alive,
    the goal completes.

    Progress is based on distance traveled away from the threat relative to
    the safe distance threshold.
    """

    def __init__(self, threat_actor_id: int) -> None:
        super().__init__(goal_id="flee")
        self.threat_actor_id = threat_actor_id
        self._start_distance: int | None = None
        self._current_distance: int | None = None

    @property
    def progress(self) -> float:
        """Progress based on how close to safe distance we are.

        Returns 0.0 when at starting distance, 1.0 when at safe distance.
        """
        if self._current_distance is None or self._start_distance is None:
            return 0.0
        if self._start_distance >= _FLEE_SAFE_DISTANCE:
            return 1.0
        distance_gained = self._current_distance - self._start_distance
        distance_needed = _FLEE_SAFE_DISTANCE - self._start_distance
        if distance_needed <= 0:
            return 1.0
        return max(0.0, min(1.0, distance_gained / distance_needed))

    def get_base_score(self) -> float:
        return 1.0

    def get_considerations(self) -> list[Consideration]:
        """Flee scores high when health is low and threat is present."""
        return [
            Consideration(
                "health_percent",
                ResponseCurve(ResponseCurveType.INVERSE),
                weight=2.0,
            ),
            Consideration(
                "threat_level",
                ResponseCurve(ResponseCurveType.LINEAR),
            ),
            Consideration(
                "has_escape_route",
                ResponseCurve(ResponseCurveType.STEP, threshold=0.5),
            ),
        ]

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Complete if threat is gone/dead or we're at safe distance."""
        from brileta.game import ranges

        threat = controller.gw.get_actor_by_id(self.threat_actor_id)

        # Threat no longer exists or is dead - goal complete
        if threat is None or (threat.health and not threat.health.is_alive()):
            self.state = GoalState.COMPLETED
            return

        distance = ranges.calculate_distance(npc.x, npc.y, threat.x, threat.y)
        self._current_distance = distance

        if self._start_distance is None:
            self._start_distance = distance

        if distance >= _FLEE_SAFE_DISTANCE:
            self.state = GoalState.COMPLETED

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Move away from the threat, picking the best escape direction."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
        from brileta.game import ranges
        from brileta.game.actions.movement import MoveIntent

        threat = controller.gw.get_actor_by_id(self.threat_actor_id)
        if threat is None:
            return None

        # Stop any existing plan - we manage movement directly
        controller.stop_plan(npc)

        game_map = controller.gw.game_map
        current_distance = ranges.calculate_distance(npc.x, npc.y, threat.x, threat.y)

        # Evaluate all adjacent tiles, pick the one that maximizes distance
        # from threat while minimizing hazard cost.
        candidates: list[tuple[int, int, int, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = npc.x + dx
                ty = npc.y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                if not game_map.walkable[tx, ty]:
                    continue
                actor_at = controller.gw.get_actor_at_location(tx, ty)
                if actor_at and actor_at.blocks_movement and actor_at is not npc:
                    continue

                dist = ranges.calculate_distance(tx, ty, threat.x, threat.y)
                if dist <= current_distance:
                    continue  # Must increase distance

                tile_id = int(game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)

                damage_per_turn = getattr(actor_at, "damage_per_turn", 0)
                if actor_at and damage_per_turn > 0:
                    fire_cost = HAZARD_BASE_COST + damage_per_turn
                    hazard_cost = max(hazard_cost, fire_cost)

                step_cost = 1 if (dx == 0 or dy == 0) else 2
                candidates.append((dx, dy, dist, hazard_cost + step_cost))

        if not candidates:
            # No escape route - goal fails
            self.state = GoalState.FAILED
            return None

        # Sort: maximize distance, minimize hazard cost
        candidates.sort(key=lambda c: (-c[2], c[3]))
        dx, dy, _, _ = candidates[0]
        return MoveIntent(controller, npc, dx, dy)


class PatrolGoal(Goal):
    """Patrol between a set of waypoints in a cycle.

    The NPC walks to each waypoint in order, cycling back to the first
    after reaching the last. Uses ActionPlans for pathfinding to each
    waypoint.
    """

    def __init__(self, waypoints: list[tuple[int, int]]) -> None:
        super().__init__(goal_id="patrol")
        if len(waypoints) < 2:
            msg = "PatrolGoal requires at least 2 waypoints"
            raise ValueError(msg)
        self.waypoints = waypoints
        self._current_waypoint_index: int = 0
        self._laps_completed: int = 0

        # Cached for progress calculation. Set on first get_next_action /
        # evaluate_completion call so the progress property can compute
        # spatial distance without method arguments.
        self._npc: NPC | None = None
        self._initial_waypoint_distance: float = 0.0

    @property
    def progress(self) -> float:
        """Progress through current lap, including spatial approach to waypoint.

        Combines waypoint-level progress (which waypoint are we heading to)
        with spatial progress (how close are we to that waypoint). This gives
        smooth 0.0-1.0 values instead of jumping in discrete steps.
        """
        from brileta.game import ranges

        total = len(self.waypoints)
        if self._npc is None or self._initial_waypoint_distance <= 0:
            # No spatial info yet - fall back to waypoint index only
            return self._current_waypoint_index / total

        target = self.waypoints[self._current_waypoint_index]
        current_dist = ranges.calculate_distance(
            self._npc.x, self._npc.y, target[0], target[1]
        )
        # How far through the current waypoint leg (0.0 = just started, 1.0 = arrived)
        leg_progress = 1.0 - min(1.0, current_dist / self._initial_waypoint_distance)
        return (self._current_waypoint_index + leg_progress) / total

    def get_base_score(self) -> float:
        return 0.25

    def get_considerations(self) -> list[Consideration]:
        """Patrol scores inversely to threat - patrol when safe."""
        return [
            Consideration(
                "threat_level",
                ResponseCurve(ResponseCurveType.INVERSE),
            ),
        ]

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Patrols don't complete on their own - they run until interrupted.

        Caches the NPC reference so the progress property can compute spatial
        distance to the current waypoint without method arguments.
        """
        self._npc = npc

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Move toward the current waypoint, advance when reached."""
        from brileta.game import ranges
        from brileta.game.action_plan import WalkToPlan

        self._npc = npc

        target = self.waypoints[self._current_waypoint_index]
        distance = ranges.calculate_distance(npc.x, npc.y, target[0], target[1])

        if distance == 0:
            # Reached waypoint - advance to next
            self._current_waypoint_index = (self._current_waypoint_index + 1) % len(
                self.waypoints
            )
            if self._current_waypoint_index == 0:
                self._laps_completed += 1
            target = self.waypoints[self._current_waypoint_index]
            # Record starting distance for the new leg so progress is smooth
            distance = ranges.calculate_distance(npc.x, npc.y, target[0], target[1])
            self._initial_waypoint_distance = float(distance)

        # First time heading to this waypoint - record the starting distance
        if self._initial_waypoint_distance <= 0:
            self._initial_waypoint_distance = float(distance) if distance > 0 else 1.0

        # Use the ActionPlan system to pathfind to the waypoint.
        # If an active plan already targets this waypoint, let it continue.
        plan = npc.active_plan
        if plan is not None and plan.context.target_position == target:
            return None  # Let the existing plan continue via TurnManager

        # Start a new plan to walk to the current waypoint
        controller.start_plan(npc, WalkToPlan, target_position=target)
        return None  # Plan will generate intents via TurnManager
