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
    WaypointRouteGoal: Shared core for route-following waypoint goals.
    PatrolGoal: Cycle between fixed waypoints.
    WanderGoal: Wander via heading + step-budget segments with stuck recovery.
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
from brileta.util import rng

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

_rng = rng.get("npc.goals")


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
_FLEE_SAFE_DISTANCE = 50


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


class WaypointRouteGoal(Goal):
    """Shared route-following core for patrol-style waypoint goals.

    This centralizes waypoint memory, progression, progress tracking, and
    ActionPlan reuse so waypoint-route goals differ only by route policy.
    """

    def __init__(
        self,
        goal_id: str,
        waypoints: list[tuple[int, int]],
        *,
        minimum_waypoints: int = 2,
    ) -> None:
        super().__init__(goal_id=goal_id)
        if len(waypoints) < minimum_waypoints:
            msg = (
                f"{self.__class__.__name__} requires at least "
                f"{minimum_waypoints} waypoints"
            )
            raise ValueError(msg)
        self.waypoints = waypoints
        self._current_waypoint_index: int = 0
        self._laps_completed: int = 0

        # Cached for progress calculation. Set on first get_next_action /
        # evaluate_completion call so progress can include spatial approach.
        self._npc: NPC | None = None
        self._initial_waypoint_distance: float = 0.0

    @property
    def progress(self) -> float:
        """Progress through the current waypoint lap, including approach."""
        from brileta.game import ranges

        total = len(self.waypoints)
        if self._npc is None or self._initial_waypoint_distance <= 0:
            return self._current_waypoint_index / total

        target = self.waypoints[self._current_waypoint_index]
        current_dist = ranges.calculate_distance(
            self._npc.x, self._npc.y, target[0], target[1]
        )
        leg_progress = 1.0 - min(1.0, current_dist / self._initial_waypoint_distance)
        return (self._current_waypoint_index + leg_progress) / total

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Waypoint-route goals don't auto-complete; cache NPC for progress."""
        self._npc = npc

    def _advance_waypoint(self) -> bool:
        """Advance to the next waypoint and return True when a lap completes."""
        self._current_waypoint_index = (self._current_waypoint_index + 1) % len(
            self.waypoints
        )
        completed_lap = self._current_waypoint_index == 0
        if completed_lap:
            self._laps_completed += 1
        self._initial_waypoint_distance = 0.0
        return completed_lap

    def _replace_route(self, waypoints: list[tuple[int, int]]) -> None:
        """Swap to a newly generated route and reset waypoint-leg state."""
        if len(waypoints) < 2:
            self.state = GoalState.FAILED
            return
        self.waypoints = waypoints
        self._current_waypoint_index = 0
        self._initial_waypoint_distance = 0.0

    def _on_waypoint_advanced(
        self,
        npc: NPC,
        controller: Controller,
        *,
        completed_lap: bool,
    ) -> None:
        """Optional hook for subclasses when waypoint progression changes."""
        return

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Move toward current waypoint and advance/refresh route when reached."""
        from brileta.game import ranges
        from brileta.game.action_plan import WalkToPlan

        self._npc = npc

        target = self.waypoints[self._current_waypoint_index]
        distance = ranges.calculate_distance(npc.x, npc.y, target[0], target[1])

        if distance == 0:
            completed_lap = self._advance_waypoint()
            self._on_waypoint_advanced(npc, controller, completed_lap=completed_lap)
            if self.state is not GoalState.ACTIVE:
                return None

            target = self.waypoints[self._current_waypoint_index]
            distance = ranges.calculate_distance(npc.x, npc.y, target[0], target[1])

        if self._initial_waypoint_distance <= 0:
            self._initial_waypoint_distance = float(distance) if distance > 0 else 1.0

        plan = npc.active_plan
        if plan is not None and plan.context.target_position == target:
            return None

        controller.start_plan(npc, WalkToPlan, target_position=target)
        return None


class PatrolGoal(WaypointRouteGoal):
    """Patrol between a fixed set of waypoints in a cycle."""

    def __init__(self, waypoints: list[tuple[int, int]]) -> None:
        super().__init__(goal_id="patrol", waypoints=waypoints, minimum_waypoints=2)

    def get_base_score(self) -> float:
        return 0.25

    def get_considerations(self) -> list[Consideration]:
        """Patrol scores inversely to threat: patrol when safe."""
        return [
            Consideration(
                "threat_level",
                ResponseCurve(ResponseCurveType.INVERSE),
            ),
        ]


class WanderGoal(Goal):
    """Wander using heading segments instead of precomputed waypoint routes.

    The goal picks a heading and a random step budget ``N`` for that segment.
    Each tick it tries to move forward; if blocked it attempts sidesteps and
    shallow turns. When the budget is exhausted or the actor is repeatedly
    stuck, it repicks heading + budget.
    """

    _DIRECTIONS: tuple[tuple[int, int], ...] = (
        (1, 0),  # East
        (1, 1),  # Southeast
        (0, 1),  # South
        (-1, 1),  # Southwest
        (-1, 0),  # West
        (-1, -1),  # Northwest
        (0, -1),  # North
        (1, -1),  # Northeast
    )

    def __init__(
        self,
        *,
        minimum_segment_steps: int = 4,
        maximum_segment_steps: int = 12,
        max_stuck_turns: int = 2,
        heading_jitter_chance: float = 0.15,
    ) -> None:
        super().__init__(goal_id="wander")
        if minimum_segment_steps <= 0:
            msg = "minimum_segment_steps must be positive"
            raise ValueError(msg)
        if maximum_segment_steps < minimum_segment_steps:
            msg = "maximum_segment_steps must be >= minimum_segment_steps"
            raise ValueError(msg)
        if max_stuck_turns <= 0:
            msg = "max_stuck_turns must be positive"
            raise ValueError(msg)
        if not 0.0 <= heading_jitter_chance <= 1.0:
            msg = "heading_jitter_chance must be in [0.0, 1.0]"
            raise ValueError(msg)

        self._minimum_segment_steps = minimum_segment_steps
        self._maximum_segment_steps = maximum_segment_steps
        self._max_stuck_turns = max_stuck_turns
        self._heading_jitter_chance = heading_jitter_chance

        self._heading: tuple[int, int] | None = None
        self._segment_length: int = 0
        self._steps_remaining: int = 0
        self._segment_steps_taken: int = 0
        self._stuck_turns: int = 0
        self._last_attempt_origin: tuple[int, int] | None = None

    def get_base_score(self) -> float:
        return 0.18

    def get_considerations(self) -> list[Consideration]:
        """Wander scores when threat is absent and disposition is approachable."""
        return [
            Consideration(
                "threat_level",
                ResponseCurve(ResponseCurveType.INVERSE),
            ),
            Consideration(
                "disposition",
                ResponseCurve(ResponseCurveType.LINEAR),
            ),
        ]

    @property
    def progress(self) -> float:
        """Progress through the current heading segment."""
        if self._segment_length <= 0:
            return 0.0
        return min(1.0, self._segment_steps_taken / self._segment_length)

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Wander is open-ended and does not auto-complete."""
        return

    @classmethod
    def _rotate_heading(cls, heading: tuple[int, int], offset: int) -> tuple[int, int]:
        """Return heading rotated by ``offset`` steps in the direction wheel."""
        idx = cls._DIRECTIONS.index(heading)
        return cls._DIRECTIONS[(idx + offset) % len(cls._DIRECTIONS)]

    def _is_step_walkable(
        self, npc: NPC, controller: Controller, dx: int, dy: int
    ) -> bool:
        """Return True when a step destination is in bounds, walkable, and unblocked."""
        game_map = controller.gw.game_map
        tx = npc.x + dx
        ty = npc.y + dy
        if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
            return False
        if not game_map.walkable[tx, ty]:
            return False
        blocker = controller.gw.get_actor_at_location(tx, ty)
        return blocker is None or blocker is npc or not blocker.blocks_movement

    def _pick_segment_heading(
        self, npc: NPC, controller: Controller
    ) -> tuple[int, int] | None:
        """Pick a random walkable heading, preferring safe tiles."""
        from brileta.environment.tile_types import get_hazard_cost

        safe: list[tuple[int, int]] = []
        hazardous: list[tuple[int, int]] = []
        game_map = controller.gw.game_map

        for dx, dy in self._DIRECTIONS:
            if not self._is_step_walkable(npc, controller, dx, dy):
                continue
            tile_id = int(game_map.tiles[npc.x + dx, npc.y + dy])
            if get_hazard_cost(tile_id) > 1:
                hazardous.append((dx, dy))
            else:
                safe.append((dx, dy))

        pool = safe if safe else hazardous
        if not pool:
            return None
        return _rng.choice(pool)

    def _start_new_segment(self, npc: NPC, controller: Controller) -> bool:
        """Pick a new heading and segment budget.

        Returns:
            True if a valid segment was started, False otherwise.
        """
        heading = self._pick_segment_heading(npc, controller)
        if heading is None:
            self._heading = None
            self._segment_length = 0
            self._steps_remaining = 0
            self._segment_steps_taken = 0
            return False

        self._heading = heading
        self._segment_length = _rng.randint(
            self._minimum_segment_steps, self._maximum_segment_steps
        )
        self._steps_remaining = self._segment_length
        self._segment_steps_taken = 0
        self._stuck_turns = 0
        return True

    def _apply_heading_jitter(self) -> None:
        """Occasionally nudge heading by 45 degrees to avoid straight-line drift."""
        if self._heading is None:
            return
        if _rng.random() >= self._heading_jitter_chance:
            return
        offset = _rng.choice((-1, 1))
        self._heading = self._rotate_heading(self._heading, offset)

    def _choose_step(self, npc: NPC, controller: Controller) -> tuple[int, int] | None:
        """Choose the next step from forward/sidestep/turn fallback candidates."""
        from brileta.environment.tile_types import get_hazard_cost

        if self._heading is None:
            return None

        game_map = controller.gw.game_map

        side_offsets = [-2, 2]  # 90-degree sidesteps
        _rng.shuffle(side_offsets)
        shallow_turn_offsets = [-1, 1]  # 45-degree turns
        _rng.shuffle(shallow_turn_offsets)

        # Priority order:
        # 1) Forward
        # 2) Sidestep left/right
        # 3) Shallow turn left/right
        # 4) Wider turns then reverse as a last resort
        groups: list[list[int]] = [
            [0],
            side_offsets,
            shallow_turn_offsets,
            [-3, 3],
            [4],
        ]

        for offsets in groups:
            candidates: list[tuple[tuple[int, int], int]] = []
            for offset in offsets:
                direction = self._rotate_heading(self._heading, offset)
                dx, dy = direction
                if not self._is_step_walkable(npc, controller, dx, dy):
                    continue
                tile_id = int(game_map.tiles[npc.x + dx, npc.y + dy])
                hazard_cost = get_hazard_cost(tile_id)
                candidates.append((direction, hazard_cost))

            if not candidates:
                continue

            # Within a priority group, prefer lower hazard; randomize ties.
            min_cost = min(cost for _, cost in candidates)
            safest = [direction for direction, cost in candidates if cost == min_cost]
            return _rng.choice(safest)

        return None

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Return a one-step movement intent for the current wander segment."""
        from brileta.game.actions.movement import MoveIntent

        # Wander now uses direct step intents, not ActionPlans.
        controller.stop_plan(npc)

        current_pos = (npc.x, npc.y)

        # Resolve whether last tick's attempted move changed position.
        if self._last_attempt_origin is not None:
            if current_pos == self._last_attempt_origin:
                self._stuck_turns += 1
            else:
                self._stuck_turns = 0
            self._last_attempt_origin = None

        needs_new_segment = (
            self._heading is None
            or self._steps_remaining <= 0
            or self._stuck_turns >= self._max_stuck_turns
        )
        if needs_new_segment and not self._start_new_segment(npc, controller):
            return None

        self._apply_heading_jitter()
        step = self._choose_step(npc, controller)
        if step is None:
            self._stuck_turns += 1
            if self._stuck_turns >= self._max_stuck_turns:
                self._heading = None
                self._steps_remaining = 0
            return None

        dx, dy = step
        self._heading = (dx, dy)
        self._steps_remaining = max(0, self._steps_remaining - 1)
        self._segment_steps_taken += 1
        self._last_attempt_origin = current_pos
        return MoveIntent(controller, npc, dx, dy)
