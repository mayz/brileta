"""Patrol behavior: PatrolAction + PatrolGoal + WaypointRouteGoal.

PatrolAction is the utility-system entry point. When it wins scoring,
AIComponent creates a PatrolGoal for cycling between waypoints.

WaypointRouteGoal is the shared base for route-following goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta.game.actors.ai.goals import Goal, GoalState
from brileta.game.actors.ai.utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    is_no_threat,
)
from brileta.types import WorldTilePos
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC

_rng = rng.get("npc.ai.actions")

# How far from the NPC's current position to pick patrol waypoints.
_PATROL_RADIUS = 6

# How many candidate waypoints to pick for a patrol route.
_PATROL_WAYPOINT_COUNT = 3


class PatrolAction(UtilityAction):
    """Patrol nearby when no threat is present.

    When this action wins scoring and the NPC has no active PatrolGoal,
    creates one with random walkable waypoints near the NPC. If a
    PatrolGoal already exists, ContinueGoalAction handles continuation
    so this action won't create duplicates.

    Preconditions: no threat present. Scores inversely to threat.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [is_no_threat]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("threat_level", ResponseCurve(ResponseCurveType.INVERSE))
    ]

    def __init__(self, base_score: float = 0.25) -> None:
        super().__init__(
            action_id="patrol",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in AIComponent.
        # This fallback should not be reached in normal flow.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a PatrolGoal with random nearby waypoints and start patrolling.

        Picks 2-3 random walkable tiles within _PATROL_RADIUS of the NPC's
        current position as waypoints. Uses the game_map.walkable array to
        validate tiles.
        """
        waypoints = _pick_patrol_waypoints(context.controller, actor)
        if len(waypoints) < 2:
            # Not enough walkable space - fall back to doing nothing
            return None

        goal = PatrolGoal(waypoints, preconditions=self.preconditions)
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        # If the goal immediately failed, clear it so the NPC doesn't
        # waste a tick holding a dead goal.
        if goal.is_complete:
            actor.current_goal = None

        return intent


# ---------------------------------------------------------------------------
# WaypointRouteGoal: shared route-following core
# ---------------------------------------------------------------------------


class WaypointRouteGoal(Goal):
    """Shared route-following core for patrol-style waypoint goals.

    This centralizes waypoint memory, progression, progress tracking, and
    ActionPlan reuse so waypoint-route goals differ only by route policy.
    """

    def __init__(
        self,
        goal_id: str,
        waypoints: list[WorldTilePos],
        *,
        minimum_waypoints: int = 2,
        preconditions: list | None = None,
    ) -> None:
        super().__init__(goal_id=goal_id, preconditions=preconditions)
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

    def _replace_route(self, waypoints: list[WorldTilePos]) -> None:
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

    def __init__(
        self,
        waypoints: list[WorldTilePos],
        preconditions: list | None = None,
    ) -> None:
        super().__init__(
            goal_id="patrol",
            waypoints=waypoints,
            minimum_waypoints=2,
            preconditions=preconditions,
        )

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


# ---------------------------------------------------------------------------
# Waypoint helpers
# ---------------------------------------------------------------------------


def _pick_patrol_waypoints(controller: Controller, actor: NPC) -> list[WorldTilePos]:
    """Pick random walkable tiles near the actor for patrol waypoints.

    Collects all walkable tiles within _PATROL_RADIUS of the actor,
    then samples _PATROL_WAYPOINT_COUNT of them.
    """
    return _pick_roam_waypoints(
        controller,
        actor,
        radius=_PATROL_RADIUS,
        waypoint_count=_PATROL_WAYPOINT_COUNT,
        require_unblocked=True,
    )


def _sample_roam_waypoints(
    candidates: list[WorldTilePos], waypoint_count: int
) -> list[WorldTilePos]:
    """Return up to ``waypoint_count`` sampled candidate waypoints."""
    if len(candidates) <= waypoint_count:
        return list(candidates)
    return _rng.sample(candidates, waypoint_count)


def _pick_roam_waypoints(
    controller: Controller,
    actor: NPC,
    *,
    radius: int,
    waypoint_count: int,
    require_unblocked: bool,
) -> list[WorldTilePos]:
    """Pick walkable roam waypoints, preferring safe tiles when possible."""
    safe_candidates, hazardous_candidates = _collect_roam_candidates(
        controller, actor, radius=radius, require_unblocked=require_unblocked
    )
    candidates = safe_candidates + hazardous_candidates
    return _sample_roam_waypoints(candidates, waypoint_count)


def _collect_roam_candidates(
    controller: Controller,
    actor: NPC,
    radius: int,
    *,
    require_unblocked: bool,
) -> tuple[list[WorldTilePos], list[WorldTilePos]]:
    """Collect nearby walkable roam candidates, partitioned by hazard.

    When ``require_unblocked`` is False, actor-occupied tiles are still included
    (only structural blocks like walls and doors are filtered).
    """
    from brileta.environment.tile_types import is_tile_hazardous
    from brileta.game.enums import StepBlock
    from brileta.util.pathfinding import probe_step

    game_map = controller.gw.game_map
    safe_candidates: list[WorldTilePos] = []
    hazardous_candidates: list[WorldTilePos] = []

    # Actor-occupancy blocks that can be ignored when require_unblocked is False
    _ACTOR_BLOCKS = (StepBlock.BLOCKED_BY_ACTOR, StepBlock.BLOCKED_BY_CONTAINER)

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            tx = actor.x + dx
            ty = actor.y + dy
            block = probe_step(game_map, controller.gw, tx, ty, exclude_actor=actor)
            if block is not None and (require_unblocked or block not in _ACTOR_BLOCKS):
                continue
            if is_tile_hazardous(int(game_map.tiles[tx, ty])):
                hazardous_candidates.append((tx, ty))
            else:
                safe_candidates.append((tx, ty))

    return safe_candidates, hazardous_candidates
