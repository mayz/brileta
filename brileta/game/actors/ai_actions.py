"""Concrete utility actions and helper predicates for NPC AI behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.game import ranges
from brileta.game.action_plan import WalkToPlan
from brileta.game.actors.utility import (
    Action as UtilityAction,
)
from brileta.game.actors.utility import Consideration, UtilityContext
from brileta.types import WorldTilePos
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent

    from . import NPC

_rng = rng.get("npc.ai.actions")

# Hostile precondition threshold in normalized disposition space.
# Matches the current ai.HOSTILE_UPPER of -51 mapped via (value + 100) / 200.
_HOSTILE_DISPOSITION_THRESHOLD = (-51 + 100) / 200.0

# How far from the NPC's current position to pick patrol waypoints.
_PATROL_RADIUS = 6

# How many candidate waypoints to pick for a patrol route.
_PATROL_WAYPOINT_COUNT = 3

# Wander tuning: sustained stroll with occasional pauses and direction changes.
_WANDER_MAX_STUCK_TURNS = 2
_WANDER_HEADING_JITTER_CHANCE = 0.15
_WANDER_PAUSE_CHANCE = 0.20
_WANDER_MIN_LINGER_TURNS = 2
_WANDER_MAX_LINGER_TURNS = 6
_WANDER_NEW_HEADING_CHANCE = 0.05
_WANDER_SPEED_MIN = 0.60
_WANDER_SPEED_MAX = 0.90


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------


def _is_threat_present(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is non-zero."""
    return context.threat_level > 0.0


def _is_no_threat(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is zero."""
    return context.threat_level <= 0.0


def _has_escape_route(context: UtilityContext) -> bool:
    """Precondition: returns True when a valid flee step exists."""
    return context.has_escape_route


def _is_hostile(context: UtilityContext) -> bool:
    """Precondition: only initiate combat when hostile toward target.

    This is a game rule, not a scoring gate. Non-hostile NPCs do not start
    fights regardless of other considerations. Disposition scoring curves
    handle the intensity of hostile behavior (e.g., attack vs flee).
    """
    return context.disposition <= _HOSTILE_DISPOSITION_THRESHOLD


def _is_not_hostile(context: UtilityContext) -> bool:
    """Precondition: only allow non-hostile social behaviors."""
    return context.disposition > _HOSTILE_DISPOSITION_THRESHOLD


# ---------------------------------------------------------------------------
# Utility Actions
# ---------------------------------------------------------------------------


class AttackAction(UtilityAction):
    """Attack the target when adjacent, or pathfind toward them."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="attack",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        from brileta.game.actions.combat import AttackIntent

        controller = context.controller
        actor = context.actor
        target = context.target

        if context.can_attack:
            controller.stop_plan(actor)
            return AttackIntent(controller, actor, target)

        plan = actor.active_plan
        if plan is not None and plan.context.target_position is not None:
            gx, gy = plan.context.target_position
            if (
                ranges.calculate_distance(target.x, target.y, gx, gy) == 1
                and controller.gw.game_map.walkable[gx, gy]
            ):
                return None

        if context.best_attack_destination and controller.start_plan(
            actor, WalkToPlan, target_position=context.best_attack_destination
        ):
            return None

        return None


class FleeAction(UtilityAction):
    """Flee from the target when health is low."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="flee",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Single-step flee fallback (used when goal system isn't active)."""
        from brileta.game.actions.movement import MoveIntent

        if context.best_flee_step is None:
            return None

        dx, dy = context.best_flee_step
        context.controller.stop_plan(context.actor)
        return MoveIntent(context.controller, context.actor, dx, dy)

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a FleeGoal and return the first flee action.

        Called by AIComponent when FleeAction wins scoring. Creates a persistent
        FleeGoal so the NPC continues fleeing across multiple turns until safe,
        rather than re-evaluating from scratch each tick.
        """
        from brileta.game.actors.goals import FleeGoal

        # Create and assign a flee goal targeting the current threat.
        goal = FleeGoal(threat_actor_id=context.target.actor_id)
        actor.current_goal = goal

        # Evaluate completion once (sets initial distance tracking)
        goal.evaluate_completion(actor, context.controller)
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        # If the goal immediately failed (e.g., cornered), clear it so the
        # NPC doesn't waste a tick holding a dead goal.
        if goal.is_complete:
            actor.current_goal = None

        return intent


class AvoidAction(UtilityAction):
    """Move one step away from the target. Used by unfriendly NPCs."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="avoid",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Move away from the target using the best flee step."""
        from brileta.game.actions.movement import MoveIntent

        if context.best_flee_step is None:
            return None

        dx, dy = context.best_flee_step
        context.controller.stop_plan(context.actor)
        return MoveIntent(context.controller, context.actor, dx, dy)


class WatchAction(UtilityAction):
    """Stay put, facing the target. Used by wary/unfriendly NPCs."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="watch",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Do nothing - stay in place, watching.
        return None


class IdleAction(UtilityAction):
    """Do nothing this turn. Always a baseline fallback."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
    ) -> None:
        super().__init__(
            action_id="idle",
            base_score=base_score,
            considerations=considerations,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        return None


class WanderAction(UtilityAction):
    """Create/continue a strolling wander goal when no threat is present."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="wander",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in AIComponent.
        # This fallback should not be reached in normal flow.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a stroll-style WanderGoal and return the first intent."""
        from brileta.game.actors.goals import WanderGoal

        goal = WanderGoal(
            pause_chance=_WANDER_PAUSE_CHANCE,
            minimum_linger_turns=_WANDER_MIN_LINGER_TURNS,
            maximum_linger_turns=_WANDER_MAX_LINGER_TURNS,
            new_heading_chance=_WANDER_NEW_HEADING_CHANCE,
            max_stuck_turns=_WANDER_MAX_STUCK_TURNS,
            heading_jitter_chance=_WANDER_HEADING_JITTER_CHANCE,
            speed_min_multiplier=_WANDER_SPEED_MIN,
            speed_max_multiplier=_WANDER_SPEED_MAX,
        )
        actor.current_goal = goal
        goal.apply_wander_speed(actor)
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        if goal.is_complete:
            actor.current_goal = None

        return intent


class PatrolAction(UtilityAction):
    """Patrol nearby when no threat is present.

    When this action wins scoring and the NPC has no active PatrolGoal,
    creates one with random walkable waypoints near the NPC. If a
    PatrolGoal already exists, ContinueGoalAction handles continuation
    so this action won't create duplicates.
    """

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="patrol",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
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
        from brileta.game.actors.goals import PatrolGoal

        waypoints = _pick_patrol_waypoints(context.controller, actor)
        if len(waypoints) < 2:
            # Not enough walkable space - fall back to doing nothing
            return None

        goal = PatrolGoal(waypoints)
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        # If the goal immediately failed, clear it so the NPC doesn't
        # waste a tick holding a dead goal.
        if goal.is_complete:
            actor.current_goal = None

        return intent


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

    ``get_hazard_cost()`` returns 1 for safe tiles and >1 for hazardous tiles.
    When ``require_unblocked`` is False, actor-occupied tiles are still included
    (only structural blocks like walls and doors are filtered).
    """
    from brileta.environment.tile_types import get_hazard_cost
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
            if get_hazard_cost(int(game_map.tiles[tx, ty])) > 1:
                hazardous_candidates.append((tx, ty))
            else:
                safe_candidates.append((tx, ty))

    return safe_candidates, hazardous_candidates
