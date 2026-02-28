"""Flee behavior: FleeAction + FleeGoal.

FleeAction is the utility-system entry point. When it wins scoring,
AIComponent creates a FleeGoal for persistent multi-turn fleeing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from brileta.game.actors.ai.goals import Goal, GoalState
from brileta.game.actors.ai.utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    is_threat_present,
    resolve_flee_from,
)
from brileta.types import DIRECTIONS, ActorId, Direction

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC, Actor

# How far the NPC should be from the threat before considering itself safe.
_FLEE_SAFE_DISTANCE = 50


@dataclass(slots=True)
class FleeCandidate:
    """One potential flee step scored for tactical desirability."""

    direction: Direction
    distance_after: float
    hazard_score: int
    step_cost: int


def collect_flee_candidates(
    controller: Controller,
    actor: NPC,
    target: Actor,
) -> list[FleeCandidate]:
    """Enumerate passable adjacent tiles that don't move toward the threat.

    Prefers tiles that increase distance. If none exist (e.g., against a
    wall), includes lateral tiles (same distance) so the NPC can slide
    along obstacles to find an opening.
    """
    from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
    from brileta.game import ranges
    from brileta.util.pathfinding import probe_step

    game_map = controller.gw.game_map
    current_distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
    increasing: list[FleeCandidate] = []
    lateral: list[FleeCandidate] = []

    for dx, dy in DIRECTIONS:
        tx = actor.x + dx
        ty = actor.y + dy
        block_reason = probe_step(
            game_map,
            controller.gw,
            tx,
            ty,
            exclude_actor=actor,
            can_open_doors=actor.can_open_doors,
        )
        if block_reason is not None:
            continue

        distance_after = ranges.calculate_distance(tx, ty, target.x, target.y)
        if distance_after < current_distance:
            continue  # Moving toward threat is never acceptable.

        tile_id = int(game_map.tiles[tx, ty])
        hazard_cost = get_hazard_cost(tile_id)
        actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
        damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
        if actor_at_tile and damage_per_turn > 0:
            fire_cost = HAZARD_BASE_COST + damage_per_turn
            hazard_cost = max(hazard_cost, fire_cost)

        step_cost = 1 if (dx == 0 or dy == 0) else 2
        candidate = FleeCandidate(
            direction=(dx, dy),
            distance_after=distance_after,
            hazard_score=hazard_cost + step_cost,
            step_cost=step_cost,
        )
        if distance_after > current_distance:
            increasing.append(candidate)
        else:
            lateral.append(candidate)

    return increasing or lateral


class FleeAction(UtilityAction):
    """Flee from the target when health is low.

    Preconditions: threat must be present.
    Scores higher with lower health, higher threat, and lower disposition.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [is_threat_present]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration(
            "health_percent", ResponseCurve(ResponseCurveType.INVERSE), weight=2.0
        ),
        Consideration("threat_level", ResponseCurve(ResponseCurveType.LINEAR)),
        Consideration("has_escape_route", ResponseCurve(ResponseCurveType.STEP)),
        # Flee scores higher when hostile (in danger of combat).
        Consideration("disposition", ResponseCurve(ResponseCurveType.INVERSE)),
    ]

    def __init__(
        self,
        base_score: float = 1.0,
        preconditions: list[Precondition] | None = None,
        considerations: list[Consideration] | None = None,
    ) -> None:
        super().__init__(
            action_id="flee",
            base_score=base_score,
            considerations=(
                considerations if considerations is not None else self.CONSIDERATIONS
            ),
            preconditions=(
                preconditions if preconditions is not None else self.PRECONDITIONS
            ),
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
        # Context usually provides flee_from directly; the fallback keeps
        # manually-constructed test contexts working.
        flee_from = context.flee_from or resolve_flee_from(
            context.target, context.threat_level, context.threat_source
        )
        if flee_from is None:
            return None

        # Create and assign a flee goal targeting the threat.
        # Pass our preconditions so ContinueGoalAction enforces the same
        # eligibility rules (e.g., threat must still be present). Also pass
        # considerations/base_score so continuation scoring mirrors the
        # originating flee action (e.g., sapient flee uses its custom urgency).
        goal = FleeGoal(
            threat_actor_id=flee_from.actor_id,
            preconditions=self.preconditions,
            considerations=self.considerations,
            base_score=self.base_score,
        )
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


class FleeGoal(Goal):
    """Flee from a threat until safely distant, then reassess.

    The FleeGoal moves the NPC away from a threat actor. Each tick it picks
    the best adjacent tile that increases distance from the threat. When the
    NPC reaches a safe distance or the threat is no longer present/alive,
    the goal completes.

    Progress is based on distance traveled away from the threat relative to
    the safe distance threshold.
    """

    # Fallback profile used when a goal is constructed directly rather than
    # created from a winning FleeAction (mostly tests).
    DEFAULT_CONSIDERATIONS: ClassVar[list[Consideration]] = list(
        FleeAction.CONSIDERATIONS
    )

    def __init__(
        self,
        threat_actor_id: ActorId,
        preconditions: list[Precondition] | None = None,
        considerations: list[Consideration] | None = None,
        base_score: float = 1.0,
    ) -> None:
        super().__init__(goal_id="flee", preconditions=preconditions)
        self.threat_actor_id = threat_actor_id
        self._start_distance: int | None = None
        self._current_distance: int | None = None
        self._considerations = (
            list(considerations)
            if considerations is not None
            else list(self.DEFAULT_CONSIDERATIONS)
        )
        self._base_score = base_score

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
        return self._base_score

    def get_considerations(self) -> list[Consideration]:
        """Return the continuation scoring profile for this flee goal."""
        return self._considerations

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
        from brileta.game.actions.movement import MoveIntent

        threat = controller.gw.get_actor_by_id(self.threat_actor_id)
        if threat is None:
            return None

        # Stop any existing plan - we manage movement directly.
        controller.stop_plan(npc)

        candidates = collect_flee_candidates(controller, npc, threat)
        if not candidates:
            # Truly cornered - no passable tile that doesn't move closer.
            self.state = GoalState.FAILED
            return None

        # Sort: maximize distance, minimize hazard cost + step cost.
        candidates.sort(key=lambda c: (-c.distance_after, c.hazard_score, c.step_cost))
        dx, dy = candidates[0].direction
        return MoveIntent(controller, npc, dx, dy)
