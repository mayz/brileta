"""Daily-routine behavior: RoutineAction + RoutineGoal.

RoutineAction is the utility-system entry point for settlement NPCs that have
a home and a workplace anchor (assigned during settlement generation). When it
wins scoring, AIComponent creates a RoutineGoal that drives the NPC between its
anchor by day and its home by night, milling near whichever it currently wants.

The routine is the *default* peaceful behavior: it outscores aimless wandering,
so residents visibly have somewhere to be. It shares the no-threat / not-hostile
preconditions with wander, so threats, flee, and combat always win over it - a
routine is a habit, never a cage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta import config
from brileta.game.actors.ai.goals import Goal
from brileta.game.actors.ai.utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    is_no_threat,
    is_not_hostile,
)
from brileta.types import Direction, WorldTilePos
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC

_rng = rng.get("npc.goals")

# The sun sits above the horizon for a DAY_FRACTION-wide window centered on noon
# (0.5). Dawn and dusk are its edges. NPCs work through the day and head home a
# little before dusk so the streets visibly empty while there is still light.
_HALF_DAY = config.DAY_FRACTION / 2.0
_DAWN = 0.5 - _HALF_DAY
_DUSK = 0.5 + _HALF_DAY
# When the workday ends. Set inside the daylight window so the walk home happens
# at dusk rather than in the dark.
_WORK_END = _DUSK - 0.10

# Per-NPC leave/return times are jittered by up to this much (time-of-day units,
# ~10% of a full day) so residents don't migrate as a synchronized flash mob.
# Sampled once per NPC at assignment and stored on npc.routine_offset.
# Personality will later replace this with trait-driven timing.
ROUTINE_SCHEDULE_JITTER = 0.10

# How close (Chebyshev tiles) counts as "at" the current target. Within this the
# NPC mills instead of pathfinding, so it clusters loosely rather than freezing
# on one exact tile.
_ARRIVAL_RADIUS = 2
# Per-tick chance to take a small idle step while milling at the target.
_MILL_STEP_CHANCE = 0.15

# While at the workplace anchor, per-tick chance to leave on a short errand to a
# public feature tile (a market/plaza spot or a doorfront) and then head back,
# so workers don't look frozen in place for the whole day.
_ERRAND_START_CHANCE = 0.02
# How many ticks to linger at the errand destination before returning.
_ERRAND_LINGER_MIN = 2
_ERRAND_LINGER_MAX = 5
# Give up an errand after this many consecutive travel ticks without moving.
# An unreachable or persistently blocked destination makes no progress, so
# without this the NPC would re-issue a doomed walk plan forever instead of
# returning to its routine. A moving NPC resets the counter every step, so
# legitimate long walks (and brief traffic jams) are unaffected.
_ERRAND_MAX_STUCK_TICKS = 4


def has_routine(context: UtilityContext) -> bool:
    """Precondition: the NPC has at least one routine anchor assigned."""
    actor = context.actor
    return actor.home_pos is not None or actor.anchor_pos is not None


def routine_target(time_of_day: float, npc: NPC) -> WorldTilePos | None:
    """Return where this NPC wants to be for the given time of day.

    Daytime (dawn until the workday ends) points at the workplace anchor;
    otherwise the NPC wants home. Each falls back to the other so an NPC with
    only one anchor assigned still behaves sensibly. The NPC's per-NPC
    routine_offset shifts both boundaries so leave/return times vary.
    """
    work_start = _DAWN + npc.routine_offset
    work_end = _WORK_END + npc.routine_offset
    if work_start <= time_of_day < work_end:
        return npc.anchor_pos or npc.home_pos
    return npc.home_pos or npc.anchor_pos


class RoutineAction(UtilityAction):
    """Create/continue a daily-routine goal when safe and routine-capable.

    Preconditions: no threat, NPC not hostile, and a routine anchor exists.
    Base score sits above wander so a resident pursues its routine instead of
    drifting aimlessly, while staying well below combat/flee scores.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [
        is_no_threat,
        is_not_hostile,
        has_routine,
    ]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("threat_level", ResponseCurve(ResponseCurveType.INVERSE)),
    ]

    def __init__(self, base_score: float = 0.4) -> None:
        super().__init__(
            action_id="routine",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in AIComponent.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a RoutineGoal and return its first intent."""
        goal = RoutineGoal(preconditions=self.preconditions)
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)
        if goal.is_complete:
            actor.current_goal = None
        return intent


class RoutineGoal(Goal):
    """Move to and linger at the NPC's time-appropriate routine target.

    Open-ended like wander: it never completes on its own. Each tick it picks
    the current target (anchor by day, home by night), pathfinds there via a
    WalkToPlan when far, and mills with occasional idle steps when close. While
    at the workplace it may occasionally slip out on a short errand to a public
    feature tile and then return, so workers don't look frozen all day.
    """

    def __init__(self, *, preconditions: list | None = None) -> None:
        super().__init__(goal_id="routine", preconditions=preconditions)
        # Active mid-workday errand: a temporary destination the NPC visits and
        # then returns from. None when not on an errand.
        self._errand_target: WorldTilePos | None = None
        self._errand_linger: int = 0
        # No-progress watchdog: the last position seen while traveling to the
        # errand, and how many consecutive travel ticks it has stayed there.
        self._errand_last_pos: WorldTilePos | None = None
        self._errand_stuck_ticks: int = 0

    def get_base_score(self) -> float:
        return 0.4

    def get_considerations(self) -> list[Consideration]:
        return [
            Consideration("threat_level", ResponseCurve(ResponseCurveType.INVERSE)),
        ]

    @property
    def progress(self) -> float:
        """Open-ended; a moderate persistence signal keeps it stable."""
        return 0.35

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Routines are open-ended and never self-complete."""
        return

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Walk to the routine target, mill there, or run/return from an errand."""
        # An active errand takes precedence until the NPC gets back.
        if self._errand_target is not None:
            return self._advance_errand(npc, controller)

        target = routine_target(controller.gw.clock.time_of_day, npc)
        if target is None:
            return None

        if not self._at_tile(npc, target):
            self._walk_to(npc, controller, target)
            return None

        # At the routine target. During the workday (target is the anchor), the
        # NPC may head out on a short errand instead of milling in place.
        controller.stop_plan(npc)
        if target == npc.anchor_pos and self._maybe_start_errand(npc, controller):
            return self._advance_errand(npc, controller)
        return self._mill(npc, controller)

    def _at_tile(self, npc: NPC, tile: WorldTilePos) -> bool:
        """Return True when the NPC is within the arrival radius of ``tile``."""
        from brileta.game import ranges

        return ranges.calculate_distance(npc.x, npc.y, tile[0], tile[1]) <= (
            _ARRIVAL_RADIUS
        )

    def _walk_to(self, npc: NPC, controller: Controller, tile: WorldTilePos) -> None:
        """(Re)issue a walk plan toward ``tile``, leaving an existing one alone."""
        from brileta.game.action_plan import WalkToPlan

        plan = npc.active_plan
        if plan is None or plan.context.target_position != tile:
            controller.start_plan(npc, WalkToPlan, target_position=tile)

    def _maybe_start_errand(self, npc: NPC, controller: Controller) -> bool:
        """Maybe pick a feature tile to run an errand to; return whether one began.

        Destinations come only from the settlement's errand pool (doorfronts and
        plaza), never arbitrary tiles, and must be beyond the arrival radius so
        the NPC actually travels.
        """
        pool = getattr(controller.gw, "routine_errand_pool", None)
        if not pool:
            return False
        if _rng.random() >= _ERRAND_START_CHANCE:
            return False

        destination = _rng.choice(pool)
        if self._at_tile(npc, destination):
            return False  # Too close to be a real trip; skip this tick.

        self._errand_target = destination
        self._errand_linger = _rng.randint(_ERRAND_LINGER_MIN, _ERRAND_LINGER_MAX)
        self._errand_last_pos = None
        self._errand_stuck_ticks = 0
        return True

    def _clear_errand(self) -> None:
        """Reset all errand state so the NPC resumes its normal routine."""
        self._errand_target = None
        self._errand_linger = 0
        self._errand_last_pos = None
        self._errand_stuck_ticks = 0

    def _advance_errand(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Walk to the errand destination, linger, then clear it to head home/work."""
        destination = self._errand_target
        if destination is None:
            return None

        if not self._at_tile(npc, destination):
            # Abandon the errand if we've made no progress for too long - the
            # destination is unreachable or persistently blocked, and re-issuing
            # the walk plan forever would otherwise strand the NPC mid-errand.
            pos = (npc.x, npc.y)
            if pos == self._errand_last_pos:
                self._errand_stuck_ticks += 1
            else:
                self._errand_stuck_ticks = 0
            self._errand_last_pos = pos
            if self._errand_stuck_ticks >= _ERRAND_MAX_STUCK_TICKS:
                controller.stop_plan(npc)
                self._clear_errand()
                return None

            self._walk_to(npc, controller, destination)
            return None

        # Arrived: linger a few ticks, then end the errand. The next tick's
        # normal routine logic walks the NPC back to its anchor.
        controller.stop_plan(npc)
        if self._errand_linger > 0:
            self._errand_linger -= 1
            return None
        self._clear_errand()
        return None

    def _mill(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Occasionally take a small safe step near the target, else idle.

        Drift is self-correcting: once a mill step pushes the NPC past
        _ARRIVAL_RADIUS, the next tick pathfinds it back toward the target.
        """
        from brileta.game.actions.movement import MoveIntent

        if _rng.random() >= _MILL_STEP_CHANCE:
            return None

        step = self._random_safe_step(npc, controller)
        if step is None:
            return None
        dx, dy = step
        return MoveIntent(controller, npc, dx, dy)

    _DIRECTIONS: ClassVar[tuple[Direction, ...]] = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )

    def _random_safe_step(self, npc: NPC, controller: Controller) -> Direction | None:
        """Return a random walkable, non-hazardous adjacent step, or None."""
        from brileta.environment.tile_types import is_tile_hazardous
        from brileta.util.pathfinding import probe_step

        game_map = controller.gw.game_map
        candidates: list[Direction] = []
        for dx, dy in self._DIRECTIONS:
            tx, ty = npc.x + dx, npc.y + dy
            if (
                probe_step(
                    game_map,
                    controller.gw,
                    tx,
                    ty,
                    exclude_actor=npc,
                    can_open_doors=npc.can_open_doors,
                )
                is not None
            ):
                continue
            if is_tile_hazardous(int(game_map.tiles[tx, ty])):
                continue
            candidates.append((dx, dy))

        if not candidates:
            return None
        return _rng.choice(candidates)
