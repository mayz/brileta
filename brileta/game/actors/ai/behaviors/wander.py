"""Wander behavior: WanderAction + WanderGoal.

WanderAction is the utility-system entry point. When it wins scoring,
AIComponent creates a WanderGoal for continuous strolling with heading
drift, pauses, and direction changes.
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
    is_not_hostile,
)
from brileta.types import Direction, WorldTilePos
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC

_rng = rng.get("npc.goals")

# Wander tuning: sustained stroll with occasional pauses and direction changes.
_WANDER_MAX_STUCK_TURNS = 2
_WANDER_HEADING_JITTER_CHANCE = 0.15
_WANDER_PAUSE_CHANCE = 0.20
_WANDER_MIN_LINGER_TURNS = 2
_WANDER_MAX_LINGER_TURNS = 6
_WANDER_NEW_HEADING_CHANCE = 0.05
_WANDER_SPEED_MIN = 0.60
_WANDER_SPEED_MAX = 0.90


class WanderAction(UtilityAction):
    """Create/continue a strolling wander goal when no threat is present.

    Preconditions: no threat and NPC not hostile.
    Scores higher with lower threat and higher (friendlier) disposition.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [is_no_threat, is_not_hostile]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("threat_level", ResponseCurve(ResponseCurveType.INVERSE)),
        Consideration("disposition", ResponseCurve(ResponseCurveType.LINEAR)),
    ]

    def __init__(self, base_score: float = 0.18) -> None:
        super().__init__(
            action_id="wander",
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
        """Create a stroll-style WanderGoal and return the first intent."""
        goal = WanderGoal(
            preconditions=self.preconditions,
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


class WanderGoal(Goal):
    """Wander with a continuous strolling model.

    Rather than finite step-budget segments, this goal keeps a heading over
    time and occasionally pauses, drifts, or repicks direction. Movement still
    uses the same directional fallback logic (forward -> sidestep -> turns).
    """

    _DIRECTIONS: tuple[Direction, ...] = (
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
        preconditions: list | None = None,
        pause_chance: float = 0.20,
        minimum_linger_turns: int = 1,
        maximum_linger_turns: int = 3,
        new_heading_chance: float = 0.05,
        max_stuck_turns: int = 2,
        heading_jitter_chance: float = 0.15,
        speed_min_multiplier: float = 0.60,
        speed_max_multiplier: float = 0.90,
    ) -> None:
        super().__init__(goal_id="wander", preconditions=preconditions)
        if not 0.0 <= pause_chance <= 1.0:
            msg = "pause_chance must be in [0.0, 1.0]"
            raise ValueError(msg)
        if minimum_linger_turns <= 0:
            msg = "minimum_linger_turns must be positive"
            raise ValueError(msg)
        if maximum_linger_turns < minimum_linger_turns:
            msg = "maximum_linger_turns must be >= minimum_linger_turns"
            raise ValueError(msg)
        if not 0.0 <= new_heading_chance <= 1.0:
            msg = "new_heading_chance must be in [0.0, 1.0]"
            raise ValueError(msg)
        if max_stuck_turns <= 0:
            msg = "max_stuck_turns must be positive"
            raise ValueError(msg)
        if not 0.0 <= heading_jitter_chance <= 1.0:
            msg = "heading_jitter_chance must be in [0.0, 1.0]"
            raise ValueError(msg)
        if speed_min_multiplier <= 0.0:
            msg = "speed_min_multiplier must be positive"
            raise ValueError(msg)
        if speed_max_multiplier < speed_min_multiplier:
            msg = "speed_max_multiplier must be >= speed_min_multiplier"
            raise ValueError(msg)

        self._pause_chance = pause_chance
        self._minimum_linger_turns = minimum_linger_turns
        self._maximum_linger_turns = maximum_linger_turns
        self._new_heading_chance = new_heading_chance
        self._max_stuck_turns = max_stuck_turns
        self._heading_jitter_chance = heading_jitter_chance

        self._heading: Direction | None = None
        self._linger_remaining: int = 0
        self._stuck_turns: int = 0
        self._last_attempt_origin: WorldTilePos | None = None
        self._speed_multiplier = _rng.uniform(
            speed_min_multiplier, speed_max_multiplier
        )
        self._original_speed: int | None = None
        self._actor_ref: NPC | None = None

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
        """Wander is open-ended; return a moderate persistence signal."""
        return 0.35

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Wander is open-ended and has no progress-based completion.

        Eligibility checks (threat, hostility) are handled by preconditions
        on ContinueGoalAction, not here. evaluate_completion is reserved for
        goal-specific progress or terminal-state logic.
        """
        return

    def _on_state_changed(
        self, previous_state: GoalState, new_state: GoalState
    ) -> None:
        """Restore actor speed whenever wander enters a terminal state."""
        _ = previous_state
        if new_state in (
            GoalState.COMPLETED,
            GoalState.FAILED,
            GoalState.ABANDONED,
        ):
            self._restore_speed()

    def apply_wander_speed(self, actor: NPC) -> None:
        """Apply wander speed once and cache original speed for restoration."""
        if self._original_speed is not None:
            return
        self._actor_ref = actor
        self._original_speed = actor.energy.speed
        actor.energy.speed = max(
            1,
            round(self._original_speed * self._speed_multiplier),
        )

    def _restore_speed(self) -> None:
        """Restore original actor speed if it was modified by wander."""
        if self._actor_ref is None or self._original_speed is None:
            return
        self._actor_ref.energy.speed = self._original_speed
        self._actor_ref = None
        self._original_speed = None

    @classmethod
    def _rotate_heading(cls, heading: Direction, offset: int) -> Direction:
        """Return heading rotated by ``offset`` steps in the direction wheel."""
        idx = cls._DIRECTIONS.index(heading)
        return cls._DIRECTIONS[(idx + offset) % len(cls._DIRECTIONS)]

    def _is_step_walkable(
        self, npc: NPC, controller: Controller, dx: int, dy: int
    ) -> bool:
        """Return True when a step destination is in bounds, walkable, and unblocked.

        Door-capable NPCs treat closed doors as passable here. The actual
        door-opening is handled by ActionRouter's bump-to-open fallback
        when the resulting MoveIntent collides with the door.
        """
        from brileta.util.pathfinding import probe_step

        tx = npc.x + dx
        ty = npc.y + dy
        return (
            probe_step(
                controller.gw.game_map,
                controller.gw,
                tx,
                ty,
                exclude_actor=npc,
                can_open_doors=npc.can_open_doors,
            )
            is None
        )

    def _pick_segment_heading(
        self, npc: NPC, controller: Controller
    ) -> Direction | None:
        """Pick a random walkable heading, preferring safe tiles."""
        from brileta.environment.tile_types import get_hazard_cost

        safe: list[Direction] = []
        hazardous: list[Direction] = []
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

    def _apply_heading_jitter(self) -> None:
        """Occasionally nudge heading by 45 degrees to avoid straight-line drift."""
        if self._heading is None:
            return
        if _rng.random() >= self._heading_jitter_chance:
            return
        offset = _rng.choice((-1, 1))
        self._heading = self._rotate_heading(self._heading, offset)

    def _choose_step(self, npc: NPC, controller: Controller) -> Direction | None:
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
            candidates: list[tuple[Direction, int]] = []
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
            safest: list[Direction] = [
                direction for direction, cost in candidates if cost == min_cost
            ]
            return _rng.choice(safest)

        return None

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Return the next wander step for a continuous stroll."""
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

        if self._linger_remaining > 0:
            self._linger_remaining = max(0, self._linger_remaining - 1)
            return None

        if _rng.random() < self._pause_chance:
            linger_turns = _rng.randint(
                self._minimum_linger_turns, self._maximum_linger_turns
            )
            # Current tick already pauses; track only remaining linger ticks.
            self._linger_remaining = max(0, linger_turns - 1)
            return None

        needs_new_heading = (
            self._heading is None
            or self._stuck_turns >= self._max_stuck_turns
            or _rng.random() < self._new_heading_chance
        )
        if needs_new_heading:
            heading = self._pick_segment_heading(npc, controller)
            if heading is None:
                self._heading = None
                return None
            self._heading = heading
            self._stuck_turns = 0

        self._apply_heading_jitter()
        step = self._choose_step(npc, controller)
        if step is None:
            self._stuck_turns += 1
            return None

        dx, dy = step
        self._heading = (dx, dy)
        self._last_attempt_origin = current_pos
        return MoveIntent(controller, npc, dx, dy)
