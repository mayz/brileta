"""Utility-based scoring system for NPC decision-making."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from brileta.types import Direction, WorldTilePos

# Callable signature shared by all precondition helpers and Goal.preconditions.
type Precondition = Callable[[UtilityContext], bool]


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


class ResponseCurveType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    INVERSE = "inverse"
    STEP = "step"
    BELL = "bell"


@dataclass(slots=True)
class ResponseCurve:
    curve_type: ResponseCurveType
    exponent: float = 2.0
    threshold: float = 0.5
    peak: float = 0.5
    width: float = 0.5

    def evaluate(self, value: float) -> float:
        value = _clamp(value)
        match self.curve_type:
            case ResponseCurveType.LINEAR:
                return value
            case ResponseCurveType.EXPONENTIAL:
                return value**self.exponent
            case ResponseCurveType.INVERSE:
                return 1.0 - value
            case ResponseCurveType.STEP:
                return 1.0 if value >= self.threshold else 0.0
            case ResponseCurveType.BELL:
                if self.width <= 0:
                    return 0.0
                return _clamp(1.0 - abs(value - self.peak) / self.width)
        return 0.0


@dataclass(slots=True)
class Consideration:
    input_key: str
    curve: ResponseCurve
    weight: float = 1.0

    def evaluate(self, context: UtilityContext) -> float:
        value = context.get_input(self.input_key)
        if value is None:
            return 0.0
        return self.curve.evaluate(value) ** self.weight


@dataclass(slots=True)
class UtilityContext:
    controller: Controller
    actor: NPC
    target: Character
    distance_to_target: int
    health_percent: float
    threat_level: float
    can_attack: bool
    has_escape_route: bool
    best_attack_destination: WorldTilePos | None
    best_flee_step: Direction | None
    # Normalized disposition: 0.0 maps to -100 (hostile), 1.0 maps to +100 (ally).
    disposition: float = 0.5

    def get_input(self, key: str) -> float | None:
        match key:
            case "health_percent":
                return self.health_percent
            case "threat_level":
                return self.threat_level
            case "has_escape_route":
                return 1.0 if self.has_escape_route else 0.0
            case "can_attack":
                return 1.0 if self.can_attack else 0.0
            case "disposition":
                return self.disposition
        return None


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------

# Hostile precondition threshold in normalized disposition space.
# Matches the current ai.HOSTILE_UPPER of -51 mapped via (value + 100) / 200.
_HOSTILE_DISPOSITION_THRESHOLD = (-51 + 100) / 200.0


def is_threat_present(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is non-zero."""
    return context.threat_level > 0.0


def is_no_threat(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is zero."""
    return context.threat_level <= 0.0


def has_escape_route(context: UtilityContext) -> bool:
    """Precondition: returns True when a valid flee step exists."""
    return context.has_escape_route


def is_hostile(context: UtilityContext) -> bool:
    """Precondition: only initiate combat when hostile toward target.

    This is a game rule, not a scoring gate. Non-hostile NPCs do not start
    fights regardless of other considerations. Disposition scoring curves
    handle the intensity of hostile behavior (e.g., attack vs flee).
    """
    return context.disposition <= _HOSTILE_DISPOSITION_THRESHOLD


def is_not_hostile(context: UtilityContext) -> bool:
    """Precondition: only allow non-hostile social behaviors."""
    return context.disposition > _HOSTILE_DISPOSITION_THRESHOLD


class UtilityAction(abc.ABC):
    """Base class for utility actions.

    A UtilityAction is a single-tick, stateless operation scored by the
    UtilityBrain every tick. Most NPC behaviors are actions: Attack, Avoid,
    Watch, Wander, Idle. If the same action keeps winning scoring each tick,
    the behavior sustains naturally without any memory between ticks.

    When a behavior requires memory across ticks (patrol waypoint sequence,
    flee distance tracking, request-help state machine), the action creates
    a Goal on its first tick. After that, ContinueGoalAction competes in
    scoring to keep the goal alive. See goals.py for the Goal system.
    """

    def __init__(
        self,
        action_id: str,
        base_score: float,
        considerations: list[Consideration] | None = None,
        preconditions: list[Precondition] | None = None,
    ) -> None:
        self.action_id = action_id
        self.base_score = base_score
        self.considerations = considerations or []
        self.preconditions: list[Precondition] = preconditions or []

    def score(self, context: UtilityContext) -> float:
        if any(not pre(context) for pre in self.preconditions):
            return 0.0
        score = self.base_score
        for consideration in self.considerations:
            score *= consideration.evaluate(context)
        return score

    @abc.abstractmethod
    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Create an intent for this action if applicable."""
        ...


@dataclass(slots=True)
class ScoredAction:
    """Debug snapshot of one action's scoring result."""

    display_name: str
    final_score: float
    base_score: float = 0.0
    persistence_bonus: float = 0.0


class UtilityBrain:
    """Evaluate utility actions and select the best one.

    When the NPC has an active goal, the brain also scores a
    ContinueGoalAction alongside all other actions. Goals compete in the
    same scoring system as atomic actions every tick - no special
    interruption logic is needed.
    """

    def __init__(self, actions: list[UtilityAction]) -> None:
        self.actions = actions

    def select_action(
        self, context: UtilityContext, current_goal: Goal | None = None
    ) -> tuple[UtilityAction | None, list[ScoredAction]]:
        """Score all actions (including goal continuation) and return the best.

        Args:
            context: The current utility evaluation context.
            current_goal: The NPC's active goal, if any. If present and not
                complete, a ContinueGoalAction is scored alongside other actions.

        Returns:
            A tuple of (best_action, scored_actions) where scored_actions is
            the full scoring breakdown for debug display.
        """
        best_action: UtilityAction | None = None
        best_score = -1.0
        scored: list[ScoredAction] = []

        for action in self.actions:
            score = action.score(context)
            scored.append(
                ScoredAction(
                    display_name=action.action_id.title(),
                    final_score=score,
                    base_score=score,
                )
            )
            if score > best_score:
                best_score = score
                best_action = action

        # Score "continue current goal" alongside atomic actions
        if current_goal is not None and not current_goal.is_complete:
            from .goals import (
                PERSISTENCE_MINIMUM,
                PERSISTENCE_WEIGHT,
                ContinueGoalAction,
            )

            continue_action = ContinueGoalAction(current_goal)
            final = continue_action.score(context)
            persist = PERSISTENCE_MINIMUM + current_goal.progress * PERSISTENCE_WEIGHT
            scored.append(
                ScoredAction(
                    display_name=f"ContinueGoal({current_goal.goal_id.title()})",
                    final_score=final,
                    base_score=final - persist,
                    persistence_bonus=persist,
                )
            )
            if final > best_score:
                best_score = final
                best_action = continue_action

        return best_action, scored


if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors import NPC, Character

    from .goals import Goal
