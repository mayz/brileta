"""Utility-based scoring system for NPC decision-making."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING


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
    player: Character
    distance_to_player: int
    health_percent: float
    threat_level: float
    can_attack: bool
    has_escape_route: bool
    best_attack_destination: tuple[int, int] | None
    best_flee_step: tuple[int, int] | None

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
        return None


class Action(abc.ABC):
    """Base class for utility actions."""

    def __init__(
        self,
        action_id: str,
        base_score: float,
        considerations: list[Consideration] | None = None,
        preconditions: list[Callable[[UtilityContext], bool]] | None = None,
    ) -> None:
        self.action_id = action_id
        self.base_score = base_score
        self.considerations = considerations or []
        self.preconditions = preconditions or []

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


class UtilityBrain:
    """Evaluate utility actions and select the best one."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions

    def select_action(self, context: UtilityContext) -> Action | None:
        best_action: Action | None = None
        best_score = -1.0
        for action in self.actions:
            score = action.score(context)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.actors import NPC, Character
