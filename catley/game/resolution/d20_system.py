from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catley.game.enums import OutcomeTier

from .base import ResolutionResult, Resolver

if TYPE_CHECKING:
    from catley.game.actors import Actor, Character
    from catley.game.items.item_core import Item


@dataclass
class D20ResolutionResult(ResolutionResult):
    """Result data for a d20 roll resolution."""

    rolls_made: list[int] = field(default_factory=list)
    final_roll_used: int = 0
    total_value: int = 0
    target_value: int = 0
    has_advantage: bool = False
    has_disadvantage: bool = False


class D20Resolver(Resolver):
    """Concrete implementation of a d20-based resolution system."""

    def __init__(
        self,
        ability_score: int,
        roll_to_exceed: int,
        *,
        has_advantage: bool = False,
        has_disadvantage: bool = False,
    ) -> None:
        self.ability_score = ability_score
        self.roll_to_exceed = roll_to_exceed
        self.has_advantage = has_advantage
        self.has_disadvantage = has_disadvantage

    def resolve(
        self,
        actor: Character,
        target: Actor,
        weapon: Item | None = None,
    ) -> D20ResolutionResult:
        """Perform the d20 roll and return the result."""
        # Advantage and disadvantage cancel out
        has_advantage = self.has_advantage
        has_disadvantage = self.has_disadvantage
        if has_advantage and has_disadvantage:
            has_advantage = False
            has_disadvantage = False

        num_dice_to_roll = 1 if not (has_advantage or has_disadvantage) else 2
        rolls_made: list[int] = [random.randint(1, 20) for _ in range(num_dice_to_roll)]

        if has_advantage:
            final_roll_used = max(rolls_made)
        elif has_disadvantage:
            final_roll_used = min(rolls_made)
        else:
            final_roll_used = rolls_made[0]

        is_critical_success = final_roll_used == 20
        is_critical_failure = final_roll_used == 1

        total_value = final_roll_used + self.ability_score

        if is_critical_success:
            outcome_tier = OutcomeTier.CRITICAL_SUCCESS
        elif is_critical_failure:
            outcome_tier = OutcomeTier.CRITICAL_FAILURE
        else:
            if total_value > self.roll_to_exceed:
                outcome_tier = OutcomeTier.SUCCESS
            elif total_value == self.roll_to_exceed:
                outcome_tier = OutcomeTier.PARTIAL_SUCCESS
            else:
                outcome_tier = OutcomeTier.FAILURE

        return D20ResolutionResult(
            outcome_tier=outcome_tier,
            rolls_made=rolls_made,
            final_roll_used=final_roll_used,
            total_value=total_value,
            target_value=self.roll_to_exceed,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )

    def calculate_success_probability(self) -> float:
        """
        Calculate the probability of success for this resolver's configuration.

        Returns:
            Probability of success as a float in the range [0.0, 1.0].
        """
        has_advantage = self.has_advantage
        has_disadvantage = self.has_disadvantage
        if has_advantage and has_disadvantage:
            has_advantage = False
            has_disadvantage = False

        prob_single_success = self._calculate_single_d20_success_probability(
            self.ability_score, self.roll_to_exceed
        )

        if not has_advantage and not has_disadvantage:
            return prob_single_success

        if has_advantage:
            prob_single_failure = 1.0 - prob_single_success
            return 1.0 - (prob_single_failure**2)

        return prob_single_success**2

    def _calculate_single_d20_success_probability(
        self, ability_score: int, roll_to_exceed: int
    ) -> float:
        """Calculate probability of success for a single d20 roll."""
        natural_roll_to_exceed = roll_to_exceed - ability_score

        # 1. Count natural 20 (always a success)
        num_successful_outcomes = 1

        # 2. Count non-critical, non-miss successes (rolls 2-19)
        min_successful_non_crit_roll = max(2, natural_roll_to_exceed + 1)
        if min_successful_non_crit_roll <= 19:
            num_successful_outcomes += 19 - min_successful_non_crit_roll + 1

        return num_successful_outcomes / 20.0
