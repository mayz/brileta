"""
Simulates complex dice rolls and d20 check/test roll mechanics.

This module provides two main areas of functionality:
1.  General Dice Rolling:
    The `Dice` class allows for creating dice objects from string notations
    (e.g., "2d10+5", "-d6", "15") and rolling them to get a numerical result.
    It handles multiple dice, modifiers, and negative dice outcomes.
    For convenience, a `roll_d()` function is provided for rolling a single die.

2.  d20 Check Roll System:
    A system for simulating d20-based checks. This includes:
    - `perform_check_roll()`: The core function that handles d20 rolls,
      addition of an ability score, comparison against a target value
      (which must be strictly exceeded), critical hits (natural 20),
      critical misses (natural 1), advantage, and disadvantage.
    - Convenience wrappers: `perform_unopposed_check_roll()` and
      `perform_opposed_check_roll()` for common scenarios.
    - `calculate_check_roll_success_probability()`: To compute the
      probability of success for any given check roll setup.
"""

import random
from dataclasses import dataclass, field


class Dice:
    """A class representing dice that can be rolled.

    This class handles parsing dice strings like "d20", "-d8", "2d10+15", etc.,
    and provides methods to roll the dice.
    """

    def __init__(self, dice_str: str) -> None:
        """Initialize a Dice object from a string representation.

        Args:
            dice_str: String representation of the dice
                      (e.g., "d20", "-d4", "2d6", "2d10+15")

        Raises:
            ValueError: If the dice string format is invalid
        """
        self.dice_str = dice_str
        self.num_dice, self.sides, self.multiplier, self.modifier = (
            self._parse_dice_str(dice_str)
        )

    def _parse_dice_str(self, dice_str: str) -> tuple[int, int, int, int]:
        """Parse a dice string into its components.

        Args:
            dice_str: String representation of the dice

        Returns:
            Tuple of (number of dice, sides, multiplier, modifier)

        Raises:
            ValueError: If the dice string format is invalid
        """
        dice_str = dice_str.replace(" ", "")  # Remove all spaces

        # Initialize modifier
        modifier = 0
        dice_part = dice_str

        # Check for modifiers (e.g., "2d10+15" or "d20-3")
        if "+" in dice_str:
            dice_part, mod_part = dice_str.split("+", 1)
            modifier = int(mod_part)
        elif "-" in dice_str and not dice_str.startswith("-"):
            # Only split on - if it's not at the beginning (to handle negative dice)
            dice_part, mod_part = dice_str.split("-", 1)
            modifier = -int(mod_part)

        # Handle fixed values (e.g., "5" or "-3")
        if dice_part.lstrip("-").isdigit():
            return 0, int(dice_part), 0, modifier  # num_dice=0 means fixed value

        # Handle negative dice (e.g., "-d8")
        if dice_part.startswith("-d"):
            return 1, int(dice_part[2:]), -1, modifier

        # Handle standard dice with optional count (e.g., "d20" or "2d6")
        if "d" in dice_part:
            parts = dice_part.split("d")

            # Handle case like "d20" (implied 1d20)
            if parts[0] == "":
                return 1, int(parts[1]), 1, modifier

            # Handle case like "2d6"
            return int(parts[0]), int(parts[1]), 1, modifier

        raise ValueError(f"Invalid dice format: {dice_str}")

    def roll(self) -> int:
        """Roll the dice and return the result.

        Returns:
            int: The result of the dice roll
        """
        # Handle fixed values
        if self.num_dice == 0:
            return self.sides + self.modifier

        # Roll the dice and apply multiplier
        result = 0
        for _ in range(self.num_dice):
            result += random.randint(1, self.sides)

        # Apply multiplier and modifier
        return (self.multiplier * result) + self.modifier

    def __str__(self) -> str:
        """Return the string representation of the dice."""
        return self.dice_str


def roll_d(sides: int) -> int:
    """Rolls a single die with the specified number of sides.

    Args:
        sides: The number of sides on the die (e.g., 4, 6, 8, 10, 12, 20, 100).

    Returns:
        The result of the die roll (an integer between 1 and `sides`, inclusive).

    Raises:
        ValueError: If `sides` is not a positive integer.
    """
    if not isinstance(sides, int) or sides <= 0:
        raise ValueError("Number of sides must be a positive integer.")

    # Use direct random.randint for efficiency - creating a Dice object for
    # every roll would be unnecessarily expensive
    return random.randint(1, sides)


@dataclass
class CheckResult:
    success: bool
    is_critical_hit: bool = False  # Natural 20
    is_critical_miss: bool = False  # Natural 1
    rolls_made: list[int] = field(default_factory=list)  # Actual d20 values rolled
    final_roll_used: int = 0  # The d20 roll chosen after advantage/disadvantage
    total_value: int = 0  # Final roll + modifier
    target_value: int = 0
    has_advantage: bool = False
    has_disadvantage: bool = False


def perform_check_roll(
    ability_score: int,
    roll_to_exceed: int,
    has_advantage: bool = False,
    has_disadvantage: bool = False,
) -> CheckResult:
    """Performs a d20 check roll against a target value.

    The Wastoid rules call this a "test" roll, but "test" is such an overloaded
    term, that we're going with "check", which is what most other RPGs call it.

    A check roll involves rolling a d20, adding an ability score, and comparing
    the total against a `roll_to_exceed` value. The roll succeeds if the
    total is strictly greater than `roll_to_exceed`.

    Natural 20s on the d20 are always successes (critical hits).
    Natural 1s on the d20 are always failures (critical misses).
    Advantage (roll two d20s, take higher) and disadvantage (roll two d20s,
    take lower) can be applied. If both are present, they cancel out.

    See also the `perform_unopposed_check_roll()` and `perform_opposed_check_roll()`
    convenience functions.

    Args:
        ability_score: The ability score value to add to the d20 roll.
        roll_to_exceed: The number to beat. The value that (d20 roll + ability_score)
                        must strictly exceed for a success (unless it's a critical
                        hit or miss).
        has_advantage: If True, two d20s are rolled and the higher result is used.
        has_disadvantage: If True, two d20s are rolled and the lower result is used.

    Returns:
        A CheckResult object detailing the outcome of the roll.
    """
    # If both an advantage and a disadvantage, they cancel out.
    if has_advantage and has_disadvantage:
        has_advantage = False
        has_disadvantage = False

    # Determine number of dice and which to keep
    num_dice_to_roll = 1 if not (has_advantage or has_disadvantage) else 2

    rolls_made: list[int] = [roll_d(20) for _ in range(num_dice_to_roll)]

    if has_advantage:
        final_roll_used = max(rolls_made)
    elif has_disadvantage:
        final_roll_used = min(rolls_made)
    else:
        final_roll_used = rolls_made[0]

    is_critical_hit = final_roll_used == 20
    is_critical_miss = final_roll_used == 1

    total_value = final_roll_used + ability_score

    success: bool
    if is_critical_hit:
        success = True
    elif is_critical_miss:
        success = False
    else:
        success = total_value > roll_to_exceed

    return CheckResult(
        success=success,
        is_critical_hit=is_critical_hit,
        is_critical_miss=is_critical_miss,
        final_roll_used=final_roll_used,
        total_value=total_value,
        target_value=roll_to_exceed,
        has_advantage=has_advantage,
        has_disadvantage=has_disadvantage,
    )


def perform_unopposed_check_roll(
    ability_score: int,
    has_advantage: bool = False,
    has_disadvantage: bool = False,
) -> CheckResult:
    """Performs an unopposed check roll against a standard target value of 15.

    This is a convenience wrapper around `perform_check_roll()` where the
    `roll_to_exceed` value is fixed at 15.

    Args:
        ability_score: The modifier to add to the d20 roll.
        has_advantage: If True, two d20s are rolled and the higher result is used.
        has_disadvantage: If True, two d20s are rolled and the lower result is used.

    Returns:
        A CheckResult object detailing the outcome of the roll.
    """
    target_dc = 15
    return perform_check_roll(ability_score, target_dc, has_advantage, has_disadvantage)


def perform_opposed_check_roll(
    actor_ability_score: int,
    opponent_ability_score: int,
    has_advantage: bool = False,
    has_disadvantage: bool = False,
) -> CheckResult:
    """Performs an opposed check roll.

    The `roll_to_exceed` value is calculated as the opponent's ability score + 10.
    This is a convenience wrapper around `perform_check_roll()`.

    Args:
        actor_ability_score: The ability score of the character making the roll.
        opponent_ability_score: The ability score of the character opposing the roll,
                                used to calculate the target value.
        has_advantage: If True, the actor's roll has advantage.
        has_disadvantage: If True, the actor's roll has disadvantage.

    Returns:
        A CheckResult object detailing the outcome of the actor's roll.
    """
    target_dc = opponent_ability_score + 10
    return perform_check_roll(
        actor_ability_score, target_dc, has_advantage, has_disadvantage
    )


def calculate_check_roll_success_probability(
    ability_score: int,
    roll_to_exceed: int,
    has_advantage: bool = False,
    has_disadvantage: bool = False,
) -> float:
    """Calculate the probability of success for a check roll."""
    if has_advantage and has_disadvantage:
        has_advantage = False
        has_disadvantage = False

    prob_single_success = _calculate_single_d20_success_probability(
        ability_score, roll_to_exceed
    )

    if not has_advantage and not has_disadvantage:
        return prob_single_success

    if has_advantage:
        # Succeed if not both dice fail
        prob_single_failure = 1.0 - prob_single_success
        return 1.0 - (prob_single_failure**2)

    # has_disadvantage
    # Succeed if both dice would individually succeed (as the lower roll is taken)
    return prob_single_success**2


def _calculate_single_d20_success_probability(
    ability_score: int, roll_to_exceed: int
) -> float:
    """
    Calculates the probability of success for a single d20 roll (loop-less).

    Because natural 20 is always a success and natural 1 is always a failure,
    the output range is [0.05, 0.95].
    """
    natural_roll_to_exceed = roll_to_exceed - ability_score

    # 1. Count natural 20 (always a success)
    # A natural 20 is one specific outcome.
    num_successful_outcomes = 1  # The outcome '20'

    # 2. Count non-critical, non-miss successes (rolls 2-19)
    # These rolls succeed if d20_roll > roll_to_exceed.
    # Smallest possible successful roll in this range is
    # roll_to_exceed + 1, but not less than 2.
    min_successful_non_crit_roll = max(2, natural_roll_to_exceed + 1)

    # Largest possible successful roll in this range is 19.
    # If min_successful_non_crit_roll > 19, there are no such successes.
    if min_successful_non_crit_roll <= 19:
        # Number of successful outcomes in the range [min_successful_non_crit_roll, 19]
        num_successful_outcomes += 19 - min_successful_non_crit_roll + 1

    return num_successful_outcomes / 20.0


if __name__ == "__main__":
    print("Example dice rolls:")
    print(f"d4: {roll_d(4)}")
    print(f"d6: {roll_d(6)}")
    print(f"d8: {roll_d(8)}")
    print(f"d12: {roll_d(12)}")
    print(f"d20: {roll_d(20)}")

    # Examples using the Dice class directly
    print("\nDice class examples:")
    print(f"d20: {Dice('d20').roll()}")
    print(f"-d8: {Dice('-d8').roll()}")
    print(f"3d6: {Dice('3d6').roll()}")
    print(f"2d10+15: {Dice('2d10+15').roll()}")
    print(f"Fixed value 5: {Dice('5').roll()}")
