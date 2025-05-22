"""
Module for simulating dice rolls used in the Deadball game.
"""

import random


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
            Tuple of (number of dice, sides, multiplier)

        Raises:
            ValueError: If the dice string format is invalid
        """
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
