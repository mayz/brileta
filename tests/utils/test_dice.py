import pytest

from catley.util import dice


def test_dice_parsing():
    d = dice.Dice("2d10+5")
    assert (d.num_dice, d.sides, d.multiplier, d.modifier) == (2, 10, 1, 5)
    d = dice.Dice("d6")
    assert (d.num_dice, d.sides, d.multiplier, d.modifier) == (1, 6, 1, 0)
    d = dice.Dice("-d8")
    assert (d.num_dice, d.sides, d.multiplier, d.modifier) == (1, 8, -1, 0)
    d = dice.Dice("5")
    assert (d.num_dice, d.sides, d.multiplier, d.modifier) == (0, 5, 0, 0)
    with pytest.raises(ValueError):
        dice.Dice("notadice")


def test_dice_average_simple():
    """Average of 1d6 is 3.5."""
    d = dice.Dice("1d6")
    assert d.average() == 3.5


def test_dice_average_multiple_dice():
    """Average of 2d6 is 7.0."""
    d = dice.Dice("2d6")
    assert d.average() == 7.0


def test_dice_average_with_modifier():
    """Average of 1d4+2 is 4.5."""
    d = dice.Dice("1d4+2")
    assert d.average() == 4.5


def test_dice_average_negative_dice():
    """Average of -d8 is -4.5."""
    d = dice.Dice("-d8")
    assert d.average() == -4.5


def test_dice_average_fixed_value():
    """Average of '5' is 5.0."""
    d = dice.Dice("5")
    assert d.average() == 5.0
