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
