from unittest.mock import patch

import pytest

from catley.util import dice


class FixedRandom:
    def __init__(self, values):
        self.values = values
        self.index = 0

    def __call__(self, a, b):
        val = self.values[self.index]
        self.index += 1
        return val


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


def test_perform_check_roll_basic_success():
    fr = FixedRandom([10])
    with patch("random.randint", fr):
        result = dice.perform_check_roll(5, 12)
    assert result.success
    assert not result.is_critical_hit
    assert not result.is_critical_miss
    assert result.final_roll_used == 10
    assert result.total_value == 15


def test_perform_check_roll_critical_hit_and_miss():
    fr = FixedRandom([20, 1])
    with patch("random.randint", fr):
        hit = dice.perform_check_roll(0, 100)
        miss = dice.perform_check_roll(10, 5)
    assert hit.is_critical_hit and hit.success
    assert miss.is_critical_miss and not miss.success


def test_perform_check_roll_advantage_disadvantage():
    fr = FixedRandom([3, 17, 17, 5])
    with patch("random.randint", fr):
        adv = dice.perform_check_roll(0, 10, has_advantage=True)
        dis = dice.perform_check_roll(0, 10, has_disadvantage=True)
    assert adv.final_roll_used == 17
    assert adv.success
    assert dis.final_roll_used == 5
    assert not dis.success
