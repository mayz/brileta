from typing import cast
from unittest.mock import patch

from catley.game.actors import Actor, Character
from catley.game.resolution.d20_system import D20ResolutionResult, D20Resolver


class FixedRandom:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.index = 0

    def __call__(self, _a: int, _b: int) -> int:
        val = self.values[self.index]
        self.index += 1
        return val


def test_d20_resolver_basic_success() -> None:
    fr = FixedRandom([10])
    resolver = D20Resolver(5, 12)
    with patch("random.randint", fr):
        result = resolver.resolve(cast(Character, None), cast(Actor, None))
    assert isinstance(result, D20ResolutionResult)
    assert result.success
    assert not result.is_critical_success
    assert not result.is_critical_failure
    assert result.final_roll_used == 10
    assert result.total_value == 15


def test_d20_resolver_critical_outcomes() -> None:
    fr = FixedRandom([20, 1])
    hit_resolver = D20Resolver(0, 100)
    miss_resolver = D20Resolver(10, 5)
    with patch("random.randint", fr):
        hit = hit_resolver.resolve(cast(Character, None), cast(Actor, None))
        miss = miss_resolver.resolve(cast(Character, None), cast(Actor, None))
    assert hit.is_critical_success and hit.success
    assert miss.is_critical_failure and not miss.success


def test_d20_resolver_advantage_disadvantage() -> None:
    fr = FixedRandom([3, 17, 17, 5])
    adv_resolver = D20Resolver(0, 10, has_advantage=True)
    dis_resolver = D20Resolver(0, 10, has_disadvantage=True)
    with patch("random.randint", fr):
        adv = adv_resolver.resolve(cast(Character, None), cast(Actor, None))
        dis = dis_resolver.resolve(cast(Character, None), cast(Actor, None))
    assert adv.final_roll_used == 17
    assert adv.success
    assert dis.final_roll_used == 5
    assert not dis.success
