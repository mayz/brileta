from unittest.mock import patch

from brileta import colors
from brileta.game.actors import Actor, Character
from brileta.game.enums import OutcomeTier
from brileta.game.resolution import d20_system
from brileta.game.resolution.d20_system import D20ResolutionResult, D20System


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
    resolver = D20System(5, 12)
    actor = Character(0, 0, "@", colors.WHITE, "A")
    target = Actor(1, 0, "T", colors.WHITE)
    with patch.object(d20_system._rng, "randint", fr):
        result = resolver.resolve(actor, target)
    assert isinstance(result, D20ResolutionResult)
    assert result.outcome_tier == OutcomeTier.SUCCESS
    assert result.final_roll_used == 10
    assert result.total_value == 15


def test_d20_resolver_critical_outcomes() -> None:
    fr = FixedRandom([20, 1])
    hit_resolver = D20System(0, 100)
    miss_resolver = D20System(10, 5)
    actor = Character(0, 0, "@", colors.WHITE, "A")
    target = Actor(1, 0, "T", colors.WHITE)
    with patch.object(d20_system._rng, "randint", fr):
        hit = hit_resolver.resolve(actor, target)
        miss = miss_resolver.resolve(actor, target)
    assert hit.outcome_tier == OutcomeTier.CRITICAL_SUCCESS
    assert miss.outcome_tier == OutcomeTier.CRITICAL_FAILURE


def test_d20_resolver_advantage_disadvantage() -> None:
    fr = FixedRandom([3, 17, 17, 5])
    adv_resolver = D20System(0, 10, has_advantage=True)
    dis_resolver = D20System(0, 10, has_disadvantage=True)
    actor = Character(0, 0, "@", colors.WHITE, "A")
    target = Actor(1, 0, "T", colors.WHITE)
    with patch.object(d20_system._rng, "randint", fr):
        adv = adv_resolver.resolve(actor, target)
        dis = dis_resolver.resolve(actor, target)
    assert adv.final_roll_used == 17
    assert adv.outcome_tier == OutcomeTier.SUCCESS
    assert dis.final_roll_used == 5
    assert dis.outcome_tier == OutcomeTier.FAILURE
