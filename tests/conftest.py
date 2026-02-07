from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from catley.game import consequences
from catley.game.resolution import d20_system
from catley.util import dice, rng
from catley.util.live_vars import live_variable_registry
from catley.view.render.effects import particles

if TYPE_CHECKING:
    from collections.abc import Callable

    # Type aliases for fixture return types (shorter than inline annotations)
    CombatRNGPatcher = Callable[
        [list[int], list[int] | None], AbstractContextManager[Any]
    ]
    D20RNGPatcher = Callable[[list[int]], AbstractContextManager[Any]]


class FixedRandom:
    """Test helper that returns predetermined values for RNG calls.

    Usage:
        fr = FixedRandom([10, 15, 20])  # Returns 10, then 15, then 20
        with patch.object(d20_system._rng, "randint", fr):
            # Code that uses the D20 RNG will get these fixed values
    """

    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.index = 0

    def __call__(self, _a: int, _b: int) -> int:
        val = self.values[self.index]
        self.index = (self.index + 1) % len(self.values)
        return val


class FixedFloat:
    """Test helper that returns predetermined float values for RNG calls.

    Usage:
        ff = FixedFloat([0.5, 0.8])  # Returns 0.5, then 0.8
        with patch.object(particles._rng, "uniform", ff):
            # Code that uses the particles RNG will get these fixed values
    """

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self, _a: float = 0.0, _b: float = 1.0) -> float:
        val = self.values[self.index]
        self.index = (self.index + 1) % len(self.values)
        return val


@pytest.fixture
def patch_d20_rng() -> Callable[[list[int]], AbstractContextManager[Any]]:
    """Fixture that returns a function to patch the D20 RNG with fixed values.

    Usage:
        def test_something(patch_d20_rng):
            with patch_d20_rng([10, 15]):  # D20 rolls return 10, then 15
                # test code
    """

    def _patch(values: list[int]) -> AbstractContextManager[Any]:
        return patch.object(d20_system._rng, "randint", FixedRandom(values))

    return _patch


@pytest.fixture
def patch_dice_rng() -> Callable[[list[int]], AbstractContextManager[Any]]:
    """Fixture that returns a function to patch the dice RNG with fixed values.

    Usage:
        def test_something(patch_dice_rng):
            with patch_dice_rng([3, 4]):  # Dice rolls return 3, then 4
                # test code
    """

    def _patch(values: list[int]) -> AbstractContextManager[Any]:
        return patch.object(dice._rng, "randint", FixedRandom(values))

    return _patch


@pytest.fixture
def patch_combat_rng() -> Callable[
    [list[int], list[int] | None], AbstractContextManager[Any]
]:
    """Fixture to patch both D20 and dice RNG for combat tests.

    Usage:
        def test_something(patch_combat_rng):
            # D20 rolls return 15, dice rolls return 3
            with patch_combat_rng([15], [3]):
                # test code
    """

    @contextmanager
    def _patch(
        d20_values: list[int], dice_values: list[int] | None = None
    ) -> Iterator[None]:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(d20_system._rng, "randint", FixedRandom(d20_values))
            )
            if dice_values is not None:
                stack.enter_context(
                    patch.object(dice._rng, "randint", FixedRandom(dice_values))
                )
            yield

    return _patch


@pytest.fixture
def patch_consequences_rng() -> Callable[[list[float]], AbstractContextManager[Any]]:
    """Fixture that returns a function to patch the consequences RNG.

    Usage:
        def test_something(patch_consequences_rng):
            with patch_consequences_rng([0.2]):  # random() returns 0.2
                # test code
    """

    def _patch(values: list[float]) -> AbstractContextManager[Any]:
        return patch.object(consequences._rng, "random", FixedFloat(values))

    return _patch


@pytest.fixture
def patch_particles_rng() -> Callable[[list[float]], AbstractContextManager[Any]]:
    """Fixture that returns a function to patch the particles RNG.

    Usage:
        def test_something(patch_particles_rng):
            with patch_particles_rng([0.1, 0.2]):  # uniform() returns 0.1, then 0.2
                # test code
    """

    def _patch(values: list[float]) -> AbstractContextManager[Any]:
        return patch.object(particles._rng, "uniform", FixedFloat(values))

    return _patch


@pytest.fixture(autouse=True)
def reset_rng_for_tests() -> Iterator[None]:
    """Reset the RNG provider before each test for determinism.

    This ensures tests start with a fresh RNG state.
    """
    rng.init(0)
    yield


@pytest.fixture(autouse=True)
def clear_live_variable_registry() -> Iterator[None]:
    """Clear the global live variable registry before and after each test.

    Sets ``strict = False`` so that timing decorators (``record_time_live_variable``)
    in production code don't raise ``KeyError`` when the registry is empty.
    Strict mode is restored after the test.
    """
    original_strict = live_variable_registry.strict
    live_variable_registry.strict = False
    live_variable_registry._variables.clear()
    yield
    live_variable_registry._variables.clear()
    live_variable_registry.strict = original_strict


@pytest.fixture(scope="module")
def _shared_dummy_controller():
    """Module-scoped controller backed by DummyGameWorld. Not for direct use.

    Use the function-scoped ``controller`` fixture instead, which resets
    state between tests for isolation.
    """

    from tests.helpers import get_controller_with_dummy_world

    return get_controller_with_dummy_world()


@pytest.fixture
def controller(_shared_dummy_controller):
    """Lightweight controller fixture reset between tests.

    Backed by DummyGameWorld (no map generation). Use this for tests that
    need a Controller with a player but don't need real pathfinding or FOV.
    """
    from tests.helpers import reset_dummy_controller

    reset_dummy_controller(_shared_dummy_controller)
    return _shared_dummy_controller
