"""Tests for the fixed-timestep game loop in App.update_game_logic.

The accumulator pattern runs logic at a fixed 60Hz rate regardless of visual
framerate. Death spiral protection caps logic steps per frame and decays
leftover time to prevent permanent slowdown after a lag spike.
"""

from __future__ import annotations

from unittest.mock import patch

from brileta.controller import Controller
from brileta.types import DeltaTime
from tests.helpers import DummyApp, get_controller_with_dummy_world


def _make_app_and_controller() -> tuple[DummyApp, Controller]:
    """Create a DummyApp wired to a real Controller."""
    controller = get_controller_with_dummy_world()
    app = DummyApp()
    app.controller = controller
    controller.app = app
    return app, controller


def _count_logic_steps(app: DummyApp, controller: Controller, delta: float) -> int:
    """Call update_game_logic while counting logic steps.

    Patches out process_player_input (needs full input stack) and metric
    recording (needs registered metrics) so we test the timing logic in isolation.
    """
    steps = 0

    def counting_step() -> None:
        nonlocal steps
        steps += 1

    with (
        patch.object(controller, "process_player_input"),
        patch("brileta.app.live_variable_registry.record_metric"),
        patch.object(app, "execute_fixed_logic_step", counting_step),
    ):
        app.update_game_logic(DeltaTime(delta))

    return steps


class TestFixedTimestepAccumulator:
    """The accumulator converts variable frame time into fixed logic steps."""

    def test_single_step_normal_frame(self) -> None:
        """A normal frame (1/60s) produces exactly one logic step."""
        app, controller = _make_app_and_controller()
        controller.accumulator = 0.0

        steps = _count_logic_steps(app, controller, controller.fixed_timestep)

        assert steps == 1
        assert controller.accumulator < controller.fixed_timestep

    def test_zero_steps_on_tiny_delta(self) -> None:
        """A very short frame produces zero logic steps."""
        app, controller = _make_app_and_controller()
        controller.accumulator = 0.0

        steps = _count_logic_steps(app, controller, controller.fixed_timestep * 0.5)

        assert steps == 0
        assert controller.accumulator > 0


class TestDeathSpiralProtection:
    """Caps logic steps per frame and decays the accumulator after a spike."""

    def test_caps_logic_steps_per_frame(self) -> None:
        """When a huge delta arrives, logic steps are capped to prevent spiral."""
        app, controller = _make_app_and_controller()
        controller.accumulator = 0.0

        # 1 second would be ~60 steps without the cap
        steps = _count_logic_steps(app, controller, 1.0)

        assert steps == controller.max_logic_steps_per_frame

    def test_accumulator_decays_after_spiral(self) -> None:
        """When the cap kicks in, the leftover accumulator is decayed by 0.8."""
        app, controller = _make_app_and_controller()
        controller.accumulator = 0.0

        _count_logic_steps(app, controller, 1.0)

        # After max steps, remaining accumulator decays:
        # leftover = (1.0 - consumed) * 0.8
        consumed = controller.max_logic_steps_per_frame * controller.fixed_timestep
        expected_leftover = (1.0 - consumed) * 0.8

        assert abs(controller.accumulator - expected_leftover) < 1e-9

    def test_no_decay_when_under_cap(self) -> None:
        """Normal frames don't trigger the 0.8 decay."""
        app, controller = _make_app_and_controller()
        controller.accumulator = 0.0

        # Two-and-a-half steps worth of time (under the cap of 5)
        delta = controller.fixed_timestep * 2.5

        _count_logic_steps(app, controller, delta)

        # Leftover should be the fractional remainder, NOT decayed
        expected_leftover = delta - (2 * controller.fixed_timestep)
        assert abs(controller.accumulator - expected_leftover) < 1e-9
