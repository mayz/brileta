"""Tests for process_player_input autopilot gating and manual action handling.

process_player_input runs once per visual frame and has two paths:
1. Manual actions - queued by keyboard/UI, execute immediately, cancel plans
2. Autopilot actions - from active plans or pathfinding, gated by multiple conditions
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

from brileta import config, input_events
from brileta.controller import Controller
from tests.helpers import DummyOverlay, get_controller_with_dummy_world


def _make_controller() -> Controller:
    """Return a controller ready for input processing tests."""
    controller = get_controller_with_dummy_world()
    assert controller.overlay_system is not None
    # InputHandler is patched to None in the test fixture, so set a mock
    controller.input_handler = MagicMock()
    return controller


class TestManualActions:
    """Manual player actions from keyboard/UI input."""

    def test_manual_action_cancels_plan_and_presentation(self) -> None:
        """Manual player action cancels active plan and presentation timing."""
        controller = _make_controller()
        player = controller.gw.player

        fake_intent = MagicMock()
        fake_intent.animation_type = None
        player.active_plan = MagicMock()

        # Queue a manual action
        controller.turn_manager._player_action_queue.append(fake_intent)

        with patch.object(
            controller, "_execute_player_action_immediately"
        ) as mock_exec:
            controller.process_player_input()

        assert player.active_plan is None
        mock_exec.assert_called_once_with(fake_intent)


class TestAutopilotGating:
    """The autopilot path has a multi-condition gate before executing plan actions."""

    def test_movement_keys_preempt_autopilot(self) -> None:
        """Movement keys generate manual actions, which take priority over plans.

        When a movement key is held, explore_mode.update() converts it into a
        manual move intent. That intent goes through the manual-action path,
        so the plan is never consulted.
        """
        controller = _make_controller()
        player = controller.gw.player

        player.active_plan = MagicMock()
        get_plan_mock = MagicMock(return_value=MagicMock(animation_type=None))
        controller.turn_manager._get_intent_from_plan = get_plan_mock

        controller.explore_mode.movement_keys.add(input_events.KeySym.UP)
        player.energy.accumulated_energy = config.ACTION_COST * 2

        with patch.object(controller, "_execute_player_action_immediately"):
            controller.process_player_input()

        # The plan was never consulted - manual input took priority
        get_plan_mock.assert_not_called()

    def test_pending_presentation_blocks_autopilot(self) -> None:
        """Autopilot waits for presentation timing to complete."""
        controller = _make_controller()
        player = controller.gw.player

        player.active_plan = MagicMock()
        player.energy.accumulated_energy = config.ACTION_COST * 2

        # Simulate pending presentation
        controller.turn_manager._pending_duration_ms = float("inf")

        with patch.object(
            controller, "_execute_player_action_immediately"
        ) as mock_exec:
            controller.process_player_input()

        mock_exec.assert_not_called()

    def test_executes_plan_action_when_all_gates_pass(self) -> None:
        """When all conditions are met, autopilot executes the plan's action."""
        controller = _make_controller()
        player = controller.gw.player

        fake_intent = MagicMock()
        fake_intent.animation_type = None
        player.active_plan = MagicMock()
        controller.turn_manager._get_intent_from_plan = MagicMock(
            return_value=fake_intent
        )

        controller.explore_mode.movement_keys.clear()
        player.energy.accumulated_energy = config.ACTION_COST * 2
        controller.turn_manager._pending_duration_ms = 0

        with patch.object(
            controller, "_execute_player_action_immediately"
        ) as mock_exec:
            controller.process_player_input()

        mock_exec.assert_called_once_with(fake_intent)

    def test_insufficient_energy_passes_time_without_moving(self) -> None:
        """When the player can't afford an action, time passes but nothing happens."""
        controller = _make_controller()
        player = controller.gw.player

        fake_intent = MagicMock()
        fake_intent.animation_type = None
        player.active_plan = MagicMock()
        controller.turn_manager._get_intent_from_plan = MagicMock(
            return_value=fake_intent
        )

        controller.explore_mode.movement_keys.clear()
        player.energy.accumulated_energy = 0
        controller.turn_manager._pending_duration_ms = 0

        with (
            patch.object(controller, "_execute_player_action_immediately") as mock_exec,
            patch.object(controller.turn_manager, "on_player_action") as mock_turn,
        ):
            controller.process_player_input()

        mock_exec.assert_not_called()
        mock_turn.assert_called_once()


class TestOverlayInteraction:
    """Interactive overlays affect mode update behavior."""

    def test_overlays_suppress_mode_updates(self) -> None:
        """When an interactive overlay is active, mode updates are skipped."""
        controller = _make_controller()

        overlay = DummyOverlay(interactive=True)
        controller.overlay_system.show_overlay(cast(Any, overlay))

        with patch.object(controller.explore_mode, "update") as mock_update:
            controller.process_player_input()

        mock_update.assert_not_called()
