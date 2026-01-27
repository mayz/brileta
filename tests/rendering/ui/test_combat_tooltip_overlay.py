"""Tests for the combat tooltip overlay.

Tests the tooltip that displays hit probability percentages when hovering
over valid combat targets during combat mode.
"""

from __future__ import annotations

from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.events import reset_event_bus_for_testing
from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.actors import NPC, Character
from catley.game.enums import Disposition
from catley.view.ui.combat_tooltip_overlay import CombatTooltipOverlay
from tests.helpers import get_controller_with_player_and_map


def _make_tooltip_test_world(
    player_pos: tuple[int, int] = (5, 5),
    enemy_pos: tuple[int, int] = (6, 5),
    *,
    enemy_alive: bool = True,
) -> tuple[Controller, Character, NPC]:
    """Create a test world with player and NPC at specified positions."""
    controller = get_controller_with_player_and_map()
    player = controller.gw.player
    gm = controller.gw.game_map

    # Move player to the specified position
    player.x = player_pos[0]
    player.y = player_pos[1]

    # Create an NPC with hostile disposition
    npc = NPC(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=controller.gw,
        disposition=Disposition.HOSTILE,
    )
    controller.gw.add_actor(npc)

    # If enemy should be dead, kill it
    if not enemy_alive and npc.health:
        npc.take_damage(npc.health.max_hp)

    # Make NPC visible
    gm.visible[npc.x, npc.y] = True

    return controller, player, npc


class TestCombatTooltipOverlayVisibility:
    """Test when the tooltip is visible or hidden."""

    def test_tooltip_hidden_when_not_in_combat_mode(self) -> None:
        """Tooltip should be hidden when not in combat mode."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_tooltip_test_world()

        # Do not enter combat mode
        overlay = CombatTooltipOverlay(controller)

        overlay._update_tooltip_content()

        # Tooltip text should be empty
        assert overlay._cached_text == ""
        assert overlay._cached_target is None

    def test_tooltip_hidden_when_no_target_under_cursor(self) -> None:
        """Tooltip should be hidden when cursor is not over a target."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_tooltip_test_world()
        controller.enter_combat_mode()

        # Select an action
        controller.combat_mode.selected_action = ActionOption(
            id="attack",
            name="Attack",
            description="Test attack",
            category=ActionCategory.COMBAT,
            action_class=None,
            requirements=[],
            static_params={},
            success_probability=0.65,
        )

        # Set mouse to an empty tile (no actor there)
        controller.gw.mouse_tile_location_on_map = (10, 10)

        overlay = CombatTooltipOverlay(controller)
        overlay._update_tooltip_content()

        assert overlay._cached_text == ""
        assert overlay._cached_target is None

    def test_tooltip_shows_probability_for_valid_target(self) -> None:
        """Tooltip should show hit probability when hovering over valid target."""
        reset_event_bus_for_testing()
        controller, _player, npc = _make_tooltip_test_world()
        controller.enter_combat_mode()

        # Create a mock action with 65% success probability
        attack_action = ActionOption(
            id="attack",
            name="Attack",
            description="Test attack",
            category=ActionCategory.COMBAT,
            action_class=None,
            requirements=[],
            static_params={},
            success_probability=0.65,
        )
        controller.combat_mode.selected_action = attack_action

        # Mock get_available_combat_actions to return our action with probability
        with patch.object(
            controller.combat_mode,
            "get_available_combat_actions",
            return_value=[attack_action],
        ):
            # Set mouse position to the NPC's location
            controller.gw.mouse_tile_location_on_map = (npc.x, npc.y)

            overlay = CombatTooltipOverlay(controller)
            overlay._update_tooltip_content()

            assert overlay._cached_text == "65%"
            assert overlay._cached_target is npc

    def test_tooltip_hidden_for_dead_target(self) -> None:
        """Tooltip should be hidden when hovering over a dead actor."""
        reset_event_bus_for_testing()
        controller, _player, npc = _make_tooltip_test_world(enemy_alive=False)
        controller.enter_combat_mode()

        # Select an action
        controller.combat_mode.selected_action = ActionOption(
            id="attack",
            name="Attack",
            description="Test attack",
            category=ActionCategory.COMBAT,
            action_class=None,
            requirements=[],
            static_params={},
            success_probability=0.65,
        )

        # Set mouse position to the dead NPC's location
        controller.gw.mouse_tile_location_on_map = (npc.x, npc.y)

        overlay = CombatTooltipOverlay(controller)
        overlay._update_tooltip_content()

        assert overlay._cached_text == ""
        assert overlay._cached_target is None

    def test_tooltip_hidden_for_player(self) -> None:
        """Tooltip should be hidden when hovering over the player character."""
        reset_event_bus_for_testing()
        controller, player, _npc = _make_tooltip_test_world()
        controller.enter_combat_mode()

        # Make player position visible
        controller.gw.game_map.visible[player.x, player.y] = True

        # Select an action
        controller.combat_mode.selected_action = ActionOption(
            id="attack",
            name="Attack",
            description="Test attack",
            category=ActionCategory.COMBAT,
            action_class=None,
            requirements=[],
            static_params={},
            success_probability=0.65,
        )

        # Set mouse position to the player's location
        controller.gw.mouse_tile_location_on_map = (player.x, player.y)

        overlay = CombatTooltipOverlay(controller)
        overlay._update_tooltip_content()

        assert overlay._cached_text == ""
        assert overlay._cached_target is None
