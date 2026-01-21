"""Tests for action panel cache key generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actors import Actor, Character
from catley.game.game_world import GameWorld
from catley.view.render.graphics import GraphicsContext
from catley.view.views.action_panel_view import ActionPanelView
from tests.helpers import DummyGameWorld


@dataclass
class DummyCombatMode:
    """Minimal combat mode for testing."""

    selected_action: Any = None


@dataclass
class DummyController:
    """Minimal controller for testing ActionPanelView."""

    gw: DummyGameWorld
    graphics: Any = None
    combat_mode: DummyCombatMode | None = None
    contextual_target: Actor | None = None

    def __post_init__(self) -> None:
        if self.combat_mode is None:
            self.combat_mode = DummyCombatMode()

    def is_combat_mode(self) -> bool:
        """Always return False for these tests - not in combat mode."""
        return False


def make_action_panel() -> tuple[DummyController, ActionPanelView]:
    """Create a minimal ActionPanelView for testing cache key generation."""
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Mock the mouse position - no target initially
    gw.mouse_tile_location_on_map = None

    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (8, 16)
    controller = DummyController(gw=gw, graphics=renderer)

    view = ActionPanelView(cast(Controller, controller))
    return controller, view


class TestActionPanelCacheKey:
    """Tests for get_cache_key() method."""

    def test_cache_key_generation_no_target(self) -> None:
        """get_cache_key() should work when no actor is under the mouse."""
        controller, view = make_action_panel()
        controller.gw.mouse_tile_location_on_map = (10, 10)

        # This should not raise any exceptions
        cache_key = view.get_cache_key()

        assert cache_key is not None
        assert isinstance(cache_key, tuple)

    def test_cache_key_generation_with_target(self) -> None:
        """get_cache_key() should work when an actor is under the mouse."""
        controller, view = make_action_panel()
        gw = controller.gw

        # Add an enemy at (3, 3)
        enemy = Character(
            3, 3, "E", colors.RED, "Enemy", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(enemy)

        # Point mouse at the enemy
        gw.mouse_tile_location_on_map = (3, 3)

        # This should not raise any exceptions
        cache_key = view.get_cache_key()

        assert cache_key is not None
        assert isinstance(cache_key, tuple)

    def test_cache_key_changes_when_target_moves_away(self) -> None:
        """Cache key should change when the target actor moves away."""
        controller, view = make_action_panel()
        gw = controller.gw

        # Add an enemy at (3, 3)
        enemy = Character(
            3, 3, "E", colors.RED, "Enemy", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(enemy)

        # Point mouse at the enemy's location
        gw.mouse_tile_location_on_map = (3, 3)
        key_with_enemy = view.get_cache_key()

        # Move enemy away (simulating them stepping back after being pushed)
        gw.actor_spatial_index.remove(enemy)
        enemy.x, enemy.y = 4, 3
        gw.actor_spatial_index.add(enemy)

        # Mouse is still at (3, 3), but enemy moved
        key_after_move = view.get_cache_key()

        assert key_with_enemy != key_after_move

    def test_cache_key_changes_when_target_moves_back(self) -> None:
        """Cache key should change when a target moves back to mouse position."""
        controller, view = make_action_panel()
        gw = controller.gw

        # Add an enemy at (4, 3) - not under mouse
        enemy = Character(
            4, 3, "E", colors.RED, "Enemy", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(enemy)

        # Point mouse at (3, 3) - empty tile
        gw.mouse_tile_location_on_map = (3, 3)
        key_no_target = view.get_cache_key()

        # Move enemy to mouse position (simulating them moving back after push)
        gw.actor_spatial_index.remove(enemy)
        enemy.x, enemy.y = 3, 3
        gw.actor_spatial_index.add(enemy)

        key_with_target = view.get_cache_key()

        assert key_no_target != key_with_target

    def test_cache_key_unchanged_for_unrelated_actor_movement(self) -> None:
        """Cache key should NOT change when an unrelated actor moves."""
        controller, view = make_action_panel()
        gw = controller.gw

        # Add enemy at (3, 3) - under the mouse
        enemy = Character(
            3, 3, "E", colors.RED, "Enemy", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(enemy)

        # Add another enemy far away
        other_enemy = Character(
            10, 10, "E", colors.RED, "Other", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(other_enemy)

        # Point mouse at the first enemy
        gw.mouse_tile_location_on_map = (3, 3)
        key_before = view.get_cache_key()

        # Move the other enemy (unrelated to mouse position)
        gw.actor_spatial_index.remove(other_enemy)
        other_enemy.x, other_enemy.y = 11, 10
        gw.actor_spatial_index.add(other_enemy)

        key_after = view.get_cache_key()

        assert key_before == key_after

    def test_cache_key_changes_when_contextual_target_changes(self) -> None:
        """Cache key should change when contextual target changes."""
        controller, view = make_action_panel()
        gw = controller.gw

        contextual = Character(
            2, 2, "C", colors.RED, "Contextual", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(contextual)

        controller.contextual_target = contextual
        key_with_contextual = view.get_cache_key()

        controller.contextual_target = None
        key_without_contextual = view.get_cache_key()

        assert key_with_contextual != key_without_contextual


class TestActionPanelContextualTarget:
    """Tests for contextual target usage in action panel."""

    def test_contextual_target_preferred_over_mouse_target(self) -> None:
        """Contextual target should override mouse hover for action panel data."""
        controller, view = make_action_panel()
        gw = controller.gw

        contextual = Character(
            2, 2, "C", colors.RED, "Contextual", game_world=cast(GameWorld, gw)
        )
        hovered = Character(
            3, 3, "H", colors.RED, "Hovered", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(contextual)
        gw.add_actor(hovered)

        controller.contextual_target = contextual
        controller.gw.mouse_tile_location_on_map = (hovered.x, hovered.y)
        view.discovery.get_options_for_target = MagicMock(return_value=[])

        view._update_cached_data()

        assert view._cached_target_name == contextual.name
