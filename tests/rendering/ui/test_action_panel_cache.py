"""Tests for action panel cache key generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.actors import Actor, Character, ItemPile
from catley.game.countables import CountableType, get_countable_display_name
from catley.game.enums import ItemSize
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemType
from catley.view.render.graphics import GraphicsContext
from catley.view.views.action_panel_view import ActionPanelView
from tests.helpers import DummyGameWorld


@dataclass
class DummyCombatMode:
    """Minimal combat mode for testing."""

    selected_action: Any = None

    def get_available_combat_actions(self, target: Any = None) -> list[ActionOption]:
        """Return empty actions by default."""
        return []

    def _ensure_valid_selection(self, actions: list[ActionOption]) -> None:
        pass


@dataclass
class DummyController:
    """Minimal controller for testing ActionPanelView."""

    gw: DummyGameWorld
    graphics: Any = None
    combat_mode: DummyCombatMode | None = None
    selected_target: Actor | None = None
    hovered_actor: Actor | None = None
    _in_combat: bool = False

    def __post_init__(self) -> None:
        if self.combat_mode is None:
            self.combat_mode = DummyCombatMode()

    def is_combat_mode(self) -> bool:
        return self._in_combat


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

    def test_cache_key_changes_when_selected_target_changes(self) -> None:
        """Cache key should change when selected target changes."""
        controller, view = make_action_panel()
        gw = controller.gw

        selected = Character(
            2, 2, "S", colors.RED, "Selected", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(selected)

        controller.selected_target = selected
        key_with_selected = view.get_cache_key()

        controller.selected_target = None
        key_without_selected = view.get_cache_key()

        assert key_with_selected != key_without_selected


class TestActionPanelSelectedTarget:
    """Tests for selected target usage in action panel."""

    def test_selected_target_preferred_over_mouse_target(self) -> None:
        """Selected target should override mouse hover for action panel data."""
        controller, view = make_action_panel()
        gw = controller.gw

        selected = Character(
            2, 2, "S", colors.RED, "Selected", game_world=cast(GameWorld, gw)
        )
        hovered = Character(
            3, 3, "H", colors.RED, "Hovered", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(selected)
        gw.add_actor(hovered)

        controller.selected_target = selected
        controller.gw.mouse_tile_location_on_map = (hovered.x, hovered.y)
        view.discovery.get_options_for_target = MagicMock(return_value=[])

        view._update_cached_data()

        assert view._cached_target_name == selected.name


class TestActionPanelHoverResolution:
    """Tests for hover target resolution behavior."""

    def test_hover_prefers_actor_over_items_for_countables(self) -> None:
        """Hover should prefer actor data when both actors and items exist."""
        controller, view = make_action_panel()
        gw = controller.gw

        item = Item(ItemType(name="Sword", description="", size=ItemSize.NORMAL))
        gw.items[(10, 10)] = [item]

        pile = ItemPile(10, 10, game_world=cast(GameWorld, gw))
        pile.inventory.add_countable(CountableType.COIN, 3)
        gw.add_actor(pile)

        gw.mouse_tile_location_on_map = (10, 10)
        view._update_cached_data()

        assert view._cached_target_name == get_countable_display_name(
            CountableType.COIN, 3
        )

    def test_selected_indicator_set_for_selected_target(self) -> None:
        """_cached_is_selected should be True when target is from selected_target."""
        controller, view = make_action_panel()
        gw = controller.gw

        selected = Character(
            2, 2, "S", colors.RED, "Selected", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(selected)

        controller.selected_target = selected
        view.discovery.get_options_for_target = MagicMock(return_value=[])

        view._update_cached_data()

        assert view._cached_is_selected is True

    def test_selected_indicator_not_set_for_hover_target(self) -> None:
        """_cached_is_selected should be False when target is from mouse hover."""
        controller, view = make_action_panel()
        gw = controller.gw

        hovered = Character(
            3, 3, "H", colors.RED, "Hovered", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(hovered)

        controller.selected_target = None
        controller.gw.mouse_tile_location_on_map = (hovered.x, hovered.y)
        view.discovery.get_options_for_target = MagicMock(return_value=[])

        view._update_cached_data()

        assert view._cached_is_selected is False


class TestCombatModeCache:
    """Tests for cache key behavior in combat mode."""

    def _make_combat_action(self, action_id: str, name: str) -> ActionOption:
        """Create a minimal combat ActionOption for testing."""
        return ActionOption(
            id=action_id,
            name=name,
            description=f"Test {name}",
            category=ActionCategory.COMBAT,
            action_class=None,
            static_params={},
        )

    def test_cache_key_changes_when_selected_combat_action_changes(self) -> None:
        """Switching the selected combat action must invalidate the cache.

        Without this, the ▶ indicator stays on the old action while
        a different action actually executes on target click.
        """
        controller, view = make_action_panel()

        attack = self._make_combat_action("attack-melee", "Attack")
        push = self._make_combat_action("push", "Push")

        # Set up combat mode with two actions
        controller._in_combat = True
        controller.combat_mode.get_available_combat_actions = MagicMock(
            return_value=[attack, push]
        )
        controller.combat_mode.selected_action = attack

        key_with_attack = view.get_cache_key()

        # Switch selected action to Push
        controller.combat_mode.selected_action = push
        key_with_push = view.get_cache_key()

        assert key_with_attack != key_with_push

    def test_combat_mode_resets_stale_explore_fields(self) -> None:
        """Entering combat mode should clear explore-mode cached fields.

        _cached_default_action_id and _cached_is_selected retain stale
        values from explore mode if not explicitly reset.
        """
        controller, view = make_action_panel()
        gw = controller.gw

        # Simulate explore mode with a selected target
        npc = Character(3, 3, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)
        controller.selected_target = npc
        view._update_cached_data()

        # Verify explore-mode values are set
        assert view._cached_is_selected is True

        # Switch to combat mode
        controller._in_combat = True
        controller.combat_mode.get_available_combat_actions = MagicMock(return_value=[])
        view._update_cached_data()

        # Stale explore-mode values should be cleared
        assert view._cached_default_action_id is None
        assert view._cached_is_selected is False
