"""Comprehensive unit tests for GameWorld class."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from catley import colors
from catley.game.actors import Actor, Character, components
from catley.game.enums import ItemSize
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemType
from catley.game.lights import DirectionalLight, DynamicLight, GlobalLight, StaticLight
from tests.helpers import DummyGameWorld

# ---------------------------------------------------------------------------
# Test Fixtures / Helpers
# ---------------------------------------------------------------------------


def make_world() -> DummyGameWorld:
    """Create a simple DummyGameWorld for testing."""
    return DummyGameWorld(width=30, height=30)


def make_actor(
    gw: DummyGameWorld, x: int = 5, y: int = 5, blocks_movement: bool = True
) -> Actor:
    """Create an actor at the given position."""
    return Actor(
        x=x,
        y=y,
        ch="A",
        color=colors.WHITE,
        name="TestActor",
        game_world=cast(GameWorld, gw),
        blocks_movement=blocks_movement,
    )


def make_character(
    gw: DummyGameWorld, x: int = 5, y: int = 5, *, alive: bool = True
) -> Character:
    """Create a character at the given position."""
    char = Character(
        x=x,
        y=y,
        ch="C",
        color=colors.WHITE,
        name="TestChar",
        game_world=cast(GameWorld, gw),
    )
    if not alive:
        # Kill the character by depleting HP.
        char.health.hp = 0
    return char


def make_static_light(x: int = 5, y: int = 5) -> StaticLight:
    """Create a StaticLight for testing."""
    return StaticLight(position=(x, y), radius=5, color=colors.WHITE)


def make_dynamic_light(
    owner: Actor | None = None, x: int = 5, y: int = 5
) -> DynamicLight:
    """Create a DynamicLight for testing, optionally attached to an owner."""
    return DynamicLight(position=(x, y), radius=5, color=colors.ORANGE, owner=owner)


def make_global_light() -> GlobalLight:
    """Create a GlobalLight (DirectionalLight) for testing."""
    return DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)


def make_mock_lighting_system() -> MagicMock:
    """Create a mock lighting system that tracks method calls."""
    mock = MagicMock()
    mock.on_light_added = MagicMock()
    mock.on_light_removed = MagicMock()
    mock.on_light_moved = MagicMock()
    mock.on_global_light_changed = MagicMock()
    return mock


def make_test_item(name: str = "TestItem") -> Item:
    """Create a simple test item."""
    item_type = ItemType(name=name, description="A test item", size=ItemSize.NORMAL)
    return item_type.create()


# ---------------------------------------------------------------------------
# TestActorManagement
# ---------------------------------------------------------------------------


class TestActorManagement:
    """Tests for actor add/remove and location queries."""

    def test_add_actor_registers_in_list(self) -> None:
        gw = make_world()
        actor = make_actor(gw)
        gw.add_actor(actor)

        assert actor in gw.actors

    def test_add_actor_registers_in_spatial_index(self) -> None:
        gw = make_world()
        actor = make_actor(gw, x=10, y=10)
        gw.add_actor(actor)

        found = gw.actor_spatial_index.get_at_point(10, 10)
        assert actor in found

    def test_add_multiple_actors_at_same_location(self) -> None:
        gw = make_world()
        actor1 = make_actor(gw, x=5, y=5, blocks_movement=True)
        actor2 = make_actor(gw, x=5, y=5, blocks_movement=False)
        gw.add_actor(actor1)
        gw.add_actor(actor2)

        assert actor1 in gw.actors
        assert actor2 in gw.actors
        found = gw.actor_spatial_index.get_at_point(5, 5)
        assert len(found) == 2

    def test_remove_actor_removes_from_list(self) -> None:
        gw = make_world()
        actor = make_actor(gw)
        gw.add_actor(actor)
        gw.remove_actor(actor)

        assert actor not in gw.actors

    def test_remove_actor_removes_from_spatial_index(self) -> None:
        gw = make_world()
        actor = make_actor(gw, x=10, y=10)
        gw.add_actor(actor)
        gw.remove_actor(actor)

        found = gw.actor_spatial_index.get_at_point(10, 10)
        assert actor not in found

    def test_remove_nonexistent_actor_does_not_raise(self) -> None:
        gw = make_world()
        actor = make_actor(gw)
        # Actor was never added - should not raise.
        gw.remove_actor(actor)

    def test_get_actor_at_location_returns_none_for_empty_tile(self) -> None:
        gw = make_world()

        result = gw.get_actor_at_location(5, 5)

        assert result is None

    def test_get_actor_at_location_returns_actor(self) -> None:
        gw = make_world()
        actor = make_actor(gw, x=5, y=5)
        gw.add_actor(actor)

        result = gw.get_actor_at_location(5, 5)

        assert result is actor

    def test_get_actor_at_location_prioritizes_blocking_actor(self) -> None:
        gw = make_world()
        non_blocking = make_actor(gw, x=5, y=5, blocks_movement=False)
        blocking = make_actor(gw, x=5, y=5, blocks_movement=True)
        # Add non-blocking first to ensure prioritization is by property, not order.
        gw.add_actor(non_blocking)
        gw.add_actor(blocking)

        result = gw.get_actor_at_location(5, 5)

        assert result is blocking

    def test_get_actor_at_location_returns_first_when_none_block(self) -> None:
        gw = make_world()
        actor1 = make_actor(gw, x=5, y=5, blocks_movement=False)
        actor2 = make_actor(gw, x=5, y=5, blocks_movement=False)
        gw.add_actor(actor1)
        gw.add_actor(actor2)

        result = gw.get_actor_at_location(5, 5)

        # Should return one of them (order depends on spatial index).
        assert result in (actor1, actor2)


# ---------------------------------------------------------------------------
# TestLightManagement
# ---------------------------------------------------------------------------


class TestLightManagement:
    """Tests for light add/remove and filtering."""

    def test_add_light_appends_to_list(self) -> None:
        gw = make_world()
        light = make_static_light()

        gw.add_light(light)

        assert light in gw.lights

    def test_add_light_notifies_lighting_system(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        light = make_static_light()

        gw.add_light(light)

        mock_system.on_light_added.assert_called_once_with(light)

    def test_add_light_works_without_lighting_system(self) -> None:
        gw = make_world()
        assert gw.lighting_system is None
        light = make_static_light()

        # Should not raise.
        gw.add_light(light)

        assert light in gw.lights

    def test_remove_light_removes_from_list(self) -> None:
        gw = make_world()
        light = make_static_light()
        gw.add_light(light)

        gw.remove_light(light)

        assert light not in gw.lights

    def test_remove_light_notifies_lighting_system(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        light = make_static_light()
        gw.add_light(light)

        gw.remove_light(light)

        mock_system.on_light_removed.assert_called_once_with(light)

    def test_remove_nonexistent_light_does_not_raise(self) -> None:
        gw = make_world()
        light = make_static_light()

        # Light was never added - should not raise.
        gw.remove_light(light)

    def test_get_global_lights_filters_global_lights_only(self) -> None:
        gw = make_world()
        static = make_static_light()
        dynamic = make_dynamic_light()
        global_light = make_global_light()
        gw.add_light(static)
        gw.add_light(dynamic)
        gw.add_light(global_light)

        result = gw.get_global_lights()

        assert global_light in result
        assert static not in result
        assert dynamic not in result

    def test_get_global_lights_returns_empty_when_none(self) -> None:
        gw = make_world()
        static = make_static_light()
        gw.add_light(static)

        result = gw.get_global_lights()

        assert result == []

    def test_get_static_lights_filters_static_non_global(self) -> None:
        gw = make_world()
        static = make_static_light()
        dynamic = make_dynamic_light()
        global_light = make_global_light()
        gw.add_light(static)
        gw.add_light(dynamic)
        gw.add_light(global_light)

        result = gw.get_static_lights()

        assert static in result
        assert dynamic not in result
        # GlobalLight is static but should be excluded.
        assert global_light not in result

    def test_get_static_lights_excludes_dynamic_lights(self) -> None:
        gw = make_world()
        dynamic = make_dynamic_light()
        gw.add_light(dynamic)

        result = gw.get_static_lights()

        assert result == []

    def test_get_static_lights_excludes_global_lights(self) -> None:
        gw = make_world()
        global_light = make_global_light()
        gw.add_light(global_light)

        result = gw.get_static_lights()

        assert result == []

    def test_on_actor_moved_updates_dynamic_light_position(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        actor = make_actor(gw, x=5, y=5)
        gw.add_actor(actor)
        dynamic_light = make_dynamic_light(owner=actor, x=5, y=5)
        gw.add_light(dynamic_light)

        # Move the actor.
        actor.x = 10
        actor.y = 10
        gw.on_actor_moved(actor)

        assert dynamic_light.position == (10, 10)
        mock_system.on_light_moved.assert_called_once_with(dynamic_light)

    def test_on_actor_moved_ignores_unowned_lights(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        actor = make_actor(gw, x=5, y=5)
        gw.add_actor(actor)
        # Light has no owner.
        unowned_light = make_dynamic_light(owner=None, x=5, y=5)
        gw.add_light(unowned_light)

        actor.x = 10
        actor.y = 10
        gw.on_actor_moved(actor)

        # Light position should not change.
        assert unowned_light.position == (5, 5)
        mock_system.on_light_moved.assert_not_called()


# ---------------------------------------------------------------------------
# TestItemOperations
# ---------------------------------------------------------------------------


class TestItemOperations:
    """Tests for item spawning and pickable item queries."""

    def test_spawn_ground_item_delegates_to_item_spawner(self) -> None:
        gw = make_world()
        item = make_test_item()

        result = gw.spawn_ground_item(item, 5, 5)

        # The spawned actor should be added to the world.
        assert result in gw.actors
        assert result.x == 5
        assert result.y == 5

    def test_spawn_ground_items_delegates_to_item_spawner(self) -> None:
        gw = make_world()
        items = [make_test_item("Item1"), make_test_item("Item2")]

        result = gw.spawn_ground_items(items, 5, 5)

        assert result in gw.actors
        assert result.x == 5
        assert result.y == 5

    def test_get_pickable_items_returns_empty_for_empty_tile(self) -> None:
        gw = make_world()

        # Call the real GameWorld implementation, not DummyGameWorld's override.
        result = GameWorld.get_pickable_items_at_location(gw, 5, 5)

        assert result == []

    def test_get_pickable_items_returns_equipped_items_from_dead_character(
        self,
    ) -> None:
        gw = make_world()
        dead_char = make_character(gw, x=5, y=5, alive=False)
        item = make_test_item("Weapon")
        dead_char.inventory.equip_to_slot(item, 0)
        gw.add_actor(dead_char)

        # Call the real GameWorld implementation, not DummyGameWorld's override.
        result = GameWorld.get_pickable_items_at_location(gw, 5, 5)

        assert item in result

    def test_get_pickable_items_ignores_alive_characters(self) -> None:
        gw = make_world()
        alive_char = make_character(gw, x=5, y=5, alive=True)
        item = make_test_item("Weapon")
        alive_char.inventory.equip_to_slot(item, 0)
        gw.add_actor(alive_char)

        # Call the real GameWorld implementation, not DummyGameWorld's override.
        result = GameWorld.get_pickable_items_at_location(gw, 5, 5)

        assert result == []

    def test_get_pickable_items_from_non_character_actors(self) -> None:
        gw = make_world()
        # Create an actor with inventory (like a pile).
        pile_actor = Actor(
            x=5,
            y=5,
            ch="*",
            color=colors.WHITE,
            name="Pile",
            game_world=cast(GameWorld, gw),
            blocks_movement=False,
            inventory=components.InventoryComponent(components.StatsComponent()),
        )
        item = make_test_item("Loot")
        assert pile_actor.inventory is not None
        pile_actor.inventory.add_to_inventory(item)
        gw.add_actor(pile_actor)

        # Call the real GameWorld implementation, not DummyGameWorld's override.
        result = GameWorld.get_pickable_items_at_location(gw, 5, 5)

        assert item in result

    def test_has_pickable_items_returns_false_for_empty_tile(self) -> None:
        gw = make_world()
        # DummyGameWorld uses self.items dict, ensure it's empty for this location.
        result = gw.has_pickable_items_at_location(5, 5)

        assert result is False

    def test_has_pickable_items_returns_true_when_items_present(self) -> None:
        gw = make_world()
        item = make_test_item("Weapon")
        # DummyGameWorld's get_pickable_items_at_location uses self.items dict.
        gw.items[(5, 5)] = [item]

        result = gw.has_pickable_items_at_location(5, 5)

        assert result is True


# ---------------------------------------------------------------------------
# TestEnvironmentMethods
# ---------------------------------------------------------------------------


class TestEnvironmentMethods:
    """Tests for environment-related methods (time of day, sky exposure)."""

    def test_set_region_sky_exposure_returns_false_without_region(self) -> None:
        gw = make_world()
        # Default DummyGameWorld has no meaningful regions.
        result = gw.set_region_sky_exposure((5, 5), 1.0)

        # Should return False since no region exists at that position.
        assert result is False

    def test_set_time_of_day_invalidates_lighting_cache(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        global_light = make_global_light()
        gw.add_light(global_light)

        gw.set_time_of_day(12.0)  # Noon

        mock_system.on_global_light_changed.assert_called()

    def test_set_time_of_day_updates_directional_light_at_noon(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        directional = DirectionalLight.create_sun(elevation_degrees=30.0, intensity=0.5)
        gw.add_light(directional)

        gw.set_time_of_day(12.0)  # Noon

        # At noon, intensity should still be positive (sun above horizon).
        assert directional.intensity > 0

    def test_set_time_of_day_sets_intensity_to_zero_at_night(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        directional = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
        gw.add_light(directional)

        # Time 22.0 (10 PM) gives time_normalized = (22-6)/12 = 1.33 > 0.5 (night).
        gw.set_time_of_day(22.0)

        # At night, sun intensity should be 0.
        assert directional.intensity == 0.0

    def test_set_time_of_day_works_without_lighting_system(self) -> None:
        gw = make_world()
        assert gw.lighting_system is None
        directional = DirectionalLight.create_sun()
        gw.add_light(directional)

        # Should not raise even without a lighting system.
        gw.set_time_of_day(6.0)  # Sunrise

    def test_set_time_of_day_at_sunrise(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        directional = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
        gw.add_light(directional)

        gw.set_time_of_day(6.0)  # Sunrise

        # At sunrise (6 AM), sun is at horizon - direction should be updated.
        mock_system.on_global_light_changed.assert_called()

    def test_set_time_of_day_at_sunset(self) -> None:
        gw = make_world()
        mock_system = make_mock_lighting_system()
        gw.lighting_system = mock_system
        directional = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
        gw.add_light(directional)

        gw.set_time_of_day(18.0)  # Sunset

        mock_system.on_global_light_changed.assert_called()
