"""Tests for the default action lookup system.

The defaults module provides the mapping from target types to default actions,
enabling quick right-click execution of the "obvious" action for any target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from brileta import colors
from brileta.environment.generators import GeneratedMapData
from brileta.environment.map import GameMap
from brileta.environment.tile_types import TileTypeID
from brileta.game.actions.discovery import (
    CombatIntentCache,
    TargetType,
    classify_target,
    get_default_action_id,
)
from brileta.game.actors import Character
from brileta.game.actors.container import Container
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from brileta.environment.map import MapRegion


class DummyController:
    """Minimal controller for testing."""

    def __init__(self, gw: DummyGameWorld) -> None:
        self.gw = gw
        self.frame_manager: object | None = None
        self.message_log: object | None = None
        self.combat_intent_cache: CombatIntentCache | None = None

    def create_resolver(self, **kwargs: object) -> object:
        from brileta.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]


def _make_test_world() -> tuple[DummyController, Character, Character]:
    """Create a test world with a player and NPC."""
    gw = DummyGameWorld()
    tiles = np.full((30, 30), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions: dict[int, MapRegion] = {}
    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=np.full((30, 30), -1, dtype=np.int16, order="F"),
    )
    gw.game_map = GameMap(30, 30, map_data)
    gw.game_map.gw = cast(GameWorld, gw)
    gw.game_map.visible[:] = True
    gw.items = {}

    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = Character(5, 5, "N", colors.YELLOW, "NPC", game_world=cast(GameWorld, gw))

    gw.add_actor(player)
    gw.add_actor(npc)
    gw.player = player

    controller = DummyController(gw=gw)
    return controller, player, npc


class TestClassifyTarget:
    """Tests for the classify_target function."""

    def test_classify_npc(self) -> None:
        """An NPC (non-player character) should be classified as NPC."""
        controller, _player, npc = _make_test_world()
        target_type = classify_target(controller, npc)  # type: ignore[arg-type]
        assert target_type == TargetType.NPC

    def test_classify_player_as_none(self) -> None:
        """The player should not be classifiable (returns None)."""
        controller, player, _npc = _make_test_world()
        target_type = classify_target(controller, player)  # type: ignore[arg-type]
        assert target_type is None

    def test_classify_container(self) -> None:
        """A Container actor should be classified as CONTAINER."""
        controller, _player, _ = _make_test_world()
        container = Container(
            1, 0, name="Crate", game_world=cast(GameWorld, controller.gw)
        )
        controller.gw.add_actor(container)
        target_type = classify_target(controller, container)  # type: ignore[arg-type]
        assert target_type == TargetType.CONTAINER

    def test_classify_closed_door_tile(self) -> None:
        """A closed door tile should be classified as DOOR_CLOSED."""
        controller, _player, _ = _make_test_world()
        controller.gw.game_map.tiles[3, 3] = TileTypeID.DOOR_CLOSED
        target_type = classify_target(controller, (3, 3))  # type: ignore[arg-type]
        assert target_type == TargetType.DOOR_CLOSED

    def test_classify_open_door_tile(self) -> None:
        """An open door tile should be classified as DOOR_OPEN."""
        controller, _player, _ = _make_test_world()
        controller.gw.game_map.tiles[3, 3] = TileTypeID.DOOR_OPEN
        target_type = classify_target(controller, (3, 3))  # type: ignore[arg-type]
        assert target_type == TargetType.DOOR_OPEN

    def test_classify_floor_tile(self) -> None:
        """A walkable floor tile should be classified as FLOOR."""
        controller, _player, _ = _make_test_world()
        # The tile at (10, 10) is a floor tile by default
        target_type = classify_target(controller, (10, 10))  # type: ignore[arg-type]
        assert target_type == TargetType.FLOOR

    def test_classify_item_pile(self) -> None:
        """A tile with items should be classified as ITEM_PILE."""
        controller, _player, _ = _make_test_world()
        from brileta.game.items.item_types import COMBAT_KNIFE_TYPE

        knife = COMBAT_KNIFE_TYPE.create()
        controller.gw.items[(10, 10)] = [knife]
        target_type = classify_target(controller, (10, 10))  # type: ignore[arg-type]
        assert target_type == TargetType.ITEM_PILE

    def test_classify_none_target(self) -> None:
        """None should return None."""
        controller, _player, _npc = _make_test_world()
        target_type = classify_target(controller, None)  # type: ignore[arg-type]
        assert target_type is None

    def test_classify_out_of_bounds(self) -> None:
        """Out of bounds coordinates should return None."""
        controller, _player, _npc = _make_test_world()
        target_type = classify_target(controller, (100, 100))  # type: ignore[arg-type]
        assert target_type is None

    def test_classify_tile_with_container(self) -> None:
        """A tile position containing a container should classify as CONTAINER."""
        controller, _player, _ = _make_test_world()
        container = Container(
            3, 3, name="Chest", game_world=cast(GameWorld, controller.gw)
        )
        controller.gw.add_actor(container)

        # Classify the tile position, not the actor
        target_type = classify_target(controller, (3, 3))  # type: ignore[arg-type]
        assert target_type == TargetType.CONTAINER


class TestGetDefaultActionId:
    """Tests for the get_default_action_id function."""

    def test_npc_default_is_talk(self) -> None:
        """NPC default action should be 'talk'."""
        assert get_default_action_id(TargetType.NPC) == "talk"

    def test_container_default_is_search(self) -> None:
        """Container default action should be 'search'."""
        assert get_default_action_id(TargetType.CONTAINER) == "search"

    def test_closed_door_default_is_open(self) -> None:
        """Closed door default action should be 'open'."""
        assert get_default_action_id(TargetType.DOOR_CLOSED) == "open"

    def test_open_door_default_is_close(self) -> None:
        """Open door default action should be 'close'."""
        assert get_default_action_id(TargetType.DOOR_OPEN) == "close"

    def test_item_pile_default_is_pickup(self) -> None:
        """Item pile default action should be 'pickup'."""
        assert get_default_action_id(TargetType.ITEM_PILE) == "pickup"

    def test_floor_default_is_walk(self) -> None:
        """Floor default action should be 'walk'."""
        assert get_default_action_id(TargetType.FLOOR) == "walk"


class TestControllerSelectedTarget:
    """Tests for the controller's selected_target functionality.

    Note: Full integration tests for the Controller's select_target/deselect_target
    methods require a fully initialized Controller, which is tested elsewhere.
    These tests verify the TargetType classification which is the core logic.
    """

    def test_target_type_classification_covers_all_cases(self) -> None:
        """Verify all TargetType values have default actions defined."""
        for target_type in TargetType:
            action_id = get_default_action_id(target_type)
            assert action_id is not None, f"No default action for {target_type}"


class TestExecuteDefaultActionInCombatMode:
    """Tests for combat mode behavior in execute_default_action."""

    def test_npc_default_is_attack_in_combat_mode(self) -> None:
        """In combat mode, right-clicking an NPC should attack, not talk."""
        from brileta.game.actions.discovery import execute_default_action

        controller, player, npc = _make_test_world()

        # Track what action gets queued
        queued_actions: list = []

        def mock_queue_action(intent):
            queued_actions.append(intent)

        def mock_is_combat_mode():
            return True

        controller.queue_action = mock_queue_action  # type: ignore[attr-defined]
        controller.is_combat_mode = mock_is_combat_mode  # type: ignore[attr-defined]

        # Move player adjacent to NPC for immediate action
        player.x = 4
        player.y = 5

        result = execute_default_action(controller, npc)  # type: ignore[arg-type]

        assert result is True
        assert len(queued_actions) == 1
        # Should be an AttackIntent, not TalkIntent
        from brileta.game.actions.combat import AttackIntent

        assert isinstance(queued_actions[0], AttackIntent)

    def test_npc_default_is_talk_outside_combat_mode(self) -> None:
        """Outside combat mode, right-clicking an NPC should start TalkPlan."""
        from brileta.game.actions.discovery import execute_default_action

        controller, player, npc = _make_test_world()

        def mock_is_combat_mode():
            return False

        # Mock the start_plan method to track calls
        plan_calls: list = []

        def mock_start_plan(actor, plan, target_actor=None, target_position=None, **kw):
            plan_calls.append((actor, plan, target_actor, target_position))
            return True

        controller.is_combat_mode = mock_is_combat_mode  # type: ignore[attr-defined]
        controller.start_plan = mock_start_plan  # type: ignore[attr-defined]

        # Position player (distant or adjacent - plan handles both)
        player.x = 4
        player.y = 5

        result = execute_default_action(controller, npc)  # type: ignore[arg-type]

        assert result is True
        assert len(plan_calls) == 1
        assert plan_calls[0][0] == player  # actor
        assert plan_calls[0][2] == npc  # target_actor


class TestAdjacentPositionSelection:
    """Tests that execute_default_action handles container and door targets."""

    def test_container_search_starts_search_plan(self) -> None:
        """Searching a container should start a SearchContainerPlan.

        The plan system handles finding the closest adjacent tile during
        execution via ApproachStep.
        """
        from brileta.game.actions.discovery import execute_default_action

        controller, player, _ = _make_test_world()
        container = Container(
            2, 0, name="Bookcase", game_world=cast(GameWorld, controller.gw)
        )
        controller.gw.add_actor(container)

        # Position player away from container
        player.x = 5
        player.y = 0
        controller.gw.actor_spatial_index.update(player)

        # Track plan calls
        plan_calls: list = []

        def mock_start_plan(actor, plan, target_actor=None, target_position=None, **kw):
            plan_calls.append((actor, plan, target_actor, target_position))
            return True

        def mock_is_combat_mode():
            return False

        controller.start_plan = mock_start_plan  # type: ignore
        controller.is_combat_mode = mock_is_combat_mode  # type: ignore

        result = execute_default_action(controller, container)  # type: ignore[arg-type]

        assert result is True
        assert len(plan_calls) == 1
        assert plan_calls[0][0] == player  # actor
        assert plan_calls[0][2] == container  # target_actor

    def test_container_search_works_from_different_positions(self) -> None:
        """SearchContainerPlan should start regardless of player position."""
        from brileta.game.actions.discovery import execute_default_action

        controller, player, _ = _make_test_world()
        container = Container(
            3, 5, name="Chest", game_world=cast(GameWorld, controller.gw)
        )
        controller.gw.add_actor(container)

        # Position player to the left of container
        player.x = 0
        player.y = 5
        controller.gw.actor_spatial_index.update(player)

        plan_calls: list = []

        def mock_start_plan(actor, plan, target_actor=None, target_position=None, **kw):
            plan_calls.append((actor, plan, target_actor, target_position))
            return True

        def mock_is_combat_mode():
            return False

        controller.start_plan = mock_start_plan  # type: ignore
        controller.is_combat_mode = mock_is_combat_mode  # type: ignore

        result = execute_default_action(controller, container)  # type: ignore[arg-type]

        assert result is True
        assert len(plan_calls) == 1
        assert plan_calls[0][2] == container  # target_actor

    def test_door_open_starts_plan(self) -> None:
        """When opening a door from distance, start an open door plan.

        Setup: Player at (5, 3), closed door at (2, 3)
        Expected: start_plan is called with OpenDoorPlan and door coordinates.
        """
        from brileta.game.actions.discovery import execute_default_action

        controller, player, _ = _make_test_world()

        # Create a closed door
        controller.gw.game_map.tiles[2, 3] = TileTypeID.DOOR_CLOSED

        # Position player to the right of door with a gap
        player.x = 5
        player.y = 3
        controller.gw.actor_spatial_index.update(player)

        plan_calls: list = []

        def mock_start_plan(actor, plan, target_actor=None, target_position=None, **kw):
            plan_calls.append((actor, plan, target_actor, target_position))
            return True

        def mock_is_combat_mode():
            return False

        controller.start_plan = mock_start_plan  # type: ignore
        controller.is_combat_mode = mock_is_combat_mode  # type: ignore

        result = execute_default_action(controller, (2, 3))  # type: ignore[arg-type]

        assert result is True
        assert len(plan_calls) == 1
        # Called with door position as target_position
        assert plan_calls[0][0] == player  # actor
        assert plan_calls[0][3] == (2, 3)  # target_position
