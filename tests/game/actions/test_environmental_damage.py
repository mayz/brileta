from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from catley import colors
from catley.controller import Controller
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap
from catley.events import (
    ActorDeathEvent,
    MessageEvent,
    reset_event_bus_for_testing,
    subscribe_to_event,
)
from catley.game.actions.environmental import EnvironmentalDamageIntent
from catley.game.actions.executors.environmental import EnvironmentalDamageExecutor
from catley.game.actors import Character
from catley.game.actors.environmental import ContainedFire
from catley.game.game_world import GameWorld
from catley.view.frame_manager import FrameManager
from catley.view.render.graphics import GraphicsContext
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from catley.environment.map import MapRegion


class DummyMap(GameMap):
    def __init__(self, width: int, height: int) -> None:
        tiles = np.full((width, height), 1, dtype=np.uint8)
        regions: dict[int, MapRegion] = {}
        map_data = GeneratedMapData(
            tiles=tiles,
            regions=regions,
            tile_to_region_id=np.full((width, height), -1, dtype=np.int16),
        )
        super().__init__(width, height, map_data)
        self._transparent_map_cache = np.ones((width, height), dtype=bool)


class DummyMessageLog:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.revision = 0
        subscribe_to_event(MessageEvent, self._on_message_event)

    def _on_message_event(self, event: MessageEvent) -> None:
        self.messages.append(event.text)
        self.revision += 1


class DummyFrameManager(FrameManager):
    def __init__(self, controller: Controller) -> None:
        # Skip FrameManager.__init__ to avoid creating graphics context
        self.controller = controller

    def update(self, dt: float) -> None:
        pass

    def draw(self, gfx: GraphicsContext) -> None:
        pass


class DummyController(Controller):
    def __init__(self, gw: GameWorld) -> None:
        # Skip Controller.__init__ to avoid creating a full game
        self.gw = gw
        self.message_log = DummyMessageLog()
        self.frame_manager = DummyFrameManager(self)


def make_environmental_world() -> tuple[
    DummyController,
    ContainedFire,
    Character,
    Character,
    EnvironmentalDamageIntent,
    EnvironmentalDamageExecutor,
]:
    """Create a test world with a fire source and target characters."""
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld()
    gw.game_map = game_map
    gw.actors = []

    # Create a fire source
    fire = ContainedFire(
        5, 5, "f", colors.RED, "Campfire", game_world=cast(GameWorld, gw)
    )
    gw.add_actor(fire)

    # Create target characters
    target1 = Character(
        5, 5, "T", colors.WHITE, "Target1", game_world=cast(GameWorld, gw)
    )
    target2 = Character(
        5, 5, "U", colors.YELLOW, "Target2", game_world=cast(GameWorld, gw)
    )
    gw.add_actor(target1)
    gw.add_actor(target2)

    controller = DummyController(cast(GameWorld, gw))

    # Create environmental damage intent
    intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=3,
        damage_type="fire",
        affected_coords=[(5, 5)],
        source_description="campfire",
    )
    executor = EnvironmentalDamageExecutor()

    return controller, fire, target1, target2, intent, executor


def test_basic_environmental_damage() -> None:
    """Test basic environmental damage application."""
    reset_event_bus_for_testing()
    _controller, fire, target1, target2, intent, executor = make_environmental_world()

    # No default armor - damage goes directly to HP
    initial_hp1 = target1.health.hp
    initial_hp2 = target2.health.hp

    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert target1.health.hp == initial_hp1 - 3
    assert target2.health.hp == initial_hp2 - 3

    # Check that fire source wasn't damaged (if it has health)
    if fire.health:
        assert fire.health.hp == fire.health.max_hp


def test_environmental_damage_message_logging() -> None:
    """Test that environmental damage generates proper message log entries."""
    reset_event_bus_for_testing()
    controller, _fire, _target1, _target2, intent, executor = make_environmental_world()

    # No default armor - damage generates messages directly
    executor.execute(intent)

    messages = controller.message_log.messages
    assert len(messages) == 2  # One message per target
    assert any("Target1 takes 3 fire damage from campfire" in msg for msg in messages)
    assert any("Target2 takes 3 fire damage from campfire" in msg for msg in messages)
    assert any("HP left" in msg for msg in messages)


def test_environmental_damage_with_different_damage_types() -> None:
    """Test environmental damage with different damage types."""
    reset_event_bus_for_testing()
    controller, fire, target1, _target2, _, executor = make_environmental_world()

    # Test radiation damage
    radiation_intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=2,
        damage_type="radiation",
        affected_coords=[(5, 5)],
        source_description="contaminated zone",
    )

    initial_hp = target1.health.hp
    executor.execute(radiation_intent)

    assert target1.health.hp == initial_hp - 2

    messages = controller.message_log.messages
    assert any("radiation damage from contaminated zone" in msg for msg in messages)


def test_environmental_damage_normal_type() -> None:
    """Test environmental damage with normal damage type (no type descriptor)."""
    reset_event_bus_for_testing()
    controller, fire, _target1, _target2, _, executor = make_environmental_world()

    # Test normal damage (should not show damage type in message)
    normal_intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=4,
        damage_type="normal",
        affected_coords=[(5, 5)],
        source_description="spike trap",
    )

    executor.execute(normal_intent)

    messages = controller.message_log.messages
    assert any("takes 4 damage from spike trap" in msg for msg in messages)
    # Should not contain damage type descriptor for "normal"
    assert not any("normal damage" in msg for msg in messages)


def test_environmental_damage_multiple_coordinates() -> None:
    """Test environmental damage affecting multiple coordinates."""
    reset_event_bus_for_testing()
    controller, fire, target1, target2, _, executor = make_environmental_world()

    # Add another target at different location
    target3 = Character(
        6, 6, "V", colors.BLUE, "Target3", game_world=cast(GameWorld, controller.gw)
    )
    controller.gw.add_actor(target3)

    # No default armor - damage goes directly to HP

    # Create intent affecting multiple coordinates
    multi_coord_intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=2,
        damage_type="fire",
        affected_coords=[(5, 5), (6, 6)],
        source_description="spreading fire",
    )

    initial_hp1 = target1.health.hp
    initial_hp2 = target2.health.hp
    initial_hp3 = target3.health.hp

    executor.execute(multi_coord_intent)

    # All targets should be damaged
    assert target1.health.hp == initial_hp1 - 2
    assert target2.health.hp == initial_hp2 - 2
    assert target3.health.hp == initial_hp3 - 2

    messages = controller.message_log.messages
    assert len(messages) == 3  # One message per target


def test_environmental_damage_zero_damage() -> None:
    """Test environmental damage with zero damage amount."""
    reset_event_bus_for_testing()
    controller, fire, _target1, _target2, _, executor = make_environmental_world()

    zero_damage_intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=0,
        damage_type="fire",
        affected_coords=[(5, 5)],
        source_description="dying embers",
    )

    result = executor.execute(zero_damage_intent)

    assert result is not None
    assert not result.succeeded  # Should fail for zero damage

    # No messages should be generated
    messages = controller.message_log.messages
    assert len(messages) == 0


def test_environmental_damage_negative_damage() -> None:
    """Test environmental damage with negative damage amount."""
    reset_event_bus_for_testing()
    controller, fire, _target1, _target2, _, executor = make_environmental_world()

    negative_damage_intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=-5,
        damage_type="fire",
        affected_coords=[(5, 5)],
        source_description="healing fire",
    )

    result = executor.execute(negative_damage_intent)

    assert result is not None
    assert not result.succeeded  # Should fail for negative damage


def test_environmental_damage_no_valid_targets() -> None:
    """Test environmental damage when no valid targets are present."""
    reset_event_bus_for_testing()
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld()
    gw.game_map = game_map
    gw.actors = []

    # Only add the fire source, no targets
    fire = ContainedFire(
        5, 5, "f", colors.RED, "Campfire", game_world=cast(GameWorld, gw)
    )
    gw.add_actor(fire)

    controller = DummyController(cast(GameWorld, gw))

    intent = EnvironmentalDamageIntent(
        controller=controller,
        source_actor=fire,
        damage_amount=3,
        damage_type="fire",
        affected_coords=[(5, 5)],
        source_description="campfire",
    )
    executor = EnvironmentalDamageExecutor()

    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded  # Should succeed even with no targets

    # No messages should be generated
    messages = controller.message_log.messages
    assert len(messages) == 0


def test_environmental_damage_death_handling() -> None:
    """Test environmental damage handling actor death."""
    reset_event_bus_for_testing()
    controller, _fire, target1, target2, intent, executor = make_environmental_world()

    death_events: list[ActorDeathEvent] = []
    subscribe_to_event(ActorDeathEvent, lambda e: death_events.append(e))

    # Reduce target health to make them die from environmental damage
    target1.health.hp = 2  # Will die from 3 damage
    target2.health.hp = 10  # Will survive

    executor.execute(intent)

    # Check death handling
    assert not target1.health.is_alive()
    assert target2.health.is_alive()
    assert len(death_events) == 1
    assert death_events[0].actor == target1

    # Check death message
    messages = controller.message_log.messages
    death_messages = [msg for msg in messages if "has been killed" in msg]
    assert len(death_messages) == 1
    assert "Target1 has been killed" in death_messages[0]


def test_contained_fire_integration() -> None:
    """Test that ContainedFire generates proper EnvironmentalDamageIntent."""
    reset_event_bus_for_testing()
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld()
    gw.game_map = game_map
    gw.actors = []

    # Create campfire with specific damage
    campfire = ContainedFire.create_campfire(3, 4, cast(GameWorld, gw))
    gw.add_actor(campfire)

    controller = DummyController(cast(GameWorld, gw))

    # Test that get_next_action returns proper intent
    intent = campfire.get_next_action(controller)

    assert intent is not None
    assert isinstance(intent, EnvironmentalDamageIntent)
    assert intent.source_actor == campfire
    assert intent.damage_amount == campfire.damage_per_turn
    assert intent.damage_type == "fire"
    assert intent.affected_coords == [(3, 4)]
    assert intent.source_description == "campfire"


def test_contained_fire_torch_integration() -> None:
    """Test that different fire types generate appropriate intents."""
    reset_event_bus_for_testing()
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld()
    gw.game_map = game_map
    gw.actors = []

    # Create torch with different damage
    torch = ContainedFire.create_torch(7, 8, cast(GameWorld, gw))
    gw.add_actor(torch)

    controller = DummyController(cast(GameWorld, gw))

    intent = torch.get_next_action(controller)

    assert intent is not None
    assert isinstance(intent, EnvironmentalDamageIntent)
    assert intent.damage_amount == 3  # Torch does less damage
    assert intent.source_description == "torch"
