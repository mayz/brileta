from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game import ranges
from catley.game.actions.combat import AttackIntent
from catley.game.actors import NPC, Character
from catley.game.enums import Disposition
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_world() -> tuple[DummyController, Character, NPC]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        3,
        0,
        "g",
        colors.RED,
        "Enemy",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw)
    return controller, player, npc


def test_hostile_ai_sets_active_plan() -> None:
    """HostileAI creates an active_plan to walk toward player."""
    controller, player, npc = make_world()
    action = npc.ai.get_action(controller, npc)
    assert action is None  # Returns None because plan was set
    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None
    tx, ty = plan.context.target_position
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


def test_hostile_ai_attacks_when_adjacent() -> None:
    controller, _player, npc = make_world()
    npc.x = 1
    npc.y = 0
    action = npc.ai.get_action(controller, npc)
    assert isinstance(action, AttackIntent)
    assert npc.active_plan is None


def test_hostile_ai_flees_when_low_health() -> None:
    """Low-health hostile NPCs should flee instead of attacking."""
    from catley.game.actions.movement import MoveIntent

    controller, _player, npc = make_world()
    npc.x = 1
    npc.y = 0

    # Reduce health to make fleeing score higher than attacking
    npc.take_damage(4)  # Max HP is 5 for default toughness

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert action.dx == 1
    assert action.dy == 0


def test_hostile_ai_avoids_hazardous_destination_tiles() -> None:
    """AI prefers non-hazardous tiles when selecting destination adjacent to player."""
    from catley.environment.tile_types import TileTypeID

    controller, player, npc = make_world()

    # Player at (0, 0), NPC at (3, 0)
    # The closest adjacent tile to player from NPC's perspective is (1, 0)
    # Make (1, 0) hazardous - AI should pick a different adjacent tile
    controller.gw.game_map.tiles[1, 0] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)
    assert action is None  # Returns None because plan was set

    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None

    # The destination should NOT be the hazardous tile
    tx, ty = plan.context.target_position
    assert (tx, ty) != (1, 0), "AI should avoid hazardous destination tile"

    # But it should still be adjacent to the player
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


def test_hostile_ai_uses_hazardous_tile_when_no_alternative() -> None:
    """AI will use hazardous destination tile if all options are hazardous."""
    from catley.environment.tile_types import TileTypeID

    controller, player, npc = make_world()

    # Make ALL tiles adjacent to the player hazardous
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = player.x + dx, player.y + dy
            if 0 <= tx < 30 and 0 <= ty < 30:
                controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)
    assert action is None

    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None

    # AI should still pick a destination (the least bad option)
    tx, ty = plan.context.target_position
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


# --- Hazard Escape Tests ---


def test_npc_escapes_hazard_before_attacking() -> None:
    """NPC on hazard should escape before pursuing player."""
    from catley.environment.tile_types import TileTypeID
    from catley.game.actions.movement import MoveIntent

    controller, _player, npc = make_world()

    # Place NPC on acid pool
    controller.gw.game_map.tiles[npc.x, npc.y] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    # Should return escape MoveIntent, not attack/pathfind
    assert isinstance(action, MoveIntent)


def test_npc_escapes_to_nearest_safe_tile() -> None:
    """NPC should escape to nearest non-hazardous tile, preferring orthogonal."""
    from catley.environment.tile_types import TileTypeID
    from catley.game.actions.movement import MoveIntent

    controller, _player, npc = make_world()
    npc.x, npc.y = 5, 5

    # Place NPC on hazard
    controller.gw.game_map.tiles[5, 5] = TileTypeID.ACID_POOL

    # Surround with hazards except one safe tile at (4, 5)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = npc.x + dx, npc.y + dy
            if (tx, ty) != (4, 5):
                controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL

    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    # Should escape to (4, 5) - the only safe tile
    assert action.dx == -1
    assert action.dy == 0


def test_npc_stays_if_all_adjacent_hazardous() -> None:
    """NPC stays put if all adjacent tiles are also hazards."""
    from catley.environment.tile_types import TileTypeID

    controller, _player, npc = make_world()
    npc.x, npc.y = 5, 5

    # Surround entirely with hazards (including the NPC's tile)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tx, ty = npc.x + dx, npc.y + dy
            controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    # Should fall through to normal behavior (setting active_plan)
    # since there's no escape, returns None and sets active plan
    assert action is None
    assert npc.active_plan is not None


def test_npc_skips_blocked_safe_tile() -> None:
    """NPC should skip safe tiles blocked by other actors."""
    from catley.environment.tile_types import TileTypeID
    from catley.game.actions.movement import MoveIntent

    controller, _player, npc = make_world()
    npc.x, npc.y = 5, 5

    # Place a blocking actor at (4, 5) - would be the closest safe tile
    blocker = NPC(
        4,
        5,
        "b",
        colors.RED,
        "Blocker",
        game_world=cast(GameWorld, controller.gw),
        disposition=Disposition.HOSTILE,
    )
    controller.gw.add_actor(blocker)

    # Place NPC on hazard
    controller.gw.game_map.tiles[5, 5] = TileTypeID.ACID_POOL

    # Surround with hazards except blocked (4, 5) and one other safe tile (5, 4)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = npc.x + dx, npc.y + dy
            if (tx, ty) not in [(4, 5), (5, 4)]:
                controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    # Should escape to (5, 4), not blocked (4, 5)
    assert action.dx == 0
    assert action.dy == -1
