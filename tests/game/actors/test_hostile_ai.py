from dataclasses import dataclass
from typing import cast

from brileta import colors
from brileta.controller import Controller
from brileta.game import ranges
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai import escalate_hostility
from brileta.game.game_world import GameWorld
from brileta.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_world(
    disposition: int = -75,
) -> tuple[DummyController, Character, NPC]:
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
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    if disposition != 0:
        npc.ai.modify_disposition(player, disposition)
    controller = DummyController(gw)
    return controller, player, npc


def test_hostile_ai_sets_active_plan() -> None:
    """UnifiedAI creates an active_plan to walk toward player."""
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
    from brileta.game.actions.movement import MoveIntent

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
    from brileta.environment.tile_types import TileTypeID

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
    from brileta.environment.tile_types import TileTypeID

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
    from brileta.environment.tile_types import TileTypeID
    from brileta.game.actions.movement import MoveIntent

    controller, _player, npc = make_world()

    # Place NPC on acid pool
    controller.gw.game_map.tiles[npc.x, npc.y] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    # Should return escape MoveIntent, not attack/pathfind
    assert isinstance(action, MoveIntent)


def test_npc_escapes_to_nearest_safe_tile() -> None:
    """NPC should escape to nearest non-hazardous tile, preferring orthogonal."""
    from brileta.environment.tile_types import TileTypeID
    from brileta.game.actions.movement import MoveIntent

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
    from brileta.environment.tile_types import TileTypeID

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
    from brileta.environment.tile_types import TileTypeID
    from brileta.game.actions.movement import MoveIntent

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
    )
    controller.gw.add_actor(blocker)
    assert controller.gw.player is not None
    blocker.ai.set_hostile(controller.gw.player)

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


def test_neutral_npc_in_range_does_not_attack() -> None:
    """Disposition 0 should not trigger attack behavior on proximity alone."""
    from brileta.game.actors.goals import WanderGoal

    controller, _player, npc = make_world(disposition=0)
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert not isinstance(action, AttackIntent)
    assert isinstance(npc.current_goal, WanderGoal)
    assert isinstance(action, MoveIntent)


def test_numeric_hostile_disposition_attacks_in_range() -> None:
    """Numeric hostile disposition should attack when adjacent."""
    controller, _player, npc = make_world(disposition=-75)  # Hostile
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, AttackIntent)


def test_avoid_action_moves_unfriendly_npc_away() -> None:
    """Unfriendly NPCs should choose AvoidAction and step away from player."""
    from brileta.game.actions.movement import MoveIntent

    controller, _player, npc = make_world(disposition=-35)  # Unfriendly
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) == (1, 0)


def test_watch_action_returns_none_for_wary_npc() -> None:
    """Wary NPCs should specifically pick WatchAction."""
    controller, _player, npc = make_world(disposition=-20)  # Wary boundary
    npc.x = 1
    npc.y = 0

    # Remove avoid options so Watch vs Idle is the relevant decision.
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False

    action = npc.ai.get_action(controller, npc)

    assert action is None
    assert npc.active_plan is None
    assert npc.ai.last_chosen_action is not None
    assert "Watch" in npc.ai.last_chosen_action


def test_avoid_action_returns_none_when_cornered() -> None:
    """Cornered unfriendly NPCs should fall back cleanly when avoid has no step."""
    controller, _player, npc = make_world(disposition=-35)  # Unfriendly
    npc.x = 1
    npc.y = 0

    # Block all tiles that would increase distance from player at (0, 0).
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False

    action = npc.ai.get_action(controller, npc)

    assert action is None
    assert npc.ai.last_chosen_action in {"Watch", "Idle"}


def test_wander_creates_goal_when_no_threat() -> None:
    """Neutral NPC should start wander as a persistent goal when safe."""
    from brileta.game.actors.goals import WanderGoal

    controller, player, npc = make_world(disposition=0)
    player.teleport(20, 20)
    npc.teleport(5, 5)

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert isinstance(npc.current_goal, WanderGoal)


def test_wander_goal_continues_across_ticks() -> None:
    """Existing WanderGoal should be continued instead of recreated each tick."""
    from brileta.game.actors.goals import WanderGoal

    controller, player, npc = make_world(disposition=0)
    player.teleport(20, 20)
    npc.teleport(5, 5)

    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, WanderGoal)
    first_goal = npc.current_goal

    npc.ai.get_action(controller, npc)

    assert npc.current_goal is first_goal


def test_set_hostile_transitions_passive_to_aggressive() -> None:
    """set_hostile() should switch a non-hostile NPC into attack behavior."""
    controller, player, npc = make_world(disposition=40)  # Friendly
    npc.x = 1
    npc.y = 0

    first_action = npc.ai.get_action(controller, npc)
    assert not isinstance(first_action, AttackIntent)

    npc.ai.set_hostile(player)
    second_action = npc.ai.get_action(controller, npc)
    assert isinstance(second_action, AttackIntent)


def test_modify_disposition_clamps_to_valid_range() -> None:
    """modify_disposition() should clamp numeric disposition to [-100, 100]."""
    _controller, player, npc = make_world(disposition=0)

    npc.ai.modify_disposition(player, 999)
    assert npc.ai.disposition_toward(player) == 100

    npc.ai.modify_disposition(player, -999)
    assert npc.ai.disposition_toward(player) == -100


def test_hostile_npc_can_target_another_npc() -> None:
    """Hostile relationship should allow NPC-vs-NPC targeting."""
    controller, player, npc = make_world(disposition=40)  # Friendly
    npc2 = NPC(
        1,
        0,
        "r",
        colors.RED,
        "Rival",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc2)
    npc2.ai.modify_disposition(player, 40)  # Friendly toward player

    player.teleport(10, 10)
    npc.teleport(2, 0)

    npc.ai.set_hostile(npc2)

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, AttackIntent)
    assert action.defender is npc2


def test_escalate_hostility_noop_when_defender_has_no_ai() -> None:
    """Escalation should no-op when the defender is the AI-less player."""
    controller, player, npc = make_world(disposition=-75)
    assert player.ai is None

    attacker_awareness_before = npc.ai._last_attacker_id
    disposition_before = npc.ai.disposition_toward(player)

    escalate_hostility(npc, player, controller)

    assert npc.ai._last_attacker_id == attacker_awareness_before
    assert npc.ai.disposition_toward(player) == disposition_before
