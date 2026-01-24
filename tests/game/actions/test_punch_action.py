"""Tests for the Punch combat action."""

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.events import reset_event_bus_for_testing
from catley.game.actions.executors.stunts import PunchExecutor
from catley.game.actions.stunts import PunchIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_types import COMBAT_KNIFE_TYPE, FISTS_TYPE
from catley.game.resolution.d20_system import D20System
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for testing punch actions."""

    gw: DummyGameWorld
    frame_manager: object | None = None
    queued_actions: list[object] | None = None

    def __post_init__(self) -> None:
        self.queued_actions = []

    def create_resolver(
        self,
        ability_score: int,
        roll_to_exceed: int,
        has_advantage: bool = False,
        has_disadvantage: bool = False,
    ) -> D20System:
        return D20System(
            ability_score=ability_score,
            roll_to_exceed=roll_to_exceed,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )

    def queue_action(self, intent: object) -> None:
        """Queue an action for later execution."""
        if self.queued_actions is not None:
            self.queued_actions.append(intent)


def _make_world_with_enemy(
    player_pos: tuple[int, int] = (5, 5),
    enemy_pos: tuple[int, int] = (6, 5),
) -> tuple[DummyController, Character, Character]:
    """Create a test world with player and enemy at specified positions."""
    gw = DummyGameWorld()
    player = Character(
        player_pos[0],
        player_pos[1],
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    enemy = Character(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw)
    return controller, player, enemy


# --- Fists Weapon Tests ---


def test_fists_damage_is_d3() -> None:
    """Fists should deal d3 damage."""
    fists = FISTS_TYPE.create()
    assert fists.melee_attack is not None
    assert fists.melee_attack.damage_dice.dice_str == "d3"


def test_fists_verb_is_punch() -> None:
    """Fists should have verb 'punch'."""
    fists = FISTS_TYPE.create()
    assert fists.melee_attack is not None
    assert fists.melee_attack._spec.verb == "punch"


# --- Punch Holster Tests ---


def test_punch_with_weapon_holsters_and_requeues() -> None:
    """Punching with a weapon equipped should holster and re-queue the punch.

    This ensures the punch follows through on the next turn after holstering,
    rather than just consuming the turn with no attack.
    """
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()

    # Give player a weapon
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(knife, slot_index=0)
    assert player.inventory.get_active_item() == knife

    intent = PunchIntent(cast(Controller, controller), player, enemy)
    executor = PunchExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Weapon should now be in inventory, not equipped
    assert player.inventory.get_active_item() is None
    # Enemy should not have taken damage (holstering consumed the turn)
    # (enemy starts at full HP)

    # The punch intent should be re-queued for follow-through
    assert controller.queued_actions is not None
    assert len(controller.queued_actions) == 1
    assert controller.queued_actions[0] is intent


# --- Punch Attack Tests ---


def test_punch_unarmed_deals_damage() -> None:
    """Punching unarmed should deal d3 damage on hit."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Ensure player is unarmed
    assert player.inventory.get_active_item() is None

    # Force a successful roll and d3 damage of 2
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:  # d20 roll
            return 15  # Success
        if b == 3:  # d3 damage
            return 2
        return a

    with patch("random.randint", fixed_randint):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have taken 2 damage
    assert enemy.health.hp == initial_hp - 2


def test_punch_miss_deals_no_damage() -> None:
    """A missed punch deals no damage."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a low roll that will miss
    with patch("random.randint", return_value=2):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Enemy should not have taken damage
    assert enemy.health.hp == initial_hp


def test_punch_critical_hit() -> None:
    """A natural 20 punch should be a critical hit."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a natural 20 and d3 damage of 3
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 20
        if b == 3:
            return 3
        return a

    with patch("random.randint", fixed_randint):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have taken 3 damage
    assert enemy.health.hp == initial_hp - 3


# --- Adjacency Validation ---


def test_punch_fails_if_not_adjacent() -> None:
    """Punch should fail if target is not adjacent."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(8, 5),  # 3 tiles away
    )

    intent = PunchIntent(cast(Controller, controller), player, enemy)
    executor = PunchExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "not_adjacent"


def test_punch_works_diagonally() -> None:
    """Punch should work on diagonally adjacent targets."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(6, 6),  # Diagonal from player
    )
    initial_hp = enemy.health.hp

    # Force success
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 15
        if b == 3:
            return 2
        return a

    with patch("random.randint", fixed_randint):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.health.hp == initial_hp - 2


# --- Punch Hostility Tests ---


def test_punch_makes_non_hostile_npc_hostile() -> None:
    """Punching a non-hostile NPC should make them hostile."""
    from catley.game.actors import NPC
    from catley.game.enums import Disposition

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    # Create a wary (non-hostile) NPC
    npc = NPC(
        6,
        5,
        "n",
        colors.YELLOW,
        "Wary NPC",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.WARY,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)

    assert npc.ai.disposition == Disposition.WARY

    # Force successful punch
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 15
        if b == 3:
            return 2
        return a

    with patch("random.randint", fixed_randint):
        intent = PunchIntent(cast(Controller, controller), player, npc)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert npc.ai.disposition == Disposition.HOSTILE


def test_missed_punch_still_triggers_hostility() -> None:
    """A missed punch should still make the NPC hostile - the attempt is aggressive."""
    from catley.game.actors import NPC
    from catley.game.enums import Disposition

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        6,
        5,
        "n",
        colors.YELLOW,
        "Wary NPC",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.WARY,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)

    assert npc.ai.disposition == Disposition.WARY

    # Force missed punch (low roll)
    with patch("random.randint", return_value=2):
        intent = PunchIntent(cast(Controller, controller), player, npc)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # The attempt was aggressive, so NPC becomes hostile
    assert npc.ai.disposition == Disposition.HOSTILE
