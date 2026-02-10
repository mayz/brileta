"""Tests for the Punch combat action."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from brileta import colors
from brileta.controller import Controller
from brileta.events import reset_event_bus_for_testing
from brileta.game.actions.executors.stunts import HolsterWeaponExecutor, PunchExecutor
from brileta.game.actions.stunts import HolsterWeaponIntent, PunchIntent
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from brileta.game.items.item_types import COMBAT_KNIFE_TYPE, FISTS_TYPE
from brileta.game.resolution.d20_system import D20System
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from tests.conftest import CombatRNGPatcher, D20RNGPatcher


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


# --- Holster Weapon Tests ---


def test_holster_weapon_executor_unequips_weapon() -> None:
    """HolsterWeaponExecutor should move active weapon to inventory."""
    reset_event_bus_for_testing()
    controller, player, _ = _make_world_with_enemy()

    # Give player a weapon
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(knife, slot_index=0)
    assert player.inventory.get_active_item() == knife

    intent = HolsterWeaponIntent(cast(Controller, controller), player)
    executor = HolsterWeaponExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Weapon should now be in inventory, not equipped
    assert player.inventory.get_active_item() is None
    # Knife should still be in inventory (not dropped/destroyed)
    assert player.inventory.has_item(knife)


def test_holster_weapon_executor_succeeds_when_unarmed() -> None:
    """HolsterWeaponExecutor should succeed silently when no weapon equipped."""
    reset_event_bus_for_testing()
    controller, player, _ = _make_world_with_enemy()

    # Ensure player is unarmed
    assert player.inventory.get_active_item() is None

    intent = HolsterWeaponIntent(cast(Controller, controller), player)
    executor = HolsterWeaponExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded


def test_punch_executor_does_not_holster(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """PunchExecutor should not handle holstering - that's HolsterWeaponExecutor's job.

    In the ActionPlan system, holstering is handled by a separate step.
    PunchExecutor assumes the attacker is ready to punch.
    """
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Give player a weapon - PunchExecutor should still execute the punch
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(knife, slot_index=0)
    assert player.inventory.get_active_item() == knife

    # Force a successful roll (15) and d3 damage (2)
    with patch_combat_rng([15], [2]):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Weapon should still be equipped (PunchExecutor doesn't holster)
    assert player.inventory.get_active_item() == knife
    # Enemy should have taken damage (punch executed)
    assert enemy.health.hp == initial_hp - 2
    # No re-queuing should have happened
    assert controller.queued_actions is not None
    assert len(controller.queued_actions) == 0


# --- Punch Attack Tests ---


def test_punch_unarmed_deals_damage(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """Punching unarmed should deal d3 damage on hit."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Ensure player is unarmed
    assert player.inventory.get_active_item() is None

    # Force a successful roll (15) and d3 damage (2)
    with patch_combat_rng([15], [2]):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have taken 2 damage
    assert enemy.health.hp == initial_hp - 2


def test_punch_miss_deals_no_damage(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A missed punch deals no damage."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a low roll that will miss
    with patch_d20_rng([2]):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Enemy should not have taken damage
    assert enemy.health.hp == initial_hp


def test_punch_critical_hit(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """A natural 20 punch should be a critical hit."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a natural 20 and d3 damage of 3
    with patch_combat_rng([20], [3]):
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


def test_punch_works_diagonally(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """Punch should work on diagonally adjacent targets."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(6, 6),  # Diagonal from player
    )
    initial_hp = enemy.health.hp

    # Force success (15), damage (2)
    with patch_combat_rng([15], [2]):
        intent = PunchIntent(cast(Controller, controller), player, enemy)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.health.hp == initial_hp - 2


# --- Punch Hostility Tests ---


def test_punch_makes_non_hostile_npc_hostile(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """Punching a non-hostile NPC should make them hostile."""
    from brileta.game.actors import NPC

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
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.ai.modify_disposition(player, -10)  # Wary
    controller = DummyController(gw=gw)

    assert npc.ai.disposition_toward(player) == -10

    # Force successful punch (15), damage (2)
    with patch_combat_rng([15], [2]):
        intent = PunchIntent(cast(Controller, controller), player, npc)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert npc.ai.disposition_toward(player) == -75


def test_missed_punch_still_triggers_hostility(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A missed punch should still make the NPC hostile - the attempt is aggressive."""
    from brileta.game.actors import NPC

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
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.ai.modify_disposition(player, -10)  # Wary
    controller = DummyController(gw=gw)

    assert npc.ai.disposition_toward(player) == -10

    # Force missed punch (low roll)
    with patch_d20_rng([2]):
        intent = PunchIntent(cast(Controller, controller), player, npc)
        executor = PunchExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # The attempt was aggressive, so NPC becomes hostile
    assert npc.ai.disposition_toward(player) == -75
