import time
from dataclasses import dataclass
from typing import Any, cast

from brileta import colors, config
from brileta.controller import Controller
from brileta.environment.tile_types import TileTypeID
from brileta.events import reset_event_bus_for_testing
from brileta.game.action_plan import WalkToPlan
from brileta.game.action_router import ActionRouter
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, PC, Character
from brileta.game.actors.status_effects import StaggeredEffect
from brileta.game.enums import Disposition
from brileta.game.game_world import GameWorld
from brileta.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld, get_controller_with_player_and_map


@dataclass
class DummyControllerAutopilot(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100

    def update_fov(self) -> None:
        pass

    def run_one_turn(self) -> None:
        # Only process if there's a player turn available
        if not self.turn_manager.is_player_turn_available():
            return

        # Start of Turn phase: All actors regenerate energy and process status effects
        for actor in self.gw.actors:
            actor.update_turn(cast(Controller, self))
            if actor.energy is not None:
                actor.energy.regenerate()

        # Player action (from active_plan or queued)
        if self.gw.player:
            player = self.gw.player
            # First check for queued manual actions
            player_action = player.get_next_action(cast(Controller, self))
            # If no manual action, check for active plan
            if player_action is None and player.active_plan is not None:
                player_action = self.turn_manager._get_intent_from_plan(player)
            if player_action:
                result = self.turn_manager.execute_intent(player_action)
                # Handle plan advancement after move
                if player.active_plan is not None:
                    self.turn_manager._on_approach_result(player, result)
                if player.energy is not None:
                    player.energy.spend(self.action_cost)

        # NPC Action Resolution: Process all NPCs with sufficient energy
        for actor in list(self.gw.actors):
            if actor is self.gw.player:
                continue
            if actor.energy is not None and actor.energy.can_afford(self.action_cost):
                action = actor.get_next_action(cast(Controller, self))
                if action is not None:
                    self.turn_manager.execute_intent(action)
                    actor.energy.spend(self.action_cost)


def _make_autopilot_world() -> tuple[DummyControllerAutopilot, Character]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyControllerAutopilot(gw)
    return controller, player


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.action_cost = 100
        self.update_fov_called = False
        self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()
        self.combat_tooltip_invalidations = 0

    def invalidate_combat_tooltip(self) -> None:
        self.combat_tooltip_invalidations += 1

    def start_plan(self, *args: Any, **kwargs: Any) -> bool:
        return Controller.start_plan(cast(Controller, self), *args, **kwargs)

    def stop_plan(self, *args: Any, **kwargs: Any) -> None:
        return Controller.stop_plan(cast(Controller, self), *args, **kwargs)

    def update_fov(self) -> None:
        self.update_fov_called = True

    def run_one_turn(self) -> None:
        # Only process if there's a player turn available
        if not self.turn_manager.is_player_turn_available():
            return

        # Start of Turn phase: All actors regenerate energy and process status effects
        for actor in self.gw.actors:
            actor.update_turn(cast(Controller, self))
            if actor.energy is not None:
                actor.energy.regenerate()

        # Player action (from autopilot)
        if self.gw.player:
            player_action = self.gw.player.get_next_action(cast(Controller, self))
            if player_action:
                self.turn_manager.execute_intent(player_action)
                if self.gw.player.energy is not None:
                    self.gw.player.energy.spend(self.action_cost)

        # NPC Action Resolution: Process all NPCs with sufficient energy
        for actor in list(self.gw.actors):
            if actor is self.gw.player:
                continue
            if actor.energy is not None and actor.energy.can_afford(self.action_cost):
                action = actor.get_next_action(cast(Controller, self))
                if action is not None:
                    self.turn_manager.execute_intent(action)
                    actor.energy.spend(self.action_cost)


def _make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)
    return controller, player


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
    controller = DummyController(gw=gw)
    return controller, player, npc


def test_turn_manager_updates_fov_using_action_result() -> None:
    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    router.execute_intent(intent)
    assert controller.update_fov_called

    controller.update_fov_called = False
    # Move into an out-of-bounds tile to ensure failure without follow-ups
    intent = MoveIntent(cast(Controller, controller), player, -2, 0)
    router.execute_intent(intent)
    assert not controller.update_fov_called


def test_is_player_turn_available_with_plan() -> None:
    controller, player = _make_autopilot_world()
    tm = controller.turn_manager
    assert not tm.is_player_turn_available()
    controller.start_plan(player, WalkToPlan, target_position=(1, 0))
    assert tm.is_player_turn_available()


def test_process_unified_round_handles_active_plan() -> None:
    controller, player = _make_autopilot_world()
    tm = controller.turn_manager
    controller.start_plan(player, WalkToPlan, target_position=(1, 0))
    controller.run_one_turn()
    assert (player.x, player.y) == (1, 0)
    assert player.active_plan is None
    assert not tm.is_player_turn_available()


def test_npc_active_plan_waits_without_player_turn() -> None:
    controller, _player, npc = make_world()
    tm = controller.turn_manager
    # Hostile NPC sets a plan toward the player
    npc.ai.get_action(cast(Controller, controller), npc)
    assert npc.active_plan is not None
    assert not tm.is_player_turn_available()

    controller.run_one_turn()
    # No movement should occur without player action
    assert (npc.x, npc.y) == (3, 0)


# --- Terrain Hazard Tests ---


def test_apply_terrain_hazard_damages_actor_on_hazardous_tile() -> None:
    """Actor standing on a hazardous tile should take damage."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        2, 2, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Place an acid pool under the player
    gw.game_map.tiles[2, 2] = TileTypeID.ACID_POOL

    controller = DummyController(gw=gw)

    # No default armor - damage goes directly to HP
    initial_hp = player.health.hp

    # Apply terrain hazard
    controller.turn_manager._apply_terrain_hazard(player)

    # Player should have taken 1d4 acid damage (1-4 HP lost)
    damage_taken = initial_hp - player.health.hp
    assert 1 <= damage_taken <= 4


def test_apply_terrain_hazard_no_damage_on_safe_tile() -> None:
    """Actor standing on a safe tile should take no damage."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        2, 2, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Ensure player is on a normal floor tile
    gw.game_map.tiles[2, 2] = TileTypeID.FLOOR

    controller = DummyController(gw=gw)

    initial_hp = player.health.hp

    # Apply terrain hazard
    controller.turn_manager._apply_terrain_hazard(player)

    # Player should not have taken any damage
    assert player.health.hp == initial_hp


def test_apply_terrain_hazard_hot_coals() -> None:
    """Hot coals should deal fire damage."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        3, 3, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Place hot coals under the player
    gw.game_map.tiles[3, 3] = TileTypeID.HOT_COALS

    controller = DummyController(gw=gw)

    # No default armor - damage goes directly to HP
    initial_hp = player.health.hp

    # Apply terrain hazard
    controller.turn_manager._apply_terrain_hazard(player)

    # Player should have taken 1d6 fire damage (1-6 HP lost)
    damage_taken = initial_hp - player.health.hp
    assert 1 <= damage_taken <= 6


def test_apply_terrain_hazard_skips_actor_without_game_world() -> None:
    """Actors without a game world reference should be skipped."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    controller = DummyController(gw=gw)

    # Remove game world reference from player
    player.gw = None

    initial_hp = player.health.hp

    # Apply terrain hazard - should not crash, just skip
    controller.turn_manager._apply_terrain_hazard(player)

    # No damage should be applied
    assert player.health.hp == initial_hp


def test_npc_on_hazard_takes_damage_once_per_player_action() -> None:
    """NPC on hazardous tile takes damage once per player action, not per tick.

    Regression test: Hazard damage was being applied every tick when
    process_all_npc_reactions() was called repeatedly, instead of once
    per player action cycle.
    """
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        2, 2, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Create an NPC with high toughness to survive multiple hits from hot coals
    # (1d6 damage each) and speed=0 so they never accumulate enough energy to act
    npc = NPC(
        3,
        3,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
        toughness=20,  # max_hp = 25, survives several 1d6 hits
        speed=0,  # No energy accumulation, so NPC can never afford an action
    )
    gw.add_actor(npc)

    # Place hot coals under the NPC
    gw.game_map.tiles[3, 3] = TileTypeID.HOT_COALS

    controller = DummyController(gw=gw)
    initial_hp = npc.health.hp

    # Simulate one player action - NPC should take hazard damage once
    controller.turn_manager.on_player_action()

    first_damage = initial_hp - npc.health.hp
    assert first_damage >= 1, "NPC should take damage from hot coals"

    hp_after_first = npc.health.hp

    # Simulate multiple ticks calling process_all_npc_reactions()
    # This should NOT apply additional hazard damage
    for _ in range(5):
        controller.turn_manager.process_all_npc_reactions()

    assert npc.health.hp == hp_after_first, (
        "process_all_npc_reactions() should not apply hazard damage - "
        "only on_player_action() should"
    )

    # Another player action should apply damage again
    controller.turn_manager.on_player_action()
    second_damage = hp_after_first - npc.health.hp
    assert second_damage >= 1, "NPC should take damage again on next player action"


# --- Action Prevention Tests ---


def test_staggered_npc_action_blocked_by_turn_manager() -> None:
    """NPC with StaggeredEffect should have their action blocked.

    Regression test: The is_action_prevented() check must happen BEFORE
    update_turn(), not after. If checked after, the 1-duration effect
    will already be expired and removed, allowing the NPC to act.
    """
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    # Place hostile NPC adjacent to player - it will want to attack
    npc = NPC(
        1,
        0,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)

    controller = DummyController(gw=gw)

    # Simulate a player action to populate energy and caches
    controller.turn_manager.on_player_action()

    # Give NPC more energy to ensure they can act
    npc.energy.regenerate()
    assert npc.energy.can_afford(controller.action_cost)

    # Apply StaggeredEffect (duration=1) - should block next action
    npc.status_effects.apply_status_effect(StaggeredEffect())
    assert npc.status_effects.is_action_prevented()

    initial_hp = player.health.hp

    # Use the REAL TurnManager.process_all_npc_reactions() method
    # This is what the game actually calls, not the test helper's run_one_turn()
    controller.turn_manager.process_all_npc_reactions()

    # NPC's action should have been blocked - player takes no damage
    assert player.health.hp == initial_hp, (
        "Staggered NPC should not be able to attack - "
        "is_action_prevented() must be checked BEFORE update_turn()"
    )

    # Effect should have expired (update_turn called when blocked)
    assert not npc.status_effects.is_action_prevented()

    # NPC's energy should be depleted (can't act again until next player turn)
    assert npc.energy.accumulated_energy == 0


def test_npc_action_invalidates_combat_tooltip() -> None:
    """NPC actions should trigger combat tooltip invalidation."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        1,
        0,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)

    controller = DummyController(gw=gw)
    controller.turn_manager.on_player_action()
    npc.energy.accumulated_energy = config.ACTION_COST
    npc.get_next_action = lambda _controller: MoveIntent(_controller, npc, dx=1, dy=0)
    controller.turn_manager.execute_intent = lambda _intent: None

    controller.turn_manager.process_all_npc_reactions()

    assert controller.combat_tooltip_invalidations == 1


def test_player_action_invalidates_combat_tooltip(monkeypatch: Any) -> None:
    """Player actions should refresh combat tooltip probabilities."""
    reset_event_bus_for_testing()
    controller = get_controller_with_player_and_map()
    controller.mode_stack.append(controller.combat_mode)

    class DummyTooltip:
        def __init__(self) -> None:
            self.is_active = True

        def invalidate(self) -> None:
            return

    tooltip = DummyTooltip()
    controller.frame_manager.combat_tooltip_overlay = tooltip

    invalidations = 0

    def _invalidate() -> None:
        nonlocal invalidations
        invalidations += 1

    monkeypatch.setattr(tooltip, "invalidate", _invalidate)
    tooltip.is_active = True

    player = controller.gw.player
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = 0, 0
    for test_dx, test_dy in directions:
        target_x, target_y = player.x + test_dx, player.y + test_dy
        if (
            controller.gw.game_map.walkable[target_x, target_y]
            and controller.gw.get_actor_at_location(target_x, target_y) is None
        ):
            dx, dy = test_dx, test_dy
            break
    else:
        blocker = controller.gw.get_actor_at_location(player.x + 1, player.y)
        if blocker:
            controller.gw.remove_actor(blocker)
        dx, dy = 1, 0

    move_intent = MoveIntent(controller, player, dx=dx, dy=dy)
    controller._execute_player_action_immediately(move_intent)

    assert invalidations == 1


def test_staggered_player_action_blocked_by_controller(controller) -> None:
    """Player with StaggeredEffect should have their action blocked.

    Regression test: The is_action_prevented() check must happen BEFORE
    update_turn(), not after. If checked after, the 1-duration effect
    will already be expired and removed, allowing the player to act.
    """
    reset_event_bus_for_testing()
    player = controller.gw.player

    # Record initial position
    initial_x, initial_y = player.x, player.y

    # Apply StaggeredEffect (duration=1) - should block next action
    player.status_effects.apply_status_effect(StaggeredEffect())
    assert player.status_effects.is_action_prevented()

    # Create a move intent to move right
    move_intent = MoveIntent(controller, player, dx=1, dy=0)

    # Execute through the REAL Controller._execute_player_action_immediately()
    # This is what the game actually calls when the player presses a key
    controller._execute_player_action_immediately(move_intent)

    # Player's action should have been blocked - they didn't move
    assert player.x == initial_x, (
        "Staggered player should not be able to move - "
        "is_action_prevented() must be checked BEFORE update_turn()"
    )
    assert player.y == initial_y

    # Effect should have expired (update_turn called when blocked)
    assert not player.status_effects.is_action_prevented()


def test_npc_energy_capped_at_action_cost() -> None:
    """NPC energy should be capped at ACTION_COST to prevent double-actions.

    Regression test: NPCs were moving twice on first activation because they
    accumulated 200 energy (max_energy) across multiple player actions while
    out of range. When they finally acted, they had enough for 2 actions.

    The fix caps NPC energy at ACTION_COST (100) after each player action,
    ensuring NPCs can only ever take one action per player action.
    """
    from brileta import config

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    # Create an NPC far from the player - it will accumulate energy but not act
    # because its AI won't return an action when player is out of sight/range
    npc = NPC(
        10,
        10,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.add_actor(npc)

    controller = DummyController(gw=gw)

    # Verify NPC starts with 0 energy
    assert npc.energy is not None
    assert npc.energy.accumulated_energy == 0

    # Simulate multiple player actions - NPC would normally accumulate
    # 100 energy per action (speed 100), reaching max_energy of 200
    for _ in range(5):
        controller.turn_manager.on_player_action()

    # NPC energy should be capped at ACTION_COST, not max_energy
    # Without the fix, this would be 200 (allowing double-actions)
    assert npc.energy.accumulated_energy == config.ACTION_COST, (
        f"NPC energy should be capped at {config.ACTION_COST}, "
        f"not {npc.energy.accumulated_energy}"
    )

    # Verify player energy is NOT capped (player can store up to max_energy)
    assert player.energy is not None
    assert player.energy.accumulated_energy > config.ACTION_COST, (
        "Player energy should NOT be capped at ACTION_COST"
    )


def test_slow_npc_can_still_accumulate_to_action_cost() -> None:
    """Slow NPCs (speed < 100) should still be able to reach ACTION_COST.

    This ensures the energy cap doesn't break slow actors like Trogs (speed 80).
    They should accumulate energy over multiple player actions until they
    reach ACTION_COST, then be able to act.
    """
    from brileta import config

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Create a slow NPC (like a Trog with speed 80)
    slow_npc = NPC(
        10,
        10,
        "T",
        colors.DARK_GREY,
        "Trog",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
        speed=80,
    )
    gw.add_actor(slow_npc)

    controller = DummyController(gw=gw)

    assert slow_npc.energy is not None
    assert slow_npc.energy.accumulated_energy == 0

    # First player action: Trog gains 80 energy (can't act yet)
    controller.turn_manager.on_player_action()
    assert slow_npc.energy.accumulated_energy == 80
    assert not slow_npc.energy.can_afford(config.ACTION_COST)

    # Second player action: Trog gains 80 more, capped at 100
    controller.turn_manager.on_player_action()
    assert slow_npc.energy.accumulated_energy == config.ACTION_COST
    assert slow_npc.energy.can_afford(config.ACTION_COST)


def test_tripped_player_blocked_for_two_turns() -> None:
    """Player with TrippedEffect (duration=2) should be blocked for 2 turns."""
    from brileta.game.actors.status_effects import TrippedEffect

    reset_event_bus_for_testing()
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Find a valid move direction (walkable tile with no actor).
    # NPCs can spawn adjacent to the player, so we need to find a clear path.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = 0, 0
    for test_dx, test_dy in directions:
        target_x, target_y = player.x + test_dx, player.y + test_dy
        if (
            gw.game_map.walkable[target_x, target_y]
            and gw.get_actor_at_location(target_x, target_y) is None
        ):
            dx, dy = test_dx, test_dy
            break
    else:
        # No clear direction found - clear the tile to the right
        blocker = gw.get_actor_at_location(player.x + 1, player.y)
        if blocker:
            gw.remove_actor(blocker)
        dx, dy = 1, 0

    initial_x, initial_y = player.x, player.y

    # Apply TrippedEffect (duration=2) - should block next 2 actions
    player.status_effects.apply_status_effect(TrippedEffect())
    assert player.status_effects.is_action_prevented()

    # First attempted action - should be blocked
    move_intent_1 = MoveIntent(controller, player, dx=dx, dy=dy)
    controller._execute_player_action_immediately(move_intent_1)

    assert player.x == initial_x, "First action should be blocked"
    assert player.y == initial_y, "First action should be blocked"
    # Effect should still be active (duration was 2, now 1)
    assert player.status_effects.is_action_prevented()

    # Second attempted action - should also be blocked
    move_intent_2 = MoveIntent(controller, player, dx=dx, dy=dy)
    controller._execute_player_action_immediately(move_intent_2)

    assert player.x == initial_x, "Second action should also be blocked"
    assert player.y == initial_y, "Second action should also be blocked"
    # Effect should now be expired (duration was 1, now 0)
    assert not player.status_effects.is_action_prevented()

    # Third action - should succeed
    move_intent_3 = MoveIntent(controller, player, dx=dx, dy=dy)
    controller._execute_player_action_immediately(move_intent_3)

    assert player.x == initial_x + dx, "Third action should succeed"
    assert player.y == initial_y + dy, "Third action should succeed"


# --- ActionPlan System Tests ---


def test_is_player_turn_available_with_active_plan() -> None:
    """is_player_turn_available returns True when player has an active_plan."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan

    controller, player = _make_world()
    tm = controller.turn_manager

    # No plan - should return False
    assert not tm.is_player_turn_available()

    # Set up an active plan
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    player.active_plan = ActivePlan(plan=WalkToPlan, context=context)

    # Now should return True
    assert tm.is_player_turn_available()


def test_get_next_player_intent_from_active_plan() -> None:
    """get_next_player_intent returns intent from active_plan."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan
    from brileta.game.actions.movement import MoveIntent

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan to walk to (3, 0)
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    player.active_plan = ActivePlan(plan=WalkToPlan, context=context)

    # Get the intent - should be a MoveIntent
    intent = tm.get_next_player_intent()

    assert intent is not None
    assert isinstance(intent, MoveIntent)
    assert intent.dx == 1  # Moving right toward (3, 0)
    assert intent.dy == 0


def test_handle_approach_step_peeks_at_path() -> None:
    """_handle_approach_step peeks at cached_path[0] without popping."""
    from brileta.game.action_plan import (
        ActivePlan,
        ApproachStep,
        PlanContext,
        WalkToPlan,
    )

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan with a pre-populated path
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    active_plan = ActivePlan(plan=WalkToPlan, context=context)
    active_plan.cached_path = [(1, 0), (2, 0), (3, 0)]  # Pre-populate the path
    player.active_plan = active_plan

    # Get the approach step
    step = active_plan.get_current_step()
    assert isinstance(step, ApproachStep)

    # Handle the approach step
    intent = tm._handle_approach_step(player, active_plan, step)

    # Intent should be created
    assert intent is not None
    assert intent.dx == 1  # type: ignore[possibly-missing-attribute]
    assert intent.dy == 0  # type: ignore[possibly-missing-attribute]

    # Path should NOT be popped yet
    assert len(active_plan.cached_path) == 3
    assert active_plan.cached_path[0] == (1, 0)


def test_handle_approach_step_sets_duration_based_on_distance() -> None:
    """_handle_approach_step assigns slower durations when close to the target."""
    from brileta.game.action_plan import (
        ActivePlan,
        ApproachStep,
        PlanContext,
        WalkToPlan,
    )

    controller, player = _make_world()
    tm = controller.turn_manager

    far_context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(30, 0),
    )
    far_plan = ActivePlan(plan=WalkToPlan, context=far_context)
    far_plan.cached_path = [(1, 0)]
    player.active_plan = far_plan

    far_step = far_plan.get_current_step()
    assert isinstance(far_step, ApproachStep)

    far_intent = tm._handle_approach_step(player, far_plan, far_step)

    assert far_intent is not None
    assert isinstance(far_intent, MoveIntent)
    assert far_intent.duration_ms == 70

    # Distance=1 is the closest approach, producing maximum duration (140ms).
    near_context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(1, 0),
    )
    near_plan = ActivePlan(plan=WalkToPlan, context=near_context)
    near_plan.cached_path = [(1, 0)]
    player.active_plan = near_plan

    near_step = near_plan.get_current_step()
    assert isinstance(near_step, ApproachStep)

    near_intent = tm._handle_approach_step(player, near_plan, near_step)

    assert near_intent is not None
    assert isinstance(near_intent, MoveIntent)
    assert near_intent.duration_ms == 140


def test_on_approach_result_pops_path_on_success() -> None:
    """_on_approach_result pops from cached_path when move succeeds."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan
    from brileta.game.actions.base import GameActionResult

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan with a pre-populated path
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    active_plan = ActivePlan(plan=WalkToPlan, context=context)
    active_plan.cached_path = [(1, 0), (2, 0), (3, 0)]
    player.active_plan = active_plan

    # Simulate a successful move result
    result = GameActionResult(succeeded=True)

    # Call _on_approach_result
    tm._on_approach_result(player, result)

    # Path should be popped
    assert len(active_plan.cached_path) == 2
    assert active_plan.cached_path[0] == (2, 0)


def test_on_approach_result_invalidates_path_on_failure() -> None:
    """_on_approach_result sets cached_path to None when move fails."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan
    from brileta.game.actions.base import GameActionResult

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan with a pre-populated path
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    active_plan = ActivePlan(plan=WalkToPlan, context=context)
    active_plan.cached_path = [(1, 0), (2, 0), (3, 0)]
    player.active_plan = active_plan

    # Simulate a failed move result
    result = GameActionResult(succeeded=False)

    # Call _on_approach_result
    tm._on_approach_result(player, result)

    # Path should be invalidated (set to None for recalculation)
    assert active_plan.cached_path is None


def test_plan_completes_when_destination_reached() -> None:
    """Active plan is cleared when player reaches destination."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan
    from brileta.game.actions.base import GameActionResult

    controller, player = _make_world()
    tm = controller.turn_manager

    # Move player to (2, 0) so they're one step from destination
    player.x = 2

    # Set up an active plan to walk to (3, 0) - only one step away
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    active_plan = ActivePlan(plan=WalkToPlan, context=context)
    active_plan.cached_path = [(3, 0)]
    player.active_plan = active_plan

    # Simulate executing the move: player moves to (3, 0)
    player.x = 3

    # Simulate a successful move result
    result = GameActionResult(succeeded=True)
    tm._on_approach_result(player, result)

    # Plan should be completed and cleared
    assert player.active_plan is None


def test_execute_player_intent_handles_plan_advancement() -> None:
    """execute_player_intent calls _on_approach_result for plan advancement."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan
    from brileta.game.actions.movement import MoveIntent

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan with a path
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    active_plan = ActivePlan(plan=WalkToPlan, context=context)
    active_plan.cached_path = [(1, 0), (2, 0), (3, 0)]
    player.active_plan = active_plan

    # Create and execute a move intent
    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = tm.execute_player_intent(intent)

    # Move should succeed
    assert result.succeeded
    assert player.x == 1

    # Path should be popped (handled by execute_player_intent)
    assert len(active_plan.cached_path) == 2


def test_walk_to_plan_full_journey() -> None:
    """Player walks full journey using WalkToPlan."""
    from brileta.game.action_plan import ActivePlan, PlanContext, WalkToPlan

    controller, player = _make_world()
    tm = controller.turn_manager

    # Set up an active plan to walk to (3, 0)
    context = PlanContext(
        actor=player,
        controller=cast(Controller, controller),
        target_position=(3, 0),
    )
    player.active_plan = ActivePlan(plan=WalkToPlan, context=context)

    # Execute turns until destination reached
    for _ in range(10):  # Safety limit
        if player.active_plan is None:
            break

        intent = tm.get_next_player_intent()
        if intent is None:
            break

        tm.execute_player_intent(intent)

    # Player should be at destination
    assert player.x == 3
    assert player.y == 0

    # Plan should be completed
    assert player.active_plan is None


# --- Presentation Timing Tests ---


def test_game_action_result_has_duration_ms_field() -> None:
    """GameActionResult should have duration_ms field defaulting to 0."""
    result = GameActionResult()
    assert hasattr(result, "duration_ms")
    assert result.duration_ms == 0

    # Can set custom timing
    result_with_timing = GameActionResult(duration_ms=350)
    assert result_with_timing.duration_ms == 350


def test_turn_manager_is_presentation_complete_no_delay() -> None:
    """is_presentation_complete returns True when no delay is pending."""
    controller, _player = _make_world()
    tm = controller.turn_manager

    # Initially, no presentation is pending
    assert tm.is_presentation_complete()


def test_turn_manager_is_presentation_complete_with_delay() -> None:
    """is_presentation_complete returns False when delay is pending."""
    controller, _player = _make_world()
    tm = controller.turn_manager

    # Manually set presentation timing to simulate a just-executed action
    tm._last_action_completed_time = time.perf_counter()
    tm._pending_duration_ms = 5000  # 5 seconds - definitely not elapsed

    # Should return False because delay hasn't elapsed
    assert not tm.is_presentation_complete()


def test_turn_manager_is_presentation_complete_after_delay() -> None:
    """is_presentation_complete returns True after delay elapses."""
    controller, _player = _make_world()
    tm = controller.turn_manager

    # Set presentation timing to a value in the past
    tm._last_action_completed_time = time.perf_counter() - 1.0  # 1 second ago
    tm._pending_duration_ms = 100  # 100ms - definitely elapsed

    # Should return True because delay has elapsed
    assert tm.is_presentation_complete()


def test_turn_manager_record_action_timing() -> None:
    """_record_action_timing stores timing from GameActionResult."""
    controller, _player = _make_world()
    tm = controller.turn_manager

    result = GameActionResult(duration_ms=350)
    tm._record_action_timing(result)

    assert tm._pending_duration_ms == 350
    # Time should be recorded (recent)
    elapsed = time.perf_counter() - tm._last_action_completed_time
    assert elapsed < 0.1  # Less than 100ms ago


def test_turn_manager_clear_presentation_timing() -> None:
    """clear_presentation_timing clears pending delay."""
    controller, _player = _make_world()
    tm = controller.turn_manager

    # Set some timing
    tm._pending_duration_ms = 500
    tm._last_action_completed_time = time.perf_counter()

    # Clear it
    tm.clear_presentation_timing()

    # Should return to complete state
    assert tm._pending_duration_ms == 0
    assert tm.is_presentation_complete()


def test_execute_intent_records_timing() -> None:
    """execute_intent records presentation timing from result."""
    controller, player = _make_world()
    tm = controller.turn_manager

    # Execute a move intent with custom timing
    intent = MoveIntent(
        cast(Controller, controller), player, dx=1, dy=0, duration_ms=250
    )
    result = tm.execute_intent(intent)

    # Timing should be recorded
    assert result.duration_ms == 250
    assert tm._pending_duration_ms == result.duration_ms


def test_move_executor_returns_presentation_timing() -> None:
    """MoveExecutor returns presentation_ms in result."""
    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = router.execute_intent(intent)

    # MoveExecutor returns AUTOPILOT_MOVE_DURATION_MS (100ms) by default
    assert result.duration_ms == config.AUTOPILOT_MOVE_DURATION_MS


def test_move_intent_custom_duration_timing() -> None:
    """MoveIntent can pass custom duration_ms to executor."""
    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    # Create intent with custom timing
    intent = MoveIntent(
        cast(Controller, controller), player, dx=1, dy=0, duration_ms=150
    )
    result = router.execute_intent(intent)

    # Should use the custom timing
    assert result.duration_ms == 150
