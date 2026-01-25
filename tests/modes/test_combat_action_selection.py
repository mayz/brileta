"""Tests for combat mode action selection.

Tests the action-centric combat UI where players select an action (Attack, Push)
before clicking on a target to execute it.
"""

from __future__ import annotations

from unittest.mock import patch

import tcod.event

from catley import colors
from catley.events import reset_event_bus_for_testing
from catley.game.actions.combat import AttackIntent
from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.actions.stunts import KickIntent, PunchIntent, PushIntent, TripIntent
from catley.game.actors import NPC
from catley.game.enums import Disposition
from tests.helpers import get_controller_with_player_and_map

# --- Helper functions ---


def _make_combat_test_world(
    player_pos: tuple[int, int] = (5, 5),
    enemy_pos: tuple[int, int] = (6, 5),
    *,
    enemy_disposition: Disposition = Disposition.HOSTILE,
):
    """Create a test world with player and NPC at specified positions."""
    controller = get_controller_with_player_and_map()
    player = controller.gw.player
    gm = controller.gw.game_map

    # Move player to the specified position
    player.x = player_pos[0]
    player.y = player_pos[1]

    # Create an NPC with the specified disposition
    npc = NPC(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=controller.gw,
        disposition=enemy_disposition,
    )
    controller.gw.add_actor(npc)

    # Make NPC and path to NPC visible and transparent for line of sight
    gm.visible[npc.x, npc.y] = True
    # Set all tiles between player and NPC to transparent for line of sight
    min_x = min(player_pos[0], enemy_pos[0])
    max_x = max(player_pos[0], enemy_pos[0])
    min_y = min(player_pos[1], enemy_pos[1])
    max_y = max(player_pos[1], enemy_pos[1])
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            gm.transparent[x, y] = True

    return controller, player, npc


# --- CombatMode Action Selection Tests ---


class TestCombatModeActionSelection:
    """Test selected_action state management on CombatMode."""

    def test_default_action_is_attack_on_enter(self) -> None:
        """selected_action should default to Attack when entering combat mode."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - so actions are available
        )
        controller.enter_combat_mode()

        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.category == ActionCategory.COMBAT

    def test_select_action_updates_state(self) -> None:
        """select_action() should update selected_action."""
        reset_event_bus_for_testing()
        controller = get_controller_with_player_and_map()
        controller.enter_combat_mode()

        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)

        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.id == "push"

    def test_selected_action_resets_on_exit(self) -> None:
        """Exiting combat mode should reset selected_action to None."""
        reset_event_bus_for_testing()
        controller = get_controller_with_player_and_map()
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)
        assert controller.combat_mode.selected_action is not None

        # Exit combat
        controller.exit_combat_mode("manual_exit")

        # selected_action should be cleared
        assert controller.combat_mode.selected_action is None

    def test_selected_action_resets_on_reenter(self) -> None:
        """Re-entering combat should reset to default Attack."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - so actions are available
        )
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)
        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.id == "push"

        # Exit and re-enter
        controller.exit_combat_mode("manual_exit")
        controller.enter_combat_mode()

        # Should be back to Attack (COMBAT category)
        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.category == ActionCategory.COMBAT


# --- Intent Creation Tests ---


class TestCombatModeIntentCreation:
    """Test _create_intent_for_target creates correct intent types."""

    def test_attack_action_starts_melee_attack_plan(self) -> None:
        """Selected melee attack action should start MeleeAttackPlan.

        Melee attacks now use the ActionPlan system which handles
        approach uniformly.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),
        )
        controller.enter_combat_mode()

        # Select a melee attack action explicitly
        melee_action = ActionOption(
            id="melee-test",
            name="Strike",
            description="Melee attack",
            category=ActionCategory.COMBAT,
            action_class=AttackIntent,
            requirements=[],
            static_params={"attack_mode": "melee", "weapon": None},
        )
        controller.combat_mode.select_action(melee_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Melee attacks now use the plan system
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Melee Attack"
        assert player.active_plan.context.target_actor == npc

    def test_push_action_starts_push_plan(self) -> None:
        """Selected Push action should start PushPlan for adjacent target.

        Push uses the ActionPlan system which handles both adjacent and
        distant targets uniformly. For adjacent targets, the ApproachStep
        is skipped and the push executes on the next turn.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Push now uses the plan system
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Push"
        assert player.active_plan.context.target_actor == npc

    def test_push_on_distant_target_starts_push_plan(self) -> None:
        """Push on non-adjacent target should start PushPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Should return None (plan system handles approach)
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Push"
        assert player.active_plan.context.target_actor == npc

    def test_trip_action_starts_trip_plan(self) -> None:
        """Selected Trip action should start TripPlan for adjacent target.

        Trip uses the ActionPlan system which handles both adjacent and
        distant targets uniformly.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Trip
        trip_action = ActionOption(
            id="trip",
            name="Trip",
            description="Knock target prone.",
            category=ActionCategory.STUNT,
            action_class=TripIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(trip_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Trip now uses the plan system
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Trip"
        assert player.active_plan.context.target_actor == npc

    def test_trip_on_distant_target_starts_trip_plan(self) -> None:
        """Trip on non-adjacent target should start TripPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select Trip
        trip_action = ActionOption(
            id="trip",
            name="Trip",
            description="Knock target prone.",
            category=ActionCategory.STUNT,
            action_class=TripIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(trip_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Should return None (plan system handles approach)
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Trip"
        assert player.active_plan.context.target_actor == npc

    def test_kick_action_starts_kick_plan(self) -> None:
        """Selected Kick action should start KickPlan for adjacent target.

        Kick uses the ActionPlan system which handles both adjacent and
        distant targets uniformly.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Kick
        kick_action = ActionOption(
            id="kick",
            name="Kick",
            description="Kick target, dealing damage and pushing them back.",
            category=ActionCategory.STUNT,
            action_class=KickIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(kick_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Kick now uses the plan system
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Kick"
        assert player.active_plan.context.target_actor == npc

    def test_kick_on_distant_target_starts_kick_plan(self) -> None:
        """Kick on non-adjacent target should start KickPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select Kick
        kick_action = ActionOption(
            id="kick",
            name="Kick",
            description="Kick target, dealing damage and pushing them back.",
            category=ActionCategory.STUNT,
            action_class=KickIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(kick_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Should return None (plan system handles approach)
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Kick"
        assert player.active_plan.context.target_actor == npc

    def test_punch_action_starts_punch_plan(self) -> None:
        """Selected Punch action should start a PunchPlan for adjacent target.

        Punch uses the ActionPlan system instead of creating intents directly.
        The plan handles approach, holstering, and the punch attack as separate steps.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Punch
        punch_action = ActionOption(
            id="punch",
            name="Punch",
            description="Punch target, potentially stunning them.",
            category=ActionCategory.STUNT,
            action_class=PunchIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(punch_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Punch now uses the plan system, so returns None (no direct intent)
        assert intent is None, "Punch should use plan system, not return an intent"
        # Player should have an active plan
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Punch"
        assert player.active_plan.context.target_actor == npc

    def test_punch_on_distant_target_starts_punch_plan(self) -> None:
        """Punch on non-adjacent target should start PunchPlan with approach step.

        The PunchPlan includes an ApproachStep that handles pathfinding internally,
        so no explicit start_actor_pathfinding call is made.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select Punch
        punch_action = ActionOption(
            id="punch",
            name="Punch",
            description="Punch target, potentially stunning them.",
            category=ActionCategory.STUNT,
            action_class=PunchIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(punch_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Plan system handles approach internally
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Punch"
        # The plan's first step should be ApproachStep
        from catley.game.action_plan import ApproachStep

        first_step = player.active_plan.plan.steps[0]
        assert isinstance(first_step, ApproachStep)

    def test_no_selected_action_falls_back_to_melee_attack_plan(self) -> None:
        """If no action is selected, should fall back to MeleeAttackPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world()
        controller.enter_combat_mode()

        # Force selected_action to None
        controller.combat_mode.selected_action = None

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Falls back to melee attack plan
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Melee Attack"

    def test_no_selected_action_on_distant_target_starts_melee_attack_plan(
        self,
    ) -> None:
        """Default attack on distant target should start MeleeAttackPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # No action selected - uses default attack
        controller.combat_mode.selected_action = None

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Should return None (plan system handles approach)
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Melee Attack"
        assert player.active_plan.context.target_actor == npc


# --- Hotkey Selection Tests ---


class TestCombatModeHotkeySelection:
    """Test hotkey input changes selected action."""

    def test_pressing_action_hotkey_selects_action(self) -> None:
        """Pressing an action's hotkey should select that action."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - so we get both Attack and Push
        )
        controller.enter_combat_mode()

        # Get available actions to find what hotkey 'b' maps to
        actions = controller.combat_mode.get_available_combat_actions()
        assert len(actions) >= 2, "Need at least 2 actions (Attack and Push)"

        # First action should be 'a', second should be 'b'
        # Press 'b' to select the second action
        event = tcod.event.KeyDown(
            sym=ord("b"),
            scancode=tcod.event.Scancode.B,
            mod=tcod.event.Modifier.NONE,
        )
        handled = controller.combat_mode.handle_input(event)

        assert handled
        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.id == actions[1].id

    def test_invalid_hotkey_returns_false(self) -> None:
        """Pressing a key with no mapped action should return False."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),
        )
        controller.enter_combat_mode()

        # Press 'z' which should not map to any action (only 2-3 actions exist)
        event = tcod.event.KeyDown(
            sym=ord("z"),
            scancode=tcod.event.Scancode.Z,
            mod=tcod.event.Modifier.NONE,
        )
        handled = controller.combat_mode.handle_input(event)

        assert not handled

    def test_first_action_hotkey_selects_first_action(self) -> None:
        """Pressing 'a' should select the first action (Attack)."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - so we get multiple actions
        )
        controller.enter_combat_mode()

        # First select something else to verify 'a' changes it back
        actions = controller.combat_mode.get_available_combat_actions()
        if len(actions) >= 2:
            controller.combat_mode.select_action(actions[1])

        # Press 'a' to select first action
        event = tcod.event.KeyDown(
            sym=ord("a"),
            scancode=tcod.event.Scancode.A,
            mod=tcod.event.Modifier.NONE,
        )
        handled = controller.combat_mode.handle_input(event)

        assert handled
        assert controller.combat_mode.selected_action is not None
        assert controller.combat_mode.selected_action.id == actions[0].id


# --- Action Discovery Tests ---


class TestPlayerCombatActionsDiscovery:
    """Test get_player_combat_actions returns correct action set."""

    def test_returns_attack_actions_for_equipped_weapon(self) -> None:
        """Should include attack options for active weapon when enemy adjacent."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )

        actions = controller.combat_mode.get_available_combat_actions()

        attack_actions = [a for a in actions if a.category == ActionCategory.COMBAT]
        assert len(attack_actions) >= 1

    def test_returns_push_stunt_when_adjacent_enemy(self) -> None:
        """Should include Push when an enemy is adjacent."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )

        actions = controller.combat_mode.get_available_combat_actions()

        push_actions = [a for a in actions if a.id == "push"]
        assert len(push_actions) == 1

    def test_push_action_has_correct_category(self) -> None:
        """Push action should be in STUNT category."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )

        actions = controller.combat_mode.get_available_combat_actions()

        push_action = next((a for a in actions if a.id == "push"), None)
        assert push_action is not None
        assert push_action.category == ActionCategory.STUNT

    def test_returns_trip_stunt_when_adjacent_enemy(self) -> None:
        """Should include Trip when an enemy is adjacent."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )

        actions = controller.combat_mode.get_available_combat_actions()

        trip_actions = [a for a in actions if a.id == "trip"]
        assert len(trip_actions) == 1
        assert trip_actions[0].category == ActionCategory.STUNT
        assert trip_actions[0].action_class == TripIntent

    def test_actions_have_valid_action_class(self) -> None:
        """Combat actions should have valid action_class set."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )

        actions = controller.combat_mode.get_available_combat_actions()

        for action in actions:
            assert action.action_class is not None

    def test_preferred_attacks_come_before_improvised(self) -> None:
        """PREFERRED attacks should be sorted before IMPROVISED attacks.

        For a pistol: Shoot (PREFERRED) should come before Pistol-whip (IMPROVISED).
        This ensures the default selection and [A] hotkey go to the primary attack.
        """
        from catley.game.items.item_types import PISTOL_TYPE

        reset_event_bus_for_testing()
        controller, player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - both attacks available
        )

        # Equip a pistol (has PREFERRED ranged and IMPROVISED melee)
        pistol = PISTOL_TYPE.create()
        player.inventory.add_to_inventory(pistol)
        player.inventory.equip_to_slot(pistol, 0)

        actions = controller.combat_mode.get_available_combat_actions()

        # Filter to just COMBAT actions (excludes Push which is STUNT)
        combat_actions = [a for a in actions if a.category == ActionCategory.COMBAT]
        assert len(combat_actions) == 2, "Expected both Shoot and Pistol-whip"

        # Shoot (PREFERRED) should come before Pistol-whip (IMPROVISED)
        assert combat_actions[0].name == "Shoot"
        assert combat_actions[1].name == "Pistol-whip"


# --- Enter Key Execution Tests ---


class TestCombatModeEnterKeyExecution:
    """Test Enter key executes selected action on current target."""

    def test_enter_key_queues_attack_intent(self) -> None:
        """Pressing Enter with Attack selected should queue AttackIntent."""
        reset_event_bus_for_testing()
        controller, _player, npc = _make_combat_test_world()
        controller.enter_combat_mode()

        # Manually set current target
        controller.combat_mode.candidates = [npc]
        controller.combat_mode.current_index = 0

        # Press Enter
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.RETURN,
            scancode=tcod.event.Scancode.RETURN,
            mod=tcod.event.Modifier.NONE,
        )

        # Track queued actions
        with patch.object(controller, "queue_action") as mock_queue:
            controller.combat_mode.handle_input(event)

            assert mock_queue.called
            queued_intent = mock_queue.call_args[0][0]
            assert isinstance(queued_intent, AttackIntent)

    def test_enter_key_with_push_starts_push_plan(self) -> None:
        """Pressing Enter with Push selected should start PushPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)

        # Manually set current target
        controller.combat_mode.candidates = [npc]
        controller.combat_mode.current_index = 0

        # Press Enter
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.RETURN,
            scancode=tcod.event.Scancode.RETURN,
            mod=tcod.event.Modifier.NONE,
        )

        # Push now uses the plan system
        controller.combat_mode.handle_input(event)

        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Push"

    def test_enter_key_with_trip_starts_trip_plan(self) -> None:
        """Pressing Enter with Trip selected should start TripPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Trip
        trip_action = ActionOption(
            id="trip",
            name="Trip",
            description="Knock target prone.",
            category=ActionCategory.STUNT,
            action_class=TripIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(trip_action)

        # Manually set current target
        controller.combat_mode.candidates = [npc]
        controller.combat_mode.current_index = 0

        # Press Enter
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.RETURN,
            scancode=tcod.event.Scancode.RETURN,
            mod=tcod.event.Modifier.NONE,
        )

        # Trip now uses the plan system
        controller.combat_mode.handle_input(event)

        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Trip"

    def test_enter_key_with_kick_starts_kick_plan(self) -> None:
        """Pressing Enter with Kick selected should start KickPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Kick
        kick_action = ActionOption(
            id="kick",
            name="Kick",
            description="Kick target, dealing damage and pushing them back.",
            category=ActionCategory.STUNT,
            action_class=KickIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(kick_action)

        # Manually set current target
        controller.combat_mode.candidates = [npc]
        controller.combat_mode.current_index = 0

        # Press Enter
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.RETURN,
            scancode=tcod.event.Scancode.RETURN,
            mod=tcod.event.Modifier.NONE,
        )

        # Kick now uses the plan system
        controller.combat_mode.handle_input(event)

        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Kick"

    def test_enter_key_with_punch_starts_punch_plan(self) -> None:
        """Pressing Enter with Punch selected should start PunchPlan."""
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select Punch
        punch_action = ActionOption(
            id="punch",
            name="Punch",
            description="Punch target, potentially stunning them.",
            category=ActionCategory.STUNT,
            action_class=PunchIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(punch_action)

        # Manually set current target
        controller.combat_mode.candidates = [npc]
        controller.combat_mode.current_index = 0

        # Press Enter
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.RETURN,
            scancode=tcod.event.Scancode.RETURN,
            mod=tcod.event.Modifier.NONE,
        )

        # Punch now uses the plan system, not queue_action
        controller.combat_mode.handle_input(event)

        # Player should have an active punch plan
        assert player.active_plan is not None, (
            "Expected PunchPlan to be set on player. "
            "Punch action should use the ActionPlan system."
        )
        assert player.active_plan.plan.name == "Punch"


# --- Combat Action Filtering Tests ---


class TestCombatActionFiltering:
    """Test that only executable actions are shown in the action panel."""

    def test_melee_shown_even_when_no_adjacent_enemy(self) -> None:
        """Melee attack should appear even when no enemy is adjacent.

        Melee actions are shown regardless of adjacency - approach is handled
        on execution via pathfinding.
        """
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(10, 5),  # Far away (distance 5)
        )

        actions = controller.combat_mode.get_available_combat_actions()

        melee_actions = [
            a for a in actions if a.static_params.get("attack_mode") == "melee"
        ]
        assert len(melee_actions) >= 1

    def test_push_shown_even_when_no_adjacent_enemy(self) -> None:
        """Push should appear even when no enemy is adjacent.

        Push is shown regardless of adjacency - approach is handled on execution
        via pathfinding.
        """
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(10, 5),  # Far away (distance 5)
        )

        actions = controller.combat_mode.get_available_combat_actions()

        push_actions = [a for a in actions if a.id == "push"]
        assert len(push_actions) == 1

    def test_melee_and_push_appear_when_adjacent(self) -> None:
        """Melee and Push should appear when enemy is adjacent."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent (distance 1)
        )

        actions = controller.combat_mode.get_available_combat_actions()

        melee_actions = [
            a for a in actions if a.static_params.get("attack_mode") == "melee"
        ]
        push_actions = [a for a in actions if a.id == "push"]
        assert len(melee_actions) >= 1
        assert len(push_actions) == 1

    def test_push_shown_even_when_no_enemies(self) -> None:
        """Push should always appear, even when no enemies exist.

        Actions are shown based on capability, not immediate executability.
        Push is available as a stunt option regardless of targets.
        """
        reset_event_bus_for_testing()
        controller = get_controller_with_player_and_map()

        actions = controller.combat_mode.get_available_combat_actions()

        # Push is always available as a capability
        push_actions = [a for a in actions if a.id == "push"]
        assert len(push_actions) == 1

    def test_no_ranged_action_when_enemy_beyond_max_range(self) -> None:
        """Ranged attack should not appear when enemy is beyond weapon's max range."""
        from catley.game.items.item_types import PISTOL_TYPE

        reset_event_bus_for_testing()
        # Pistol has max_range=12. Place enemy at distance 15.
        controller, player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(20, 5),  # Distance 15, beyond pistol max_range of 12
        )

        # Equip player with a pistol
        pistol = PISTOL_TYPE.create()
        pistol.ranged_attack.current_ammo = 6  # type: ignore[union-attr]
        player.inventory.equip_to_slot(pistol, slot_index=0)

        actions = controller.combat_mode.get_available_combat_actions()

        # Should have no ranged actions (enemy is out of range)
        ranged_actions = [
            a for a in actions if a.static_params.get("attack_mode") == "ranged"
        ]
        assert len(ranged_actions) == 0, (
            f"Expected no ranged actions, got {ranged_actions}"
        )

    def test_ranged_action_appears_when_enemy_within_max_range(self) -> None:
        """Ranged attack should appear when enemy is within weapon's max range."""
        from catley.game.items.item_types import PISTOL_TYPE

        reset_event_bus_for_testing()
        # Pistol has max_range=12. Place enemy at distance 10.
        controller, player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(15, 5),  # Distance 10, within pistol max_range of 12
        )

        # Equip player with a pistol
        pistol = PISTOL_TYPE.create()
        pistol.ranged_attack.current_ammo = 6  # type: ignore[union-attr]
        player.inventory.equip_to_slot(pistol, slot_index=0)

        actions = controller.combat_mode.get_available_combat_actions()

        # Should have a ranged action (enemy is in range)
        ranged_actions = [
            a for a in actions if a.static_params.get("attack_mode") == "ranged"
        ]
        assert len(ranged_actions) >= 1, (
            "Expected ranged action when enemy is within range"
        )


# --- Click Selection Tests ---


class TestCombatModeClickSelection:
    """Test clicking on actions in the action panel to select them."""

    def test_click_on_action_selects_it(self) -> None:
        """Clicking an action in the panel should select it."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent - so actions are available
        )
        controller.enter_combat_mode()

        # Get available actions so we know what could be selected
        actions = controller.combat_mode.get_available_combat_actions()
        assert len(actions) >= 2, "Need at least 2 actions for this test"

        # Mock the frame_manager and action_panel_view
        second_action = actions[1]
        with patch.object(
            controller,
            "frame_manager",
            create=True,
        ) as mock_fm:
            # Configure the mock action_panel_view to return the second action
            mock_fm.action_panel_view.x = 0
            mock_fm.action_panel_view.y = 0
            mock_fm.action_panel_view.width = 10
            mock_fm.action_panel_view.height = 5
            mock_fm.action_panel_view.get_action_at_pixel.return_value = second_action

            # Create a click event at position (50, 30) pixels
            event = tcod.event.MouseButtonDown(
                (50, 30), (50, 30), tcod.event.MouseButton.LEFT
            )

            result = controller.combat_mode._try_select_action_by_click(event)

            assert result is True
            assert controller.combat_mode.selected_action is not None
            assert controller.combat_mode.selected_action.id == second_action.id

    def test_click_outside_panel_returns_false(self) -> None:
        """Clicks outside the action panel should not select anything."""
        reset_event_bus_for_testing()
        controller, _player, _npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),
        )
        controller.enter_combat_mode()

        # Remember the initially selected action
        initial_action = controller.combat_mode.selected_action

        with patch.object(
            controller,
            "frame_manager",
            create=True,
        ) as mock_fm:
            # Configure panel bounds: panel is at (100, 100) with 160x80 pixel size
            # (10x5 tiles * 16 pixels/tile)
            mock_fm.action_panel_view.x = 100  # Panel starts at tile x=100
            mock_fm.action_panel_view.y = 100  # Panel starts at tile y=100
            mock_fm.action_panel_view.width = 10  # 10 tiles wide
            mock_fm.action_panel_view.height = 5  # 5 tiles tall

            # Click at (50, 50) pixels - outside the panel which starts at (1600, 1600)
            event = tcod.event.MouseButtonDown(
                (50, 50), (50, 50), tcod.event.MouseButton.LEFT
            )

            result = controller.combat_mode._try_select_action_by_click(event)

            assert result is False
            # Selected action should remain unchanged
            assert controller.combat_mode.selected_action == initial_action

    def test_click_selection_without_frame_manager_returns_false(self) -> None:
        """Guard clause should return False when frame_manager is None."""
        reset_event_bus_for_testing()
        controller = get_controller_with_player_and_map()
        controller.enter_combat_mode()

        # Set frame_manager to None
        controller.frame_manager = None  # type: ignore[assignment]

        event = tcod.event.MouseButtonDown(
            (50, 30), (50, 30), tcod.event.MouseButton.LEFT
        )

        result = controller.combat_mode._try_select_action_by_click(event)

        assert result is False


# --- Approach and Attack Tests ---


class TestApproachAndAttack:
    """Test approach-and-attack behavior for distant targets."""

    def test_melee_on_distant_target_starts_melee_attack_plan(self) -> None:
        """Melee attack on non-adjacent target should start MeleeAttackPlan."""
        from catley.game.actions.combat import AttackIntent

        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select a melee attack action
        melee_action = ActionOption(
            id="melee-test",
            name="Strike",
            description="Melee attack",
            category=ActionCategory.COMBAT,
            action_class=AttackIntent,
            requirements=[],
            static_params={"attack_mode": "melee", "weapon": None},
        )
        controller.combat_mode.select_action(melee_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Should return None (plan system handles approach)
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Melee Attack"
        assert player.active_plan.context.target_actor == npc

    def test_melee_on_adjacent_target_starts_melee_attack_plan(self) -> None:
        """Melee attack on adjacent target should start MeleeAttackPlan.

        Melee attacks use the ActionPlan system uniformly for both adjacent
        and distant targets. For adjacent targets, the ApproachStep is skipped.
        """
        from catley.game.actions.combat import AttackIntent

        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        # Select a melee attack action
        melee_action = ActionOption(
            id="melee-test",
            name="Strike",
            description="Melee attack",
            category=ActionCategory.COMBAT,
            action_class=AttackIntent,
            requirements=[],
            static_params={"attack_mode": "melee", "weapon": None},
        )
        controller.combat_mode.select_action(melee_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Melee attacks now use the plan system
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Melee Attack"
        assert player.active_plan.context.target_actor == npc

    def test_ranged_on_distant_target_executes_immediately(self) -> None:
        """Ranged attack should execute immediately regardless of distance."""
        from catley.game.actions.combat import AttackIntent

        reset_event_bus_for_testing()
        controller, _player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(10, 5),  # 5 tiles away
        )
        controller.enter_combat_mode()

        # Select a ranged attack action
        ranged_action = ActionOption(
            id="ranged-test",
            name="Shoot",
            description="Ranged attack",
            category=ActionCategory.COMBAT,
            action_class=AttackIntent,
            requirements=[],
            static_params={"attack_mode": "ranged", "weapon": None},
        )
        controller.combat_mode.select_action(ranged_action)

        # Should NOT start pathfinding for ranged attack
        with patch.object(controller, "start_actor_pathfinding") as mock_pathfind:
            intent = controller.combat_mode._create_intent_for_target(npc)

            # Should return intent for immediate execution
            assert isinstance(intent, AttackIntent)
            assert intent.defender == npc

            # Pathfinding should NOT have been called
            mock_pathfind.assert_not_called()

    def test_push_on_target_starts_plan_regardless_of_path(self) -> None:
        """Push should start a plan regardless of whether path exists.

        The ActionPlan system handles pathfinding lazily during execution,
        not at plan creation time. Path failures are handled during plan
        execution, not at intent creation time.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(8, 5),  # 3 tiles away
        )
        controller.enter_combat_mode()

        # Select Push
        push_action = ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={},
        )
        controller.combat_mode.select_action(push_action)

        intent = controller.combat_mode._create_intent_for_target(npc)

        # Plan is created regardless of path availability
        assert intent is None
        assert player.active_plan is not None
        assert player.active_plan.plan.name == "Push"


# --- Integration Tests: Discovery <-> Combat Mode Handler Consistency ---


class TestAllDiscoverableActionsHaveHandlers:
    """Integration tests ensuring every discoverable action works in combat mode.

    These tests catch the case where a new action is added to discovery but
    the combat mode handler is not updated to create the corresponding intent.
    """

    def test_all_stunt_actions_have_combat_mode_handlers(self) -> None:
        """Every discoverable STUNT action must have a handler in combat mode.

        This test gets all stunt actions from the action discovery system and
        verifies each one can be successfully executed via combat mode without
        raising an "Unhandled action type" error.

        If this test fails after adding a new stunt, you need to update
        CombatMode._create_intent_for_target() to handle the new intent class.

        All stunts now use the ActionPlan system, returning None and setting
        an active_plan on the player.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent so all melee actions work
        )
        controller.enter_combat_mode()

        # Get all available actions and filter to STUNT category
        actions = controller.combat_mode.get_available_combat_actions()
        stunt_actions = [a for a in actions if a.category == ActionCategory.STUNT]

        assert len(stunt_actions) > 0, "Expected at least one stunt action"

        # Map of action_class to expected plan name
        expected_plan_names = {
            PushIntent: "Push",
            TripIntent: "Trip",
            KickIntent: "Kick",
            PunchIntent: "Punch",
        }

        # Try each stunt action - if any raises ValueError, the handler is missing
        for stunt in stunt_actions:
            # Clear any existing plan before testing this action
            player.active_plan = None

            controller.combat_mode.select_action(stunt)

            # This should NOT raise "Unhandled action type" ValueError
            try:
                intent = controller.combat_mode._create_intent_for_target(npc)
            except ValueError as e:
                if "Unhandled action type" in str(e):
                    raise AssertionError(
                        f"Stunt action '{stunt.name}' (class={stunt.action_class}) "
                        f"is discoverable but has no handler in CombatMode. "
                        f"Add a handler in _create_intent_for_target()."
                    ) from e
                raise

            # All stunts now use the ActionPlan system
            assert intent is None, (
                f"Stunt '{stunt.name}' should use plan system, not return intent"
            )
            assert player.active_plan is not None, (
                f"Stunt '{stunt.name}' should set active_plan on player"
            )

            expected_name = expected_plan_names.get(stunt.action_class)
            if expected_name:
                actual_name = player.active_plan.plan.name
                assert actual_name == expected_name, (
                    f"Stunt '{stunt.name}' created plan '{actual_name}' "
                    f"but expected '{expected_name}'"
                )

    def test_all_combat_actions_have_handlers(self) -> None:
        """Every discoverable COMBAT action must work in combat mode.

        Similar to the stunt test, but for attack actions.
        Melee attacks use the ActionPlan system (return None, set active_plan).
        Ranged attacks return an AttackIntent directly.
        """
        reset_event_bus_for_testing()
        controller, player, npc = _make_combat_test_world(
            player_pos=(5, 5),
            enemy_pos=(6, 5),  # Adjacent
        )
        controller.enter_combat_mode()

        actions = controller.combat_mode.get_available_combat_actions()
        combat_actions = [a for a in actions if a.category == ActionCategory.COMBAT]

        assert len(combat_actions) > 0, "Expected at least one combat action"

        for combat_action in combat_actions:
            # Clear any existing plan before testing this action
            player.active_plan = None

            controller.combat_mode.select_action(combat_action)

            try:
                intent = controller.combat_mode._create_intent_for_target(npc)
            except ValueError as e:
                if "Unhandled action type" in str(e):
                    raise AssertionError(
                        f"Combat action '{combat_action.name}' is discoverable "
                        f"but has no handler in CombatMode."
                    ) from e
                raise

            attack_mode = combat_action.static_params.get("attack_mode")
            if attack_mode == "ranged":
                # Ranged attacks return intent directly
                assert intent is not None, (
                    f"Ranged action '{combat_action.name}' returned None for target"
                )
                assert isinstance(intent, AttackIntent)
            else:
                # Melee attacks use the plan system
                assert intent is None, (
                    f"Melee action '{combat_action.name}' should use plan system"
                )
                assert player.active_plan is not None, (
                    f"Melee action '{combat_action.name}' should set active_plan"
                )
                assert player.active_plan.plan.name == "Melee Attack"
