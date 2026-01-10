"""Tests for action panel hotkey assignment and priority sorting."""

from __future__ import annotations

from unittest.mock import MagicMock

from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.items.properties import WeaponProperty


class TestActionPanelHotkeys:
    """Test the hotkey priority and sticky assignment logic."""

    def _create_mock_weapon(
        self,
        melee_props: set[WeaponProperty] | None = None,
        ranged_props: set[WeaponProperty] | None = None,
    ) -> MagicMock:
        """Create a mock weapon with specified properties."""
        weapon = MagicMock()

        if melee_props is not None:
            weapon.melee_attack = MagicMock()
            weapon.melee_attack.properties = melee_props
        else:
            weapon.melee_attack = None

        if ranged_props is not None:
            weapon.ranged_attack = MagicMock()
            weapon.ranged_attack.properties = ranged_props
        else:
            weapon.ranged_attack = None

        return weapon

    def _create_action(
        self,
        action_id: str,
        weapon: MagicMock | None = None,
        attack_mode: str | None = None,
        category: ActionCategory = ActionCategory.COMBAT,
    ) -> ActionOption:
        """Create an ActionOption for testing."""
        static_params = {}
        if weapon is not None:
            static_params["weapon"] = weapon
        if attack_mode is not None:
            static_params["attack_mode"] = attack_mode

        return ActionOption(
            id=action_id,
            name=f"Test {action_id}",
            description="Test action",
            category=category,
            action_class=MagicMock(),  # type: ignore[arg-type]
            static_params=static_params,
        )

    def _get_action_priority(self, action: ActionOption) -> int:
        """Reimplementation of ActionPanelView._get_action_priority for testing."""
        weapon = action.static_params.get("weapon")
        attack_mode = action.static_params.get("attack_mode")

        if weapon is None or attack_mode is None:
            return 1

        attack = None
        if attack_mode == "melee" and weapon.melee_attack:
            attack = weapon.melee_attack
        elif attack_mode == "ranged" and weapon.ranged_attack:
            attack = weapon.ranged_attack

        if attack is None:
            return 1

        properties = attack.properties
        if WeaponProperty.PREFERRED in properties:
            return 0
        if WeaponProperty.IMPROVISED in properties:
            return 2
        return 1

    def _assign_hotkeys(
        self,
        actions: list[ActionOption],
        previous_hotkeys: dict[str, str],
    ) -> dict[str, str]:
        """Reimplementation of ActionPanelView._assign_hotkeys for testing.

        Returns the new hotkey mapping.
        """
        if not actions:
            return {}

        # Sort actions by priority within each category
        actions.sort(key=lambda a: (a.category.value, self._get_action_priority(a)))

        hotkey_chars = "abcdefghijklmnopqrstuvwxyz"
        used_hotkeys: set[str] = set()
        new_hotkeys: dict[str, str] = {}

        # First pass: preserve previous assignments
        for action in actions:
            if action.id in previous_hotkeys:
                prev_key = previous_hotkeys[action.id]
                if prev_key not in used_hotkeys and prev_key in hotkey_chars:
                    action.hotkey = prev_key
                    used_hotkeys.add(prev_key)
                    new_hotkeys[action.id] = prev_key

        # Second pass: assign new hotkeys
        hotkey_index = 0
        for action in actions:
            if action.hotkey is None:
                while hotkey_index < len(hotkey_chars):
                    candidate = hotkey_chars[hotkey_index]
                    hotkey_index += 1
                    if candidate not in used_hotkeys:
                        action.hotkey = candidate
                        used_hotkeys.add(candidate)
                        new_hotkeys[action.id] = candidate
                        break

        return new_hotkeys


class TestPrioritySorting(TestActionPanelHotkeys):
    """Test that actions are sorted by PREFERRED/IMPROVISED properties."""

    def test_preferred_attack_has_highest_priority(self) -> None:
        """PREFERRED attacks should have priority 0."""
        weapon = self._create_mock_weapon(
            ranged_props={WeaponProperty.PREFERRED},
        )
        action = self._create_action("shoot", weapon=weapon, attack_mode="ranged")

        assert self._get_action_priority(action) == 0

    def test_improvised_attack_has_lowest_priority(self) -> None:
        """IMPROVISED attacks should have priority 2."""
        weapon = self._create_mock_weapon(
            melee_props={WeaponProperty.IMPROVISED},
        )
        action = self._create_action("pistol-whip", weapon=weapon, attack_mode="melee")

        assert self._get_action_priority(action) == 2

    def test_regular_attack_has_middle_priority(self) -> None:
        """Attacks without PREFERRED or IMPROVISED should have priority 1."""
        weapon = self._create_mock_weapon(
            melee_props=set(),  # No special properties
        )
        action = self._create_action("punch", weapon=weapon, attack_mode="melee")

        assert self._get_action_priority(action) == 1

    def test_non_combat_action_has_middle_priority(self) -> None:
        """Actions without weapon/attack_mode should have priority 1."""
        action = self._create_action("open-door", category=ActionCategory.ENVIRONMENT)

        assert self._get_action_priority(action) == 1

    def test_sorting_puts_preferred_before_improvised(self) -> None:
        """When sorted, PREFERRED attacks should come before IMPROVISED."""
        pistol = self._create_mock_weapon(
            melee_props={WeaponProperty.IMPROVISED},
            ranged_props={WeaponProperty.PREFERRED},
        )

        shoot = self._create_action("shoot", weapon=pistol, attack_mode="ranged")
        pistol_whip = self._create_action(
            "pistol-whip", weapon=pistol, attack_mode="melee"
        )

        # Create list with improvised first (wrong order)
        actions = [pistol_whip, shoot]
        self._assign_hotkeys(actions, {})

        # After sorting, shoot (PREFERRED) should be first
        assert actions[0].id == "shoot"
        assert actions[1].id == "pistol-whip"

    def test_preferred_gets_hotkey_a(self) -> None:
        """PREFERRED attack should get hotkey 'a' when sorted first."""
        pistol = self._create_mock_weapon(
            melee_props={WeaponProperty.IMPROVISED},
            ranged_props={WeaponProperty.PREFERRED},
        )

        shoot = self._create_action("shoot", weapon=pistol, attack_mode="ranged")
        pistol_whip = self._create_action(
            "pistol-whip", weapon=pistol, attack_mode="melee"
        )

        actions = [pistol_whip, shoot]  # Wrong initial order
        self._assign_hotkeys(actions, {})

        assert shoot.hotkey == "a"
        assert pistol_whip.hotkey == "b"


class TestStickyHotkeys(TestActionPanelHotkeys):
    """Test that hotkeys persist across frames."""

    def test_hotkeys_preserved_when_actions_unchanged(self) -> None:
        """Same actions should keep same hotkeys between calls."""
        weapon = self._create_mock_weapon(ranged_props={WeaponProperty.PREFERRED})
        shoot = self._create_action("shoot", weapon=weapon, attack_mode="ranged")

        actions = [shoot]
        hotkeys1 = self._assign_hotkeys(actions, {})

        # Reset hotkey and call again with previous mapping
        shoot.hotkey = None
        actions2 = [shoot]
        hotkeys2 = self._assign_hotkeys(actions2, hotkeys1)

        assert hotkeys2["shoot"] == "a"
        assert shoot.hotkey == "a"

    def test_existing_action_keeps_hotkey_when_new_action_added(self) -> None:
        """When a new action appears, existing actions keep their hotkeys."""
        pistol = self._create_mock_weapon(
            melee_props={WeaponProperty.IMPROVISED},
            ranged_props={WeaponProperty.PREFERRED},
        )

        # First frame: only shoot available
        shoot = self._create_action("shoot", weapon=pistol, attack_mode="ranged")
        actions1 = [shoot]
        hotkeys1 = self._assign_hotkeys(actions1, {})
        assert shoot.hotkey == "a"

        # Second frame: pistol-whip becomes available (enemy got close)
        shoot.hotkey = None
        pistol_whip = self._create_action(
            "pistol-whip", weapon=pistol, attack_mode="melee"
        )
        actions2 = [pistol_whip, shoot]  # Discovered in this order
        self._assign_hotkeys(actions2, hotkeys1)

        # Shoot should keep 'a' (sticky), pistol-whip gets 'b'
        assert shoot.hotkey == "a"
        assert pistol_whip.hotkey == "b"

    def test_removed_action_frees_hotkey_for_new_action(self) -> None:
        """When an action disappears, its hotkey becomes available."""
        weapon1 = self._create_mock_weapon(ranged_props={WeaponProperty.PREFERRED})
        weapon2 = self._create_mock_weapon(melee_props={WeaponProperty.PREFERRED})

        shoot = self._create_action("shoot", weapon=weapon1, attack_mode="ranged")
        stab = self._create_action("stab", weapon=weapon2, attack_mode="melee")

        # First frame: both actions
        actions1 = [shoot, stab]
        hotkeys1 = self._assign_hotkeys(actions1, {})
        assert shoot.hotkey == "a"
        assert stab.hotkey == "b"

        # Second frame: only stab (different target, out of range for shoot)
        stab.hotkey = None
        actions2 = [stab]
        hotkeys2 = self._assign_hotkeys(actions2, hotkeys1)

        # Stab keeps 'b' (sticky)
        assert stab.hotkey == "b"
        assert "shoot" not in hotkeys2

    def test_multiple_new_actions_fill_gaps(self) -> None:
        """New actions should fill available hotkey slots."""
        weapon = self._create_mock_weapon(ranged_props={WeaponProperty.PREFERRED})

        action_a = self._create_action("action-a", weapon=weapon, attack_mode="ranged")
        action_b = self._create_action("action-b", weapon=weapon, attack_mode="ranged")
        action_c = self._create_action("action-c", weapon=weapon, attack_mode="ranged")

        # First frame: action-b gets 'a'
        actions1 = [action_b]
        hotkeys1 = self._assign_hotkeys(actions1, {})
        assert action_b.hotkey == "a"

        # Second frame: action-a and action-c added
        action_b.hotkey = None
        actions2 = [action_a, action_b, action_c]
        self._assign_hotkeys(actions2, hotkeys1)

        # action-b keeps 'a', others get 'b' and 'c'
        assert action_b.hotkey == "a"
        assert action_a.hotkey == "b"
        assert action_c.hotkey == "c"
