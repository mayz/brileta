from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from catley import colors
from catley.game import range_system
from catley.game.actions import AttackAction, ReloadAction
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.ui.menu_core import Menu, MenuOption

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class QuickActionBar(Menu):
    """
    Small action bar that appears when selecting an actor,
    showing most common actions."""

    def __init__(self, controller: Controller, target_actor: Character) -> None:
        super().__init__(f"{target_actor.name}", controller, width=35, max_height=10)
        self.target_actor = target_actor
        self.distance = range_system.calculate_distance(
            controller.gw.player.x,
            controller.gw.player.y,
            target_actor.x,
            target_actor.y,
        )

    def populate_options(self) -> None:
        """Populate quick actions - only the most common 2-3 actions."""
        player = self.controller.gw.player

        # Show basic stats first (no key, just info)
        if self.target_actor.health:
            hp_text = (
                f"HP: {self.target_actor.health.hp}/{self.target_actor.health.max_hp}"
            )
            if self.target_actor.health.ap > 0:
                hp_text += f" | AP: {self.target_actor.health.ap}"
            self.add_option(
                MenuOption(key=None, text=hp_text, enabled=False, color=colors.CYAN)
            )

        # Separator
        self.add_option(
            MenuOption(key=None, text="─" * 25, enabled=False, color=colors.GREY)
        )

        attack_action = AttackAction(self.controller, player, self.target_actor)
        can_attack, reason = attack_action.can_execute()

        if can_attack:
            prob = attack_action.get_success_probability()
            prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

            # Determine weapon name for display
            primary_weapon = player.inventory.attack_slots[0]
            if primary_weapon:
                if self.distance == 1 or not primary_weapon.ranged_attack:
                    weapon_name = (
                        primary_weapon.name
                        if primary_weapon and primary_weapon.melee_attack
                        else "fists"
                    )
                    action_text = f"Attack with {weapon_name}{prob_text}"
                else:
                    ammo_display = (
                        f"[{primary_weapon.ranged_attack.current_ammo}/"
                        f"{primary_weapon.ranged_attack.max_ammo}]"
                    )
                    action_text = (
                        f"Shoot {primary_weapon.name} {ammo_display}{prob_text}"
                    )
            else:
                action_text = f"Attack (unarmed){prob_text}"

            action_func = functools.partial(self._perform_quick_attack)
            self.add_option(
                MenuOption(
                    key="1", text=action_text, action=action_func, color=colors.WHITE
                )
            )
        else:
            self.add_option(
                MenuOption(
                    key=None,
                    text=f"Can't attack: {reason}",
                    enabled=False,
                    color=colors.RED,
                )
            )

        # Quick reload if needed and possible
        primary_weapon = player.inventory.attack_slots[0]
        if (
            primary_weapon
            and primary_weapon.ranged_attack
            and primary_weapon.ranged_attack.current_ammo
            < primary_weapon.ranged_attack.max_ammo
        ):
            # Check if player has compatible ammo
            has_ammo = any(
                isinstance(ammo_item, Item)
                and ammo_item.ammo
                and ammo_item.ammo.ammo_type == primary_weapon.ranged_attack.ammo_type
                for ammo_item in player.inventory
            )

            if has_ammo:
                action_func = functools.partial(self._perform_quick_reload)
                self.add_option(
                    MenuOption(
                        key="r",
                        text=f"Reload {primary_weapon.name}",
                        action=action_func,
                        color=colors.GREEN,
                    )
                )

        # Always show "More options" to open full target menu
        action_func = functools.partial(self._open_full_menu)
        self.add_option(
            MenuOption(
                key="t",
                text="More options... (T)",
                action=action_func,
                color=colors.YELLOW,
            )
        )

    def _perform_quick_attack(self) -> None:
        """Perform attack with primary weapon."""
        player = self.controller.gw.player
        attack_action = AttackAction(self.controller, player, self.target_actor)
        self.controller.queue_action(attack_action)

    def _perform_quick_reload(self) -> None:
        """Reload primary weapon."""
        player = self.controller.gw.player
        primary_weapon = player.inventory.attack_slots[0]

        if primary_weapon and primary_weapon.ranged_attack:
            reload_action = ReloadAction(self.controller, player, primary_weapon)
            self.controller.queue_action(reload_action)

    def _open_full_menu(self) -> None:
        """Open the full target menu."""
        # Hide the quick bar first
        self.hide()
        # Import here to avoid circular imports
        from catley.ui.ui_commands import OpenTargetMenuUICommand

        command = OpenTargetMenuUICommand(
            self.controller, target_actor=self.target_actor
        )
        command.execute()


class TargetMenu(Menu):
    """Menu for selecting actions to perform on a target actor or location."""

    def __init__(
        self,
        controller: Controller,
        target_actor: Character | None = None,
        target_location: tuple[int, int] | None = None,
    ) -> None:
        if target_actor:
            title = f"Target: {target_actor.name}"
            self.target_actor = target_actor
            self.target_location = (target_actor.x, target_actor.y)
            self.distance = range_system.calculate_distance(
                controller.gw.player.x,
                controller.gw.player.y,
                target_actor.x,
                target_actor.y,
            )
        elif target_location:
            title = f"Target: Location ({target_location[0]}, {target_location[1]})"
            self.target_actor = None
            self.target_location = target_location
            self.distance = range_system.calculate_distance(
                controller.gw.player.x,
                controller.gw.player.y,
                target_location[0],
                target_location[1],
            )
        else:
            raise ValueError("Either target_actor or target_location must be provided")

        super().__init__(title, controller, width=50)

    def populate_options(self) -> None:
        """Populate targeting options based on the target and player capabilities."""
        player = self.controller.gw.player

        # Attack actions - test each equipped weapon
        if self.target_actor:
            options_added = False

            for item, slot_index in player.inventory.get_equipped_items():
                if item.melee_attack or item.ranged_attack:
                    # Temporarily equip this weapon to test it
                    old_weapon = player.inventory.attack_slots[0]
                    player.inventory.equip_to_slot(item, 0)  # Test in primary slot

                    attack_action = AttackAction(
                        self.controller, player, self.target_actor
                    )
                    can_attack, reason = attack_action.can_execute()
                    reason = reason.upper() if reason else None

                    # Restore original weapon
                    if old_weapon:
                        player.inventory.equip_to_slot(old_weapon, 0)
                    else:
                        player.inventory.unequip_slot(0)

                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    key = chr(ord("a") + len(self.options))

                    if can_attack:
                        prob = attack_action.get_success_probability()
                        prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

                        # Format text based on attack type
                        if self.distance == 1 and item.melee_attack:
                            action_text = (
                                f"Attack with {item.name} ({slot_name}){prob_text}"
                            )
                            action_func = functools.partial(
                                self._perform_melee_attack, item, self.target_actor
                            )
                        elif item.ranged_attack:
                            ammo_display = (
                                f"[{item.ranged_attack.current_ammo}/"
                                f"{item.ranged_attack.max_ammo}]"
                            )
                            action_text = (
                                f"Shoot {item.name} ({slot_name}) "
                                f"{ammo_display}{prob_text}"
                            )
                            action_func = functools.partial(
                                self._perform_ranged_attack, item, self.target_actor
                            )

                        self.add_option(
                            MenuOption(
                                key=key,
                                text=action_text,
                                action=action_func,
                                color=colors.WHITE,
                            )
                        )
                        options_added = True
                    else:
                        # Show disabled option with reason
                        if item.ranged_attack:
                            disabled_text = (
                                f"Shoot {item.name} ({slot_name}) [{reason}]"
                            )
                        else:
                            disabled_text = (
                                f"Attack with {item.name} ({slot_name}) [{reason}]"
                            )
                        self.add_option(
                            MenuOption(
                                key=None,
                                text=disabled_text,
                                enabled=False,
                                color=colors.RED,
                            )
                        )

            # Add unarmed option if no valid weapons
            if not options_added and self.distance == 1:
                attack_action = AttackAction(self.controller, player, self.target_actor)
                can_attack, reason = attack_action.can_execute()

                if can_attack:
                    prob = attack_action.get_success_probability()
                    prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""
                    action_func = functools.partial(
                        self._perform_melee_attack,
                        FISTS_TYPE.create(),
                        self.target_actor,
                    )
                    self.add_option(
                        MenuOption(
                            key="1",
                            text=f"Attack (unarmed){prob_text}",
                            action=action_func,
                            color=colors.WHITE,
                        )
                    )

        # Reload actions
        self._add_reload_options()

        # If no actions available, show a message
        if not self.options:
            self.add_option(
                MenuOption(
                    key=None,
                    text="No actions available",
                    enabled=False,
                    color=colors.GREY,
                )
            )

    def _add_reload_options(self) -> None:
        """Add reload actions for partially loaded weapons."""
        player = self.controller.gw.player

        for item, slot_index in player.inventory.get_equipped_items():
            ranged_attack = item.ranged_attack
            if ranged_attack and ranged_attack.current_ammo < ranged_attack.max_ammo:
                # Check if player has compatible ammo
                has_ammo = any(
                    isinstance(ammo_item, Item)
                    and ammo_item.ammo
                    and ammo_item.ammo.ammo_type == ranged_attack.ammo_type
                    for ammo_item in player.inventory
                )
                if has_ammo:
                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    reload_key = chr(
                        ord("a") + len(self.options)
                    )  # Continue sequential letters
                    action_func = functools.partial(self._perform_reload, item)
                    self.add_option(
                        MenuOption(
                            key=reload_key,
                            text=f"Reload {item.name} ({slot_name})",
                            action=action_func,
                            color=colors.GREEN,
                        )
                    )

    def _perform_melee_attack(self, weapon_item, target_actor) -> None:
        """Perform a melee attack with the specified weapon."""
        # Temporarily equip the weapon if it's not the player's fists
        player = self.controller.gw.player

        if weapon_item.name != "Fists":
            # Find which slot this weapon is in and make sure it's equipped
            for item, slot_index in player.inventory.get_equipped_items():
                if item == weapon_item:
                    break

        # Perform the attack using existing AttackAction
        attack_action = AttackAction(self.controller, player, target_actor)
        self.controller.queue_action(attack_action)

    def _perform_ranged_attack(self, weapon_item, target_actor) -> None:
        """Perform a ranged attack with the specified weapon."""
        # For now, use the existing AttackAction which will automatically
        # choose ranged attack based on distance
        player = self.controller.gw.player
        attack_action = AttackAction(self.controller, player, target_actor, weapon_item)
        self.controller.queue_action(attack_action)

    def _perform_reload(self, weapon_item) -> None:
        """Reload the specified weapon."""
        player = self.controller.gw.player
        reload_action = ReloadAction(self.controller, player, weapon_item)
        self.controller.queue_action(reload_action)
