from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

from catley import colors
from catley.game import range_system
from catley.game.actions import AttackAction
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.ui.menu_core import Menu, MenuOption
from catley.util import dice

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor
    from catley.game.components import StatsComponent


class QuickActionBar(Menu):
    """
    Small action bar that appears when selecting an actor,
    showing most common actions."""

    def __init__(self, controller: Controller, target_actor: Actor) -> None:
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

        # Quick attack action
        if self.distance == 1:
            # Melee range - show primary melee attack
            primary_weapon = player.inventory.equipped_weapon
            if primary_weapon and primary_weapon.melee_attack:
                weapon_name = primary_weapon.name
            else:
                weapon_name = "fists"
                primary_weapon = None

            # Calculate success probability
            prob = _calculate_attack_success_probability(
                primary_weapon,
                player,
                self.target_actor,
                self.distance,
                self.controller,
            )
            prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

            action_func = functools.partial(self._perform_quick_attack)
            self.add_option(
                MenuOption(
                    key="1",
                    text=f"Attack with {weapon_name}{prob_text}",
                    action=action_func,
                    color=colors.WHITE,
                )
            )
        else:
            # Ranged - show primary ranged attack if possible
            primary_weapon = player.inventory.equipped_weapon
            if (
                primary_weapon
                and primary_weapon.ranged_attack
                and primary_weapon.ranged_attack.current_ammo > 0
            ):
                # Check if we can actually shoot
                range_category = range_system.get_range_category(
                    self.distance, primary_weapon
                )
                range_mods = range_system.get_range_modifier(
                    primary_weapon, range_category
                )
                has_los = range_system.has_line_of_sight(
                    self.controller.gw.game_map,
                    player.x,
                    player.y,
                    self.target_actor.x,
                    self.target_actor.y,
                )

                if range_mods is not None and has_los:
                    ammo_display = (
                        f"[{primary_weapon.ranged_attack.current_ammo}/"
                        f"{primary_weapon.ranged_attack.max_ammo}]"
                    )

                    # Calculate success probability
                    prob = _calculate_attack_success_probability(
                        primary_weapon,
                        player,
                        self.target_actor,
                        self.distance,
                        self.controller,
                    )
                    prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

                    action_func = functools.partial(self._perform_quick_attack)
                    self.add_option(
                        MenuOption(
                            key="1",
                            text=(
                                f"Shoot {primary_weapon.name} {ammo_display}{prob_text}"
                            ),
                            action=action_func,
                            color=colors.WHITE,
                        )
                    )
                else:
                    # Can't shoot - show why
                    reason = "no line of sight" if not has_los else "out of range"
                    self.add_option(
                        MenuOption(
                            key=None,
                            text=f"Can't shoot: {reason}",
                            enabled=False,
                            color=colors.RED,
                        )
                    )

        # Quick reload if needed and possible
        primary_weapon = player.inventory.equipped_weapon
        if (
            primary_weapon
            and primary_weapon.ranged_attack
            and primary_weapon.ranged_attack.current_ammo
            < primary_weapon.ranged_attack.max_ammo
        ):
            # Check if player has compatible ammo
            has_ammo = any(
                ammo_item.ammo
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
        primary_weapon = player.inventory.equipped_weapon

        if primary_weapon and primary_weapon.ranged_attack:
            # Find compatible ammo
            ammo_item = None
            for item in player.inventory:
                if (
                    item.ammo
                    and item.ammo.ammo_type == primary_weapon.ranged_attack.ammo_type
                ):
                    ammo_item = item
                    break

            if ammo_item:
                player.inventory.remove_from_inventory(ammo_item)
                primary_weapon.ranged_attack.current_ammo = (
                    primary_weapon.ranged_attack.max_ammo
                )
                self.controller.message_log.add_message(
                    f"Reloaded {primary_weapon.name}", colors.GREEN
                )
                # Refresh the quick bar to show updated ammo
                self.populate_options()

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
        target_actor: Actor | None = None,
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
        target_x, target_y = self.target_location

        # Check line of sight first for ranged options
        has_los = range_system.has_line_of_sight(
            self.controller.gw.game_map, player.x, player.y, target_x, target_y
        )

        # Melee actions (adjacent only)
        if self.distance == 1 and self.target_actor:  # Can only melee attack actors
            melee_added = False
            for item, slot_index in player.inventory.get_equipped_items():
                if item.melee_attack:
                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    key = chr(ord("a") + len(self.options))  # Sequential letters

                    # Calculate success probability
                    prob = _calculate_attack_success_probability(
                        item, player, self.target_actor, self.distance, self.controller
                    )
                    prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

                    action_func = functools.partial(
                        self._perform_melee_attack, item, self.target_actor
                    )
                    self.add_option(
                        MenuOption(
                            key=key,
                            text=f"Attack with {item.name} ({slot_name}){prob_text}",
                            action=action_func,
                            color=colors.WHITE,
                        )
                    )
                    melee_added = True

            # Always allow unarmed attack if no melee weapons equipped
            if not melee_added:
                fists = FISTS_TYPE.create()

                # Calculate success probability for unarmed
                prob = _calculate_attack_success_probability(
                    fists, player, self.target_actor, self.distance, self.controller
                )
                prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

                action_func = functools.partial(
                    self._perform_melee_attack, fists, self.target_actor
                )
                self.add_option(
                    MenuOption(
                        key="1",
                        text=f"Attack (unarmed){prob_text}",
                        action=action_func,
                        color=colors.WHITE,
                    )
                )

        # Ranged actions (if line of sight)
        if has_los and self.target_actor:  # Can only shoot at actors for now
            for item, slot_index in player.inventory.get_equipped_items():
                if item.ranged_attack:
                    range_category = range_system.get_range_category(
                        self.distance, item
                    )
                    range_mods = range_system.get_range_modifier(item, range_category)

                    if range_mods is None:  # Out of range
                        continue

                    slot_name = player.inventory.get_slot_display_name(slot_index)
                    key = chr(
                        ord("a") + len(self.options)
                    )  # Sequential letters: a, b, c, d...
                    if item.ranged_attack.current_ammo > 0:
                        ammo_display = (
                            f"[{item.ranged_attack.current_ammo}/"
                            f"{item.ranged_attack.max_ammo}]"
                        )

                        # Calculate success probability
                        prob = _calculate_attack_success_probability(
                            item,
                            player,
                            self.target_actor,
                            self.distance,
                            self.controller,
                        )
                        prob_text = f" ({prob * 100:.0f}%)" if prob is not None else ""

                        action_func = functools.partial(
                            self._perform_ranged_attack, item, self.target_actor
                        )
                        self.add_option(
                            MenuOption(
                                key=key,
                                text=(
                                    f"Shoot {item.name} ({slot_name}) "
                                    f"{ammo_display}{prob_text}"
                                ),
                                action=action_func,
                                color=colors.WHITE,
                            )
                        )
                    else:
                        # Show out of ammo option but disabled
                        self.add_option(
                            MenuOption(
                                key=None,
                                text=f"Shoot {item.name} ({slot_name}) [OUT OF AMMO]",
                                enabled=False,
                                color=colors.RED,
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
                    action_func = functools.partial(
                        self._perform_reload, item, slot_index
                    )
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
        attack_action = AttackAction(self.controller, player, target_actor)
        self.controller.queue_action(attack_action)

    def _perform_reload(self, weapon_item, slot_index) -> None:
        """Reload the specified weapon."""
        player = self.controller.gw.player
        ranged_attack = weapon_item.ranged_attack

        # Find compatible ammo in inventory
        ammo_item = None
        for item in player.inventory:
            if item.ammo and item.ammo.ammo_type == ranged_attack.ammo_type:
                ammo_item = item
                break

        if ammo_item:
            # Remove ammo from inventory and reload weapon
            player.inventory.remove_from_inventory(ammo_item)
            ranged_attack.current_ammo = ranged_attack.max_ammo
            slot_name = player.inventory.get_slot_display_name(slot_index)
            self.controller.message_log.add_message(
                f"Reloaded {weapon_item.name} ({slot_name})", colors.GREEN
            )
        else:
            self.controller.message_log.add_message(
                f"No {ranged_attack.ammo_type} ammo available!", colors.RED
            )


def _calculate_attack_success_probability(
    weapon: Item | None,
    attacker: Actor,
    defender: Actor,
    distance: int,
    controller: Controller,
) -> float | None:
    """Calculate the probability of a successful attack."""
    # Use fists if no weapon
    if weapon is None:
        weapon = FISTS_TYPE.create()

    # Check if actors have required stats
    if not attacker.stats or not defender.stats:
        return None

    attacker_stats = cast("StatsComponent", attacker.stats)
    defender_stats = cast("StatsComponent", defender.stats)

    # Determine which attack to use (same logic as AttackAction)
    attack = None
    range_modifiers = {}

    if distance == 1 and weapon.melee_attack:
        attack = weapon.melee_attack
    elif weapon.ranged_attack:
        attack = weapon.ranged_attack
        # Get range modifiers for ranged attacks
        if distance > 1:
            if not range_system.has_line_of_sight(
                controller.gw.game_map, attacker.x, attacker.y, defender.x, defender.y
            ):
                return None  # No line of sight

            range_category = range_system.get_range_category(distance, weapon)
            range_mods = range_system.get_range_modifier(weapon, range_category)
            if range_mods is None:
                return None  # Out of range
            range_modifiers = range_mods
    elif weapon.melee_attack:
        attack = weapon.melee_attack
    else:
        return None  # No attack possible

    if not attack:
        return None

    # Get ability scores
    attacker_ability_score = getattr(attacker_stats, attack.stat_name)
    defender_ability_score = defender_stats.agility

    # Calculate opposed check probability
    roll_to_exceed = defender_ability_score + 10
    return dice.calculate_check_roll_success_probability(
        attacker_ability_score,
        roll_to_exceed,
        has_advantage=range_modifiers.get("has_advantage", False),
        has_disadvantage=range_modifiers.get("has_disadvantage", False),
    )
