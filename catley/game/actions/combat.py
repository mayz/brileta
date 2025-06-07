"""
Combat actions for direct actor-vs-actor confrontations.

Handles attack resolution, damage calculation, weapon reloading, and all
combat-related mechanics including hit/miss determination and effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game import range_system
from catley.game.actions.base import GameAction
from catley.game.actors import Character, Disposition
from catley.game.ai import DispositionBasedAI
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.items.properties import WeaponProperty
from catley.game.resolution.base import ResolutionResult
from catley.game.resolution.d20_system import D20Resolver

if TYPE_CHECKING:
    from catley.controller import Controller


class AttackAction(GameAction):
    """Action for one `Actor` attacking another in combat."""

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
        weapon: Item | None = None,
    ) -> None:
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender
        self.weapon = weapon

    def execute(self) -> None:
        # 1. Determine what attack method to use
        attack, weapon = self._determine_attack_method()
        if not attack:
            return

        # 2. Validate the attack can be performed
        range_modifiers = self._validate_attack(attack, weapon)
        if range_modifiers is None:
            return  # Validation failed, error messages already logged

        # 3. Perform the attack roll and immediate effects
        attack_result = self._execute_attack_roll(attack, weapon, range_modifiers)

        # 4. Resolve the outcome based on hit/miss
        damage = self._resolve_attack_outcome(attack_result, attack, weapon)

        # 5. Handle post-attack effects (screen shake, messages, AI)
        self._handle_post_attack_effects(attack_result, attack, weapon, damage)

    def _determine_attack_method(self) -> tuple[Attack | None, Item]:
        """Determine which attack method and weapon to use
        based on distance and capabilities."""
        weapon = (
            self.weapon
            or self.attacker.inventory.get_active_weapon()
            or FISTS_TYPE.create()
        )

        distance = range_system.calculate_distance(
            self.attacker.x, self.attacker.y, self.defender.x, self.defender.y
        )

        ranged_attack = weapon.ranged_attack

        #######################################################################
        # FIXME: In Phase 6, REPLACE THIS DISTANCE-BASED HEURISTIC WITH
        #        PROPER ATTACK MODE SELECTION!
        #
        # Current limitation: This automatically chooses attack mode based on distance,
        # which works for most cases but prevents intentional choices like:
        # - Pistol-whipping when you could shoot (stealth, ammo conservation)
        # - Throwing a knife instead of stabbing (ranged vs melee for same weapon)
        # - Choosing between single-target ranged vs area effect for dual-mode weapons
        #
        # Phase 6 solution:
        # - Pass attack_type parameter to AttackAction constructor ("melee", "ranged")
        # - UI provides attack mode selection when multiple modes available
        # - Move this logic to UI layer, not combat execution layer
        #######################################################################

        # Determine which attack to use based on distance and weapon capabilities
        if distance == 1 and weapon.melee_attack:
            # Adjacent and has melee capability - use melee
            return weapon.melee_attack, weapon

        if ranged_attack:
            # Not adjacent or no melee - use ranged if available
            return ranged_attack, weapon

        if weapon.melee_attack:
            # Fallback to melee if no ranged (shouldn't happen with proper
            # distance checking)
            return weapon.melee_attack, weapon

        # No attack capabilities
        self.controller.message_log.add_message(
            f"{self.attacker.name} has no way to attack!", colors.RED
        )
        return None, weapon

    def _validate_attack(self, attack: Attack, weapon: Item) -> dict | None:
        """Validate the attack can be performed and return range modifiers."""
        distance = range_system.calculate_distance(
            self.attacker.x, self.attacker.y, self.defender.x, self.defender.y
        )

        range_modifiers = {}
        ranged_attack = weapon.ranged_attack

        # For ranged attacks, check ammo, check line of sight and apply range modifiers
        if ranged_attack is not None and attack == ranged_attack and distance > 1:
            if ranged_attack.current_ammo <= 0:
                self.controller.message_log.add_message(
                    f"{weapon.name} is out of ammo!", colors.RED
                )
                return None

            # Check line of sight
            if not range_system.has_line_of_sight(
                self.controller.gw.game_map,
                self.attacker.x,
                self.attacker.y,
                self.defender.x,
                self.defender.y,
            ):
                self.controller.message_log.add_message(
                    f"No clear shot to {self.defender.name}!", colors.RED
                )
                return None

            # Get range modifiers
            range_category = range_system.get_range_category(distance, weapon)
            range_modifiers = range_system.get_range_modifier(weapon, range_category)

            if range_modifiers is None:
                self.controller.message_log.add_message(
                    f"{self.defender.name} is out of range!", colors.RED
                )
                return None

        return range_modifiers

    def _execute_attack_roll(
        self, attack: Attack, weapon: Item, range_modifiers: dict
    ) -> ResolutionResult:
        """Perform the attack roll and handle immediate effects
        like ammo consumption."""
        # Get ability scores for the roll
        stat_name = attack.stat_name
        attacker_ability_score = getattr(self.attacker.stats, stat_name)
        defender_ability_score = self.defender.stats.agility

        resolver = D20Resolver(
            attacker_ability_score,
            defender_ability_score + 10,
            has_advantage=range_modifiers.get("has_advantage", False),
            has_disadvantage=range_modifiers.get("has_disadvantage", False),
        )
        attack_result = resolver.resolve(self.attacker, self.defender, weapon)

        # Consume ammo for ranged attacks
        ranged_attack = weapon.ranged_attack
        if attack == ranged_attack and ranged_attack is not None:
            if ranged_attack.current_ammo > 0:
                ranged_attack.current_ammo -= 1
            else:
                # This shouldn't happen if validation worked, but safety check
                self.controller.message_log.add_message(
                    f"No ammo left in {weapon.name}!", colors.RED
                )

        # Emit muzzle flash for ranged attacks
        if attack == weapon.ranged_attack:
            self._emit_muzzle_flash()

        return attack_result

    def _emit_muzzle_flash(self) -> None:
        """Emit muzzle flash particle effect."""
        direction_x = self.defender.x - self.attacker.x
        direction_y = self.defender.y - self.attacker.y

        self.frame_manager.create_effect(
            "muzzle_flash",
            x=self.attacker.x,
            y=self.attacker.y,
            direction_x=direction_x,
            direction_y=direction_y,
        )

    def _resolve_attack_outcome(
        self, attack_result: ResolutionResult, attack: Attack, weapon: Item
    ) -> int:
        """Resolve the attack outcome and return the damage dealt."""
        if attack_result.success:
            return self._handle_successful_hit(attack_result, attack, weapon)
        self._handle_attack_miss(attack_result, attack, weapon)
        return 0

    def _handle_successful_hit(
        self, attack_result: ResolutionResult, attack: Attack, weapon: Item
    ) -> int:
        """Handle a successful attack hit - damage, crits, particles, messages."""
        # Hit - roll damage
        damage_dice = attack.damage_dice
        damage = damage_dice.roll()

        if attack_result.is_critical_success:
            # "Crits favoring an attack deal an extra die of damage and break
            # the target's armor (lowering to 0 AP) before applying damage - or
            # cause an injury to an unarmored target."
            damage += damage_dice.roll()

            if self.defender.health.ap > 0:
                self.defender.health.ap = 0
                # FIXME: Break the defender's armor.
            else:
                # FIXME: Give the defender an injury.
                pass

        # Apply damage.
        self.defender.take_damage(damage)

        # Blood splatter only on successful hits
        if attack_result.success:
            self.frame_manager.create_effect(
                "blood_splatter",
                x=self.defender.x,
                y=self.defender.y,
                intensity=damage / 20.0,  # Scale intensity with damage
            )

        # Log hit messages
        self._log_hit_message(attack_result, weapon, damage)

        # Check if defender is defeated
        if not self.defender.health.is_alive():
            self.controller.message_log.add_message(
                f"{self.defender.name} has been killed!", colors.RED
            )

            # Update targeting if needed
            self.controller.notify_actor_death(self.defender)

        return damage

    def _handle_attack_miss(
        self, attack_result: ResolutionResult, attack: Attack, weapon: Item
    ) -> None:
        """Handle an attack miss - messages and weapon property effects."""
        # Miss
        if attack_result.is_critical_failure:
            # "Crits favoring defense cause the attacker's weapon to break,
            # and leave them confused or off-balance."
            # FIXME: Break the attacker's weapon.
            # FIXME: If the attacker is unarmed and attacking with fists or
            # kicking, etc., they pull a muscle and have disadvantage on their
            # next attack. (house rule)
            # FIXME: Give the attacker the condition "confused" or "off-balance".

            miss_message = (
                f"Critical miss! {self.attacker.name}'s attack on "
                f"{self.defender.name} fails."
            )
            miss_color = colors.ORANGE  # A warning color for critical miss
        else:
            miss_message = f"{self.attacker.name} misses {self.defender.name}."
            miss_color = colors.GREY  # Standard miss color
        self.controller.message_log.add_message(miss_message, miss_color)

        # Handle 'awkward' weapon property on miss
        if (
            weapon
            and weapon.melee_attack
            and WeaponProperty.AWKWARD in weapon.melee_attack.properties
        ):
            self.controller.message_log.add_message(
                f"{self.attacker.name} is off balance from the awkward swing "
                f"with {weapon.name}!",
                colors.LIGHT_BLUE,  # Informational color for status effects
            )
            # TODO: Implement off-balance effect (maybe skip next turn?)

    def _log_hit_message(
        self, attack_result: ResolutionResult, weapon: Item, damage: int
    ) -> None:
        """Log appropriate hit message based on critical status."""
        if attack_result.is_critical_success:
            hit_message = (
                f"Critical hit! {self.attacker.name} strikes {self.defender.name} "
                f"with {weapon.name} for {damage} damage."
            )
            hit_color = colors.YELLOW
        else:
            hit_message = (
                f"{self.attacker.name} hits {self.defender.name} "
                f"with {weapon.name} for {damage} damage."
            )
            hit_color = colors.WHITE  # Default color for a standard hit
        hp_message_part = (
            f" ({self.defender.name} has {self.defender.health.hp} HP left.)"
        )
        self.controller.message_log.add_message(
            hit_message + hp_message_part, hit_color
        )

    def _handle_post_attack_effects(
        self,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Handle post-attack effects like screen shake and AI disposition changes."""
        # Screen shake only when PLAYER gets hit
        if attack_result.success and self.defender == self.controller.gw.player:
            self._apply_screen_shake(attack_result, attack, weapon, damage)

        # Update AI disposition if player attacked an NPC
        self._update_ai_disposition()

    def _apply_screen_shake(
        self,
        attack_result: ResolutionResult,
        _attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Apply screen shake when player is hit."""
        # Screen shake only when PLAYER gets hit
        # (player's perspective getting jarred)
        base_damage = damage

        # Different shake for different attack types
        if weapon.melee_attack:
            # Melee attacks - use probability instead of pixel distance
            shake_intensity = min(base_damage * 0.15, 0.8)  # 0.15-0.8 probability
            shake_duration = 0.2 + (base_damage * 0.03)
        else:
            # Ranged attacks - lower probability
            shake_intensity = min(base_damage * 0.08, 0.5)  # 0.08-0.5 probability
            shake_duration = 0.1 + (base_damage * 0.02)

        # Extra shake for critical hits against player
        if attack_result.is_critical_success:
            shake_intensity *= 1.5  # Higher probability for crits
            shake_duration *= 1.3

        self.frame_manager.trigger_screen_shake(shake_intensity, shake_duration)

    def _update_ai_disposition(self) -> None:
        """Update AI disposition if player attacked an NPC."""
        if (
            self.attacker == self.controller.gw.player
            and self.defender != self.controller.gw.player
            and isinstance(self.defender.ai, DispositionBasedAI)
            and self.defender.ai.disposition != Disposition.HOSTILE
        ):
            self.defender.ai.disposition = Disposition.HOSTILE
            self.controller.message_log.add_message(
                f"{self.defender.name} becomes hostile towards {self.attacker.name} "
                "due to the attack!",
                colors.ORANGE,
            )


class ReloadAction(GameAction):
    """Action for reloading a ranged weapon."""

    def __init__(self, controller: Controller, actor: Character, weapon: Item) -> None:
        super().__init__(controller, actor)
        self.weapon = weapon

        # Type narrowing.
        self.actor: Character

    def execute(self) -> None:
        ranged_attack = self.weapon.ranged_attack
        if not ranged_attack:
            return

        # Find compatible ammo in inventory
        ammo_item = None
        for item in self.actor.inventory:
            if (
                isinstance(item, Item)
                and item.ammo
                and item.ammo.ammo_type == ranged_attack.ammo_type
            ):
                ammo_item = item
                break

        if ammo_item:
            # Remove ammo from inventory and reload weapon
            self.actor.inventory.remove_from_inventory(ammo_item)
            ranged_attack.current_ammo = ranged_attack.max_ammo
            self.controller.message_log.add_message(
                f"{self.actor.name} reloaded {self.weapon.name}.", colors.GREEN
            )
        else:
            self.controller.message_log.add_message(
                f"No {ranged_attack.ammo_type} ammo available!", colors.RED
            )
