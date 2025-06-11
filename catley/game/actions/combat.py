"""
Combat actions for direct actor-vs-actor confrontations.

Handles attack resolution, damage calculation, weapon reloading, and all
combat-related mechanics including hit/miss determination and effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from game import ranges

from catley import colors
from catley.constants.combat import CombatConstants as Combat
from catley.environment import tile_types
from catley.events import (
    ActorDeathEvent,
    EffectEvent,
    MessageEvent,
    ScreenShakeEvent,
    publish_event,
)
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character, Disposition
from catley.game.ai import DispositionBasedAI
from catley.game.consequences import (
    AttackConsequenceGenerator,
    ConsequenceHandler,
)
from catley.game.enums import OutcomeTier
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.items.properties import WeaponProperty
from catley.game.resolution import combat_arbiter
from catley.game.resolution.base import ResolutionResult
from catley.game.resolution.outcomes import CombatOutcome

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

    def execute(self) -> GameActionResult | None:
        # 1. Determine what attack method to use
        attack, weapon = self._determine_attack_method()
        if not attack:
            return None

        # 2. Validate the attack can be performed
        range_modifiers = self._validate_attack(attack, weapon)
        if range_modifiers is None:
            return None  # Validation failed, error messages already logged

        # 3. Perform the attack roll and immediate effects
        attack_result = self._execute_attack_roll(attack, weapon, range_modifiers)
        if attack_result is None:
            return None

        # 4. Determine combat consequences based on the resolution result
        outcome = combat_arbiter.determine_outcome(
            attack_result, self.attacker, self.defender, weapon
        )

        # 5. Apply the outcome and post-attack effects
        damage = self._apply_combat_outcome(attack_result, outcome, attack, weapon)
        self._handle_post_attack_effects(attack_result, attack, weapon, damage)

        # 6. Generate and apply additional consequences
        generator = AttackConsequenceGenerator()
        consequences = generator.generate(
            attacker=self.attacker,
            weapon=weapon,
            outcome_tier=attack_result.outcome_tier,
        )
        handler = ConsequenceHandler()
        for consequence in consequences:
            handler.apply_consequence(consequence)

        return GameActionResult(consequences=consequences)

    def _determine_attack_method(self) -> tuple[Attack | None, Item]:
        """Determine which attack method and weapon to use
        based on distance and capabilities."""
        weapon = (
            self.weapon
            or self.attacker.inventory.get_active_weapon()
            or FISTS_TYPE.create()
        )

        distance = ranges.calculate_distance(
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
        publish_event(
            MessageEvent(f"{self.attacker.name} has no way to attack!", colors.RED)
        )
        return None, weapon

    def _validate_attack(self, attack: Attack, weapon: Item) -> dict | None:
        """Validate the attack can be performed and return range modifiers."""
        distance = ranges.calculate_distance(
            self.attacker.x, self.attacker.y, self.defender.x, self.defender.y
        )

        range_modifiers = {}
        ranged_attack = weapon.ranged_attack

        # For ranged attacks, check ammo, check line of sight and apply range modifiers
        if ranged_attack is not None and attack == ranged_attack and distance > 1:
            if ranged_attack.current_ammo <= 0:
                publish_event(
                    MessageEvent(f"{weapon.name} is out of ammo!", colors.RED)
                )
                return None

            # Check line of sight
            if not ranges.has_line_of_sight(
                self.controller.gw.game_map,
                self.attacker.x,
                self.attacker.y,
                self.defender.x,
                self.defender.y,
            ):
                publish_event(
                    MessageEvent(f"No clear shot to {self.defender.name}!", colors.RED)
                )
                return None

            # Get range modifiers
            range_category = ranges.get_range_category(distance, weapon)
            range_modifiers = ranges.get_range_modifier(weapon, range_category)

            if range_modifiers is None:
                publish_event(
                    MessageEvent(f"{self.defender.name} is out of range!", colors.RED)
                )
                return None

        return range_modifiers

    def _adjacent_cover_bonus(self) -> int:
        """Return the highest cover bonus adjacent to the defender."""
        game_map = self.controller.gw.game_map
        max_bonus = 0
        x, y = self.defender.x, self.defender.y

        # Cover checks are infrequent, so we prioritize memory usage over
        # lookup speed by querying tile types directly rather than caching a
        # full map of bonuses.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < game_map.width and 0 <= ny < game_map.height:
                    tile_id = game_map.tiles[nx, ny]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))
                    bonus = int(tile_data["cover_bonus"])
                    max_bonus = max(max_bonus, bonus)

        return max_bonus

    def _execute_attack_roll(
        self, attack: Attack, weapon: Item, range_modifiers: dict
    ) -> ResolutionResult | None:
        """Perform the attack roll and handle immediate effects
        like ammo consumption."""
        stat_name = attack.stat_name
        attacker_score = getattr(self.attacker.stats, stat_name)
        defender_score = self.defender.stats.agility + self._adjacent_cover_bonus()

        resolution_args = {"has_advantage": False, "has_disadvantage": False}
        for effect in self.attacker.status_effects:
            resolution_args = effect.apply_to_resolution(resolution_args)

        if resolution_args.get("action_prevented", False):
            return None

        final_advantage = range_modifiers.get(
            "has_advantage", False
        ) or resolution_args.get("has_advantage", False)
        final_disadvantage = range_modifiers.get(
            "has_disadvantage", False
        ) or resolution_args.get("has_disadvantage", False)

        resolver = self.controller.create_resolver(
            ability_score=attacker_score,
            roll_to_exceed=defender_score + 10,
            has_advantage=final_advantage,
            has_disadvantage=final_disadvantage,
        )
        attack_result = resolver.resolve(self.attacker, self.defender, weapon)

        # Consume ammo for ranged attacks
        ranged_attack = weapon.ranged_attack
        if attack == ranged_attack and ranged_attack is not None:
            if ranged_attack.current_ammo > 0:
                ranged_attack.current_ammo -= 1
            else:
                # This shouldn't happen if validation worked, but safety check
                publish_event(
                    MessageEvent(f"No ammo left in {weapon.name}!", colors.RED)
                )

        # Emit muzzle flash for ranged attacks
        if attack == weapon.ranged_attack:
            self._emit_muzzle_flash()

        return attack_result

    def _emit_muzzle_flash(self) -> None:
        """Emit muzzle flash particle effect."""
        direction_x = self.defender.x - self.attacker.x
        direction_y = self.defender.y - self.attacker.y

        publish_event(
            EffectEvent(
                "muzzle_flash",
                self.attacker.x,
                self.attacker.y,
                direction_x=direction_x,
                direction_y=direction_y,
            )
        )

    def _apply_combat_outcome(
        self,
        attack_result: ResolutionResult,
        outcome: CombatOutcome,
        attack: Attack,
        weapon: Item,
    ) -> int:
        """Apply the combat outcome and return the damage dealt."""
        if attack_result.outcome_tier in (
            OutcomeTier.SUCCESS,
            OutcomeTier.CRITICAL_SUCCESS,
            OutcomeTier.PARTIAL_SUCCESS,
        ):
            damage = outcome.damage_dealt
            if outcome.armor_damage > 0:
                self.defender.health.ap = max(
                    0, self.defender.health.ap - outcome.armor_damage
                )
            if outcome.injury_inflicted is not None:
                self.defender.inventory.add_to_inventory(outcome.injury_inflicted)
            if damage > 0:
                self.defender.take_damage(damage)
                publish_event(
                    EffectEvent(
                        "blood_splatter",
                        self.defender.x,
                        self.defender.y,
                        intensity=damage / 20.0,
                    )
                )
                self._log_hit_message(attack_result, weapon, damage)

                if not self.defender.health.is_alive():
                    publish_event(
                        MessageEvent(
                            f"{self.defender.name} has been killed!", colors.RED
                        )
                    )
                    publish_event(ActorDeathEvent(self.defender))
            return damage

        self._handle_attack_miss(attack_result, attack, weapon)
        return 0

    def _handle_attack_miss(
        self, attack_result: ResolutionResult, attack: Attack, weapon: Item
    ) -> None:
        """Handle an attack miss - messages and weapon property effects."""
        # Miss
        if attack_result.outcome_tier == OutcomeTier.CRITICAL_FAILURE:
            # "Crits favoring defense cause the attacker's weapon to break,
            # and leave them confused or off-balance."
            # FIXME: Break the attacker's weapon.
            # FIXME: If the attacker is unarmed and attacking with fists or
            # kicking, etc., they pull a muscle and have disadvantage on their
            # next attack. (house rule)
            # FIXME: Give the attacker the condition "confused" or "off-balance".

            miss_color = colors.ORANGE  # A warning color for critical miss
            publish_event(
                MessageEvent(
                    (
                        f"Critical miss! {self.attacker.name}'s attack on "
                        f"{self.defender.name} fails."
                    ),
                    miss_color,
                )
            )
        else:
            miss_color = colors.GREY  # Standard miss color
            publish_event(
                MessageEvent(
                    f"{self.attacker.name} misses {self.defender.name}.",
                    miss_color,
                )
            )

        # Handle 'awkward' weapon property on miss
        if (
            weapon
            and weapon.melee_attack
            and attack is weapon.melee_attack
            and WeaponProperty.AWKWARD in weapon.melee_attack.properties
        ):
            publish_event(
                MessageEvent(
                    f"{self.attacker.name} is off balance from the awkward swing "
                    f"with {weapon.name}!",
                    colors.LIGHT_BLUE,
                )
            )
            # TODO: Implement off-balance effect (maybe skip next turn?)

    def _log_hit_message(
        self, attack_result: ResolutionResult, weapon: Item, damage: int
    ) -> None:
        """Log appropriate hit message based on critical status."""
        if attack_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
            hit_color = colors.YELLOW
            message = (
                f"Critical hit! {self.attacker.name} strikes {self.defender.name} "
                f"with {weapon.name} for {damage} damage."
            )
        else:
            hit_color = colors.WHITE  # Default color for a standard hit
            message = (
                f"{self.attacker.name} hits {self.defender.name} "
                f"with {weapon.name} for {damage} damage."
            )
        hp_message_part = (
            f" ({self.defender.name} has {self.defender.health.hp} HP left.)"
        )
        publish_event(MessageEvent(message + hp_message_part, hit_color))

    def _handle_post_attack_effects(
        self,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Handle post-attack effects like screen shake and AI disposition changes."""
        # Screen shake only when PLAYER gets hit
        if (
            attack_result.outcome_tier
            in (
                OutcomeTier.SUCCESS,
                OutcomeTier.CRITICAL_SUCCESS,
                OutcomeTier.PARTIAL_SUCCESS,
            )
            and self.defender == self.controller.gw.player
        ):
            self._apply_screen_shake(attack_result, attack, weapon, damage)

        # Update AI disposition if player attacked an NPC
        self._update_ai_disposition()

    def _apply_screen_shake(
        self,
        attack_result: ResolutionResult,
        attack: Attack,
        weapon: Item,
        damage: int,
    ) -> None:
        """Apply screen shake when player is hit."""
        # Screen shake only when PLAYER gets hit
        # (player's perspective getting jarred)
        base_damage = damage

        # Different shake for different attack types
        if weapon.melee_attack and attack is weapon.melee_attack:
            # Melee attacks - use probability instead of pixel distance
            shake_intensity = min(
                base_damage * Combat.MELEE_SHAKE_INTENSITY_MULT,
                Combat.MELEE_SHAKE_INTENSITY_CAP,
            )
            shake_duration = (
                Combat.MELEE_SHAKE_DURATION_BASE
                + base_damage * Combat.MELEE_SHAKE_DURATION_MULT
            )
        else:
            # Ranged attacks - lower probability
            shake_intensity = min(
                base_damage * Combat.RANGED_SHAKE_INTENSITY_MULT,
                Combat.RANGED_SHAKE_INTENSITY_CAP,
            )
            shake_duration = (
                Combat.RANGED_SHAKE_DURATION_BASE
                + base_damage * Combat.RANGED_SHAKE_DURATION_MULT
            )

        # Extra shake for critical hits against player
        if attack_result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
            shake_intensity *= Combat.CRIT_SHAKE_INTENSITY_MULT
            shake_duration *= Combat.CRIT_SHAKE_DURATION_MULT

        publish_event(ScreenShakeEvent(shake_intensity, shake_duration))

    def _update_ai_disposition(self) -> None:
        """Update AI disposition if player attacked an NPC."""
        if (
            self.attacker == self.controller.gw.player
            and self.defender != self.controller.gw.player
            and isinstance(self.defender.ai, DispositionBasedAI)
            and self.defender.ai.disposition != Disposition.HOSTILE
        ):
            self.defender.ai.disposition = Disposition.HOSTILE
            publish_event(
                MessageEvent(
                    (
                        f"{self.defender.name} becomes hostile towards "
                        f"{self.attacker.name} due to the attack!"
                    ),
                    colors.ORANGE,
                )
            )


class ReloadAction(GameAction):
    """Action for reloading a ranged weapon."""

    def __init__(self, controller: Controller, actor: Character, weapon: Item) -> None:
        super().__init__(controller, actor)
        self.weapon = weapon

        # Type narrowing.
        self.actor: Character

    def execute(self) -> GameActionResult | None:
        ranged_attack = self.weapon.ranged_attack
        if not ranged_attack:
            return None

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
            publish_event(
                MessageEvent(
                    f"{self.actor.name} reloaded {self.weapon.name}.", colors.GREEN
                )
            )
        else:
            publish_event(
                MessageEvent(
                    f"No {ranged_attack.ammo_type} ammo available!", colors.RED
                )
            )
        return None
