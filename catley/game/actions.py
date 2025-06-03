"""
Game world actions that affect gameplay state.

Defines actions that actors can perform within the game world that directly affect
game state and typically consume turns in the action economy.

GameAction:
    Base class for all in-world actions. These represent meaningful decisions
    made by actors (player or NPCs) that change the game world state.

Examples:
    - MoveAction: Moving an actor to a new position
    - AttackAction: One actor attacking another in combat

These actions are distinct from UI commands - they represent actual gameplay
decisions rather than interface interactions. Game actions typically:
- Are performed by actors with agency
- Consume the actor's turn
- Can trigger consequences and reactions from other actors
- Advance the game's narrative/mechanical state
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from catley import colors
from catley.game import range_system
from catley.game.items.item_core import Item
from catley.util import dice

from .actors import Actor, Character, Disposition
from .ai import DispositionBasedAI
from .items.item_types import FISTS_TYPE

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.items.capabilities import Attack


class GameAction(abc.ABC):
    """An action that represents a game turn, such as moving an actor or
    performing an attack. This is distinct from UI actions and is meant to
    be executed within the game loop.


    An action performed by an actor (player or NPC) that directly affects the
    game world's state and typically consumes that actor's turn. Examples include
    moving, attacking, performing stunts or tricks, etc.
    """

    def __init__(self, controller: Controller, actor: Actor) -> None:
        self.controller = controller
        self.actor = actor
        self.renderer = self.controller.renderer

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute this action."""
        pass

    def can_execute(self) -> tuple[bool, str | None]:
        """Check if this action can be executed. Returns (can_execute, reason_if_not)"""
        return True, None

    def get_success_probability(self) -> float | None:
        """Get probability of success for this action, if applicable"""
        return None


class MoveAction(GameAction):
    """Action for moving an actor on the game map."""

    def __init__(
        self, controller: Controller, actor: Character, dx: int, dy: int
    ) -> None:
        super().__init__(controller, actor)

        # Type narrowing.
        self.actor: Character

        self.game_map = controller.gw.game_map

        self.dx = dx
        self.dy = dy
        self.newx = self.actor.x + self.dx
        self.newy = self.actor.y + self.dy

    def execute(self) -> None:
        if not self.game_map.walkable[self.newx, self.newy]:
            return

        # Check for blocking actors.
        for actor in self.controller.gw.actors:
            if actor.blocks_movement and actor.x == self.newx and actor.y == self.newy:
                if isinstance(actor, Character) and actor.health.is_alive():
                    attack_action = AttackAction(
                        controller=self.controller,
                        attacker=self.actor,
                        defender=actor,
                    )
                    attack_action.execute()
                return  # Cannot move into blocking actor

        self.actor.move(self.dx, self.dy)


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
        self._resolve_attack_outcome(attack_result, attack, weapon)

        # 5. Handle post-attack effects (screen shake, messages, AI)
        self._handle_post_attack_effects(attack_result, attack, weapon)

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
    ) -> dice.CheckResult:
        """Perform the attack roll and handle immediate effects
        like ammo consumption."""
        # Get ability scores for the roll
        stat_name = attack.stat_name
        attacker_ability_score = getattr(self.attacker.stats, stat_name)
        defender_ability_score = self.defender.stats.agility

        # Perform attack roll
        attack_result = dice.perform_opposed_check_roll(
            attacker_ability_score,
            defender_ability_score,
            has_advantage=range_modifiers.get("has_advantage", False),
            has_disadvantage=range_modifiers.get("has_disadvantage", False),
        )

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

        self.renderer.create_effect(
            "muzzle_flash",
            x=self.attacker.x,
            y=self.attacker.y,
            direction_x=direction_x,
            direction_y=direction_y,
        )

    def _resolve_attack_outcome(
        self, attack_result: dice.CheckResult, attack: Attack, weapon: Item
    ) -> None:
        """Resolve the attack outcome - apply damage, handle crits, emit particles."""
        if attack_result.success:
            self._handle_successful_hit(attack_result, attack, weapon)
        else:
            self._handle_attack_miss(attack_result, attack, weapon)

    def _handle_successful_hit(
        self, attack_result: dice.CheckResult, attack: Attack, weapon: Item
    ) -> None:
        """Handle a successful attack hit - damage, crits, particles, messages."""
        # Hit - roll damage
        damage_dice = attack.damage_dice
        damage = damage_dice.roll()

        if attack_result.is_critical_hit:
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
            self.renderer.create_effect(
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

    def _handle_attack_miss(
        self, attack_result: dice.CheckResult, attack: Attack, weapon: Item
    ) -> None:
        """Handle an attack miss - messages and weapon property effects."""
        # Miss
        if attack_result.is_critical_miss:
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
            and "awkward" in weapon.melee_attack.properties
        ):
            self.controller.message_log.add_message(
                f"{self.attacker.name} is off balance from the awkward swing "
                f"with {weapon.name}!",
                colors.LIGHT_BLUE,  # Informational color for status effects
            )
            # TODO: Implement off-balance effect (maybe skip next turn?)

    def _log_hit_message(
        self, attack_result: dice.CheckResult, weapon: Item, damage: int
    ) -> None:
        """Log appropriate hit message based on critical status."""
        if attack_result.is_critical_hit:
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
        self, attack_result: dice.CheckResult, attack: Attack, weapon: Item
    ) -> None:
        """Handle post-attack effects like screen shake and AI disposition changes."""
        # Screen shake only when PLAYER gets hit
        if attack_result.success and self.defender == self.controller.gw.player:
            self._apply_screen_shake(attack_result, attack, weapon)

        # Update AI disposition if player attacked an NPC
        self._update_ai_disposition()

    def _apply_screen_shake(
        self, attack_result: dice.CheckResult, attack: Attack, weapon: Item
    ) -> None:
        """Apply screen shake when player is hit."""
        # Screen shake only when PLAYER gets hit
        # (player's perspective getting jarred)
        damage_dice = attack.damage_dice
        base_damage = damage_dice.roll()  # Approximate damage for shake calculation

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
        if attack_result.is_critical_hit:
            shake_intensity *= 1.5  # Higher probability for crits
            shake_duration *= 1.3

        self.controller.renderer.trigger_screen_shake(shake_intensity, shake_duration)

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
