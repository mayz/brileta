"""
This module contains the Action class and its subclasses that represent game actions.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import colors
import dice
import items
from combat import CombatManager
from model import Actor, WastoidActor
from play_mode import PlayMode

if TYPE_CHECKING:
    import tcod
    from engine import Controller
    from model import Entity


class Action(abc.ABC):
    """Base class for all game actions."""

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute the action."""
        raise NotImplementedError()


class MoveAction(Action):
    """Action for moving an entity on the game map."""

    def __init__(
        self, controller: Controller, entity: Entity, dx: int, dy: int
    ) -> None:
        self.controller = controller
        self.game_map = controller.model.game_map
        self.entity = entity

        self.dx = dx
        self.dy = dy
        self.newx = self.entity.x + self.dx
        self.newy = self.entity.y + self.dy

    def execute(self) -> None:
        if self.game_map.tiles[self.newx][self.newy].blocked:
            return

        # Check for blocking entities
        for entity in self.controller.model.entities:
            if (
                entity.blocks_movement
                and entity.x == self.newx
                and entity.y == self.newy
            ):
                # Bumping an actor no longer initiates combat.
                # Instead, the player can choose to attack.
                # TODO: Implement.
                # if isinstance(entity, Actor) and entity.is_alive():
                # attack_action = AttackAction(
                # controller=self.controller,
                # attacker=self.entity,
                # defender=entity,
                # )
                # attack_action.execute()
                return  # Cannot move into blocking entity

        self.entity.move(self.dx, self.dy)
        self.controller.fov.fov_needs_recomputing = True


class ToggleFullscreenAction(Action):
    """Action for toggling fullscreen mode."""

    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitAction(Action):
    """Action for quitting the game."""

    def execute(self) -> None:
        raise SystemExit()


class AttackAction(Action):
    """Action for attacking another actor in combat."""

    def __init__(
        self, controller: Controller, attacker: Actor, defender: Actor
    ) -> None:
        self.controller = controller
        self.attacker = attacker
        self.defender = defender

    def execute(self) -> None:
        # Determine attacker's ability score
        weapon = self.attacker.equipped_weapon or items.FISTS
        if weapon.melee:
            attacker_ability_score = self.attacker.strength
        else:
            attacker_ability_score = self.attacker.observation

        if isinstance(self.defender, WastoidActor):
            defender_ability_score = self.defender.agility
        else:
            defender_ability_score = 0

        # Perform attack roll
        attack_result = dice.perform_opposed_check_roll(
            attacker_ability_score,
            defender_ability_score,
            has_advantage=False,
            has_disadvantage=False,
        )

        # TODO: Ask LLM whether/how to implement combat as a separate "mode",
        #       like entering combat in Fallout 1 or 2. Rather than the current
        #       "bash into opponent" a la Brogue. Would be fun to show the player
        #       a menu of options, including regular attacks, stunts, etc., with
        #       a probability of success shown for each.
        #       The character of the entity being targeted could pulsate in color.

        # FIXME: Rather than handling combat here in AttackAction, have some
        # kind of separate combat module that handles, e.g., initiative, turns,
        # movement, stunts, etc. Rather than automatically making an attack roll
        # on every turn like in Brogue.
        #
        # Stunts can be fun - e.g., "attempt to trip", "attempt to disarm", etc.
        # Then if character is knocked down, future attack rolls made on them are
        # with advantage.
        #
        # "Stunts are combat maneuvers performed with an action and an opposed
        # test, like disarming, tripping, grappling, or otherwise vying for an
        # advantage. A character can combine a stunt with an attack into a
        # single action and roll (like: hit ’em so hard they fall down) by
        # foregoing a bonus from advantage, or by accepting a disadvantage."

        # Does the attack hit?
        if attack_result.success:
            # Hit - roll damage
            damage_dice = weapon.damage_dice
            damage = damage_dice.roll()

            if attack_result.is_critical_hit:
                # "Crits favoring an attack deal an extra die of damage and break
                # the target’s armor (lowering to 0 AP) before applying damage - or
                # cause an injury to an unarmored target."
                damage += damage_dice.roll()

                if self.defender.ap > 0:
                    self.defender.ap = 0
                    # FIXME: Break the defender's armor.
                else:
                    # FIXME: Give the defender an injury.
                    pass

            # Apply damage.
            self.defender.take_damage(damage)

            # Log hit message
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
            hp_message_part = f" ({self.defender.name} has {self.defender.hp} HP left.)"
            self.controller.message_log.add_message(
                hit_message + hp_message_part, hit_color
            )

            # Check if defender is defeated
            if not self.defender.is_alive():
                self.controller.message_log.add_message(
                    f"{self.defender.name} has been killed!", colors.RED
                )
        else:
            # Miss
            if attack_result.is_critical_miss:
                # "Crits favoring defense cause the attacker’s weapon to break,
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
            if weapon and "awkward" in weapon.properties:
                self.controller.message_log.add_message(
                    f"{self.attacker.name} is off balance from the awkward swing "
                    f"with {weapon.name}!",
                    colors.LIGHT_BLUE,  # Informational color for status effects
                )
                # TODO: Implement off-balance effect (maybe skip next turn?)

        # Signal that the attacker's turn is over
        # TODO: Implement turn system to handle this


class ToggleCombatModeAction(Action):
    """Action for entering or exiting combat mode."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.model = controller.model
        self.player = controller.model.player

    def execute(self) -> None:
        if self.model.play_mode == PlayMode.COMBAT:
            # Exit combat mode
            self.model.play_mode = PlayMode.ROAMING
            self.model.combat_manager = None
        else:
            # Enter combat mode
            self.model.combat_manager = CombatManager(self.controller)
            self.model.combat_manager.start_planning_phase()
            self.model.play_mode = PlayMode.COMBAT
