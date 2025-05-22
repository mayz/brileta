"""
This module contains the Action class and its subclasses that represent game actions.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import dice
import items
import tcod
from model import Actor, WastoidActor

if TYPE_CHECKING:
    from engine import Controller
    from model import Actor, Entity


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
                if isinstance(entity, Actor) and entity.is_alive():
                    attack_action = AttackAction(
                        controller=self.controller,
                        attacker=self.entity,
                        defender=entity,
                    )
                    attack_action.execute()
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

        ##########
        # FIXME: Handle critical hits (20) and misses (1).
        #
        # "Crits favoring an attack deal an extra die of damage and
        # break the target’s armor (lowering to 0 AP) before applying damage—or
        # cause an injury to an unarmored target! Crits favoring defense cause the
        # attacker’s weapon to break, and leave them confused or off-balance.
        ##########

        # Does the attack hit?
        if attack_result.success:
            # Hit - roll damage
            damage_dice = weapon.damage_dice
            damage = damage_dice.roll()

            if attack_result.is_critical_hit:
                # Roll an extra die of damage.
                damage += damage_dice.roll()

            # Apply damage.
            self.defender.take_damage(damage)

            # Log the hit
            critical_hit_text = (
                "Critical hit! " if attack_result.is_critical_hit else ""
            )
            print(
                f"{critical_hit_text}{self.attacker.ch} hits {self.defender.ch} "
                f"with {weapon.name} for {damage} damage. "
                f"({self.defender.ch} has {self.defender.hp} HP left.)"
            )

            # Check if defender is defeated
            if not self.defender.is_alive():
                print(f"{self.defender.ch} has been killed!")
        else:
            # Miss
            print(
                "Critical miss! "
                if attack_result.is_critical_miss
                else f"{self.attacker.ch} misses {self.defender.ch}."
            )

            # Handle 'awkward' weapon property
            if (
                weapon
                and hasattr(weapon, "properties")
                and "awkward" in weapon.properties
            ):
                print(f"{self.attacker.ch} is off balance from the awkward swing!")
                # TODO: Implement off-balance effect (maybe skip next turn?)

        # Signal that the attacker's turn is over
        # TODO: Implement turn system to handle this
