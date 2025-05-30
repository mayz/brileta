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
from catley.util import dice

from . import items
from .actors import Actor, Disposition
from .ai import DispositionBasedAI

if TYPE_CHECKING:
    from catley.controller import Controller


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

    @abc.abstractmethod
    def execute(self) -> None:
        """Execute this action."""
        pass


class MoveAction(GameAction):
    """Action for moving an actor on the game map."""

    def __init__(self, controller: Controller, actor: Actor, dx: int, dy: int) -> None:
        super().__init__(controller, actor)
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
                if actor.health and actor.health.is_alive():
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
        self, controller: Controller, attacker: Actor, defender: Actor
    ) -> None:
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender

    def execute(self) -> None:
        # Determine attacker's ability score
        weapon = self.attacker.inventory.equipped_weapon or items.FISTS
        if weapon.melee:
            attacker_ability_score = self.attacker.stats.strength
        else:
            attacker_ability_score = self.attacker.stats.observation

        defender_ability_score = self.defender.stats.agility

        # Perform attack roll
        attack_result = dice.perform_opposed_check_roll(
            attacker_ability_score,
            defender_ability_score,
            has_advantage=False,
            has_disadvantage=False,
        )

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

                if self.defender.health.ap > 0:
                    self.defender.health.ap = 0
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
            hp_message_part = (
                f" ({self.defender.name} has {self.defender.health.hp} HP left.)"
            )
            self.controller.message_log.add_message(
                hit_message + hp_message_part, hit_color
            )

            # Check if defender is defeated
            if not self.defender.health.is_alive():
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

        # If the player attacked an `Actor`, they become hostile towards the player.
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
