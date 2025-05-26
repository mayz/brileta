"""
This module contains the Action class and its subclasses that represent game actions.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import colors
import dice
import items
from menu_system import Menu, PickupMenu
from model import Actor, WastoidActor

if TYPE_CHECKING:
    import tcod
    from controller import Controller
    from model import Entity


class Action(abc.ABC):
    """Abstract base class for any command or operation that can be performed within
    the game, whether by an entity in the game world or by the player interacting
    with the UI.

    The two main subclasses of Action are `UIAction` and `GameTurnAction`.
    """

    @abc.abstractmethod
    def execute(self) -> None:
        """Perform the action."""
        raise NotImplementedError()


class UIAction(Action):
    """An action for UI-related operations, such as opening menus,
    toggling fullscreen, or quitting the game.
    """

    pass


class GameTurnAction(Action):
    """An action that represents a game turn, such as moving an entity or
    performing an attack. This is distinct from UI actions and is meant to
    be executed within the game loop.


    An action performed by an entity (player or NPC) that directly affects the
    game world's state and typically consumes that entity's turn. Examples include
    moving, attacking, performing stunts or tricks, etc.
    """

    def __init__(self, controller: Controller, entity: Entity) -> None:
        self.controller = controller
        self.entity = entity


################################################################################


class ToggleFullscreenAction(UIAction):
    """Action for toggling fullscreen mode."""

    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitAction(UIAction):
    """Action for quitting the game."""

    def execute(self) -> None:
        raise SystemExit()


class SelectOrDeselectEntityAction(UIAction):
    """
    Action to select a new entity or deselect (clear) the current selection.
    """

    def __init__(self, controller: Controller, selection: Entity | None):
        """
        Args:
            controller: The game controller.
            selection: The entity to select (or None to deselect).
        """
        super().__init__()
        self.controller = controller
        self.selection = selection

    def execute(self) -> None:
        self.controller.model.selected_entity = self.selection


class OpenMenuAction(UIAction):
    """Action to open a menu, like the inventory or help menu."""

    def __init__(self, controller: Controller, menu_class: type[Menu]):
        self.controller = controller
        self.menu_class = menu_class

    def execute(self) -> None:
        menu = self.menu_class(self.controller)
        self.controller.menu_system.show_menu(menu)


class OpenPickupMenuAction(OpenMenuAction):
    """Action to open the pickup menu for items at the player's location."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.player = controller.model.player

    def execute(self) -> None:
        # Check if there are items to pick up, otherwise do nothing.
        if self.controller.model.has_pickable_items_at_location(
            self.player.x, self.player.y
        ):
            menu = PickupMenu(self.controller, (self.player.x, self.player.y))
            self.controller.menu_system.show_menu(menu)


################################################################################


class MoveAction(GameTurnAction):
    """Action for moving an entity on the game map."""

    def __init__(
        self, controller: Controller, entity: Entity, dx: int, dy: int
    ) -> None:
        super().__init__(controller, entity)
        self.game_map = controller.model.game_map

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


class AttackAction(GameTurnAction):
    """Action for one `Actor` attacking another in combat."""

    def __init__(
        self, controller: Controller, attacker: Actor, defender: Actor
    ) -> None:
        super().__init__(controller, attacker)
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
