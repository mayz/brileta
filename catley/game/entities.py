"""
Entities in the game world.

This module defines the fundamental classes for objects that exist in the game world:

Entity:
    Base class for any object with a position that can be rendered and interacted with.
    Examples: treasure chests, doors, traps, projectiles, mechanisms.

Actor:
    An entity with agency. It has wants, motivations, and makes autonomous decisions.
    Participates in combat, social dynamics, and the turn-based simulation.
    Examples: player character, NPCs, monsters, robots, sentient creatures.

The key distinction is agency: Actors have internal desires and make decisions based on
those motivations, while Entities are reactive objects that follow programmed behaviors.

Most game objects will be Actors, but this architecture provides flexibility for
implementing complex interactive objects that don't need full autonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors

from .components import (
    HealthComponent,
    InventoryComponent,
    StatsComponent,
    VisualEffectsComponent,
)
from .enums import Disposition

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.lighting import LightSource
    from catley.world.game_state import GameWorld

    from .actions import GameAction
    from .ai import AIComponent


class Entity:
    """Any object that exists in the game world with a position.

    Entities are reactive and interactive, but they follow programmed behavior.
    They don't have their own motivations or wants.

    Examples: treasure chests, certain doors, traps, mechanisms, etc.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        game_world: GameWorld | None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.ch = ch  # Character that represents the entity.
        self.color = color
        self.gw = game_world
        self.light_source = light_source
        self.blocks_movement = blocks_movement
        if self.light_source and self.gw:
            self.light_source.attach(self, self.gw.lighting)
        self.name = "<Unnamed Entity>"

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
        # Update the light source position when entity moves
        if self.light_source:
            self.light_source.position = (self.x, self.y)

    def update_turn(self, controller: Controller) -> None:
        """Called once per game turn for this entity to perform its turn-based logic.
        Base implementation does nothing. Subclasses should override this.
        """
        pass


class Actor(Entity):
    """An entity with agency.

    It has wants, motivations, and makes autonomous decisions. It participates in the
    social/economic/narrative simulation of the game.

    Examples: player, NPCs, monsters, sentient robots, animals.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str = "<Unnamed Actor>",
        stats: StatsComponent | None = None,
        health: HealthComponent | None = None,
        inventory: InventoryComponent | None = None,
        visual_effects: VisualEffectsComponent | None = None,
        ai: AIComponent | None = None,
        # World and appearance
        game_world: GameWorld | None = None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        disposition: Disposition = Disposition.WARY,
        speed: int = 100,
    ) -> None:
        super().__init__(x, y, ch, color, game_world, light_source, blocks_movement)
        self.name = name

        self.stats = stats or StatsComponent()
        self.health = health or HealthComponent(self.stats)
        self.inventory = inventory or InventoryComponent(self.stats)
        self.visual_effects = visual_effects or VisualEffectsComponent()
        self.ai = ai

        self.disposition = disposition
        self.speed = speed
        self.accumulated_energy: int = self.speed

        self.tricks: list = []  # Will hold Trick instances later

    def take_damage(self, amount: int) -> None:
        """Handle damage to the actor.

        That includes:
        - Update health math.
        - Visual feedback.
        - Handle death consequences, if any.

        Args:
            amount: Amount of damage to take
        """
        # Delegate health math to the health component.
        self.health.take_damage(amount)

        # Visual feedback.
        self.visual_effects.flash(colors.RED)

        if not self.health.is_alive():
            # Handle death consequences.
            self.ch = "x"
            self.color = colors.DEAD
            self.blocks_movement = False
            # If this actor was selected, deselect it.
            if self.gw and self.gw.selected_entity == self:
                self.gw.selected_entity = None

    def update_turn(self, controller: Controller) -> None:
        """
        Handles turn-based logic for actors. For NPCs, this includes AI.
        For the player, this could handle passive effects like poison/regeneration.
        """
        super().update_turn(controller)

        # TODO: Implement passive effects here if any (e.g., poison, regeneration)

        # For example:
        # if self.has_condition("Poisoned"):
        #     self.take_damage(1)
        #     controller.message_log.add_message(
        #         f"{self.name} takes 1 poison damage.", colors.GREEN)
        #
        # Active actions (including AI for NPCs and player input) are now
        # handled via get_next_action() in the main game loop.
        pass

    def get_next_action(self, controller: Controller) -> GameAction | None:
        """
        Determines the next action for this actor.
        For the player, it retrieves the pending action from the event handler.
        For NPCs, it queries their AI component.
        """
        if self == controller.gw.player:
            action = controller.event_handler.pending_action
            controller.event_handler.pending_action = None  # Clear after retrieving
            return action
        if self.ai and self.health.is_alive():
            # Allow AI to update its internal state before deciding on an action.
            self.ai.update(
                controller
            )  # Pass only controller as per AIComponent.update signature
            return self.ai.get_action(controller, self)
        return None

    def regenerate_energy(self) -> None:
        """Regenerates energy for the actor based on their speed."""
        self.accumulated_energy += self.speed

    def can_afford_action(self, cost: int) -> bool:
        """Checks if the actor has enough energy to perform an action."""
        return self.accumulated_energy >= cost

    def spend_energy(self, cost: int) -> None:
        self.accumulated_energy -= cost
