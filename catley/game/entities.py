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

from enum import Enum, auto
from typing import TYPE_CHECKING

from catley import colors

from .components import (
    HealthComponent,
    InventoryComponent,
    StatsComponent,
    VisualEffectsComponent,
)

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.lighting import LightSource
    from catley.world.game_state import GameWorld

    from .actions import GameAction


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


class Disposition(Enum):
    HOSTILE = auto()  # Will attack/flee.
    UNFRIENDLY = auto()
    WARY = auto()
    APPROACHABLE = auto()
    FRIENDLY = auto()
    ALLY = auto()


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
        # World and appearance
        game_world: GameWorld | None = None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        disposition: Disposition = Disposition.WARY,
    ) -> None:
        super().__init__(x, y, ch, color, game_world, light_source, blocks_movement)
        self.name = name

        self.stats = stats or StatsComponent()
        self.health = health or HealthComponent(self.stats)
        self.inventory = inventory or InventoryComponent(self.stats)
        self.visual_effects = visual_effects or VisualEffectsComponent()

        self.disposition = disposition

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

        player = controller.gw.player
        if self == player:
            # TODO: Implement passive player effects here if any
            # (e.g., poison, regeneration)

            # For example:
            # if self.has_condition("Poisoned"):
            #     self.take_damage(1)
            #     controller.message_log.add_message(
            #         f"{self.name} takes 1 poison damage.", colors.GREEN)
            #
            # The player's active actions are driven by input via EventHandler.
            return

        # NPC AI logic.
        # TODO: Move this to a separate module for better organization later on.

        if not self.health.is_alive() or not player.health.is_alive():
            return

        from catley.game.actions import AttackAction, MoveAction

        # Determine action based on proximity to player
        dx = player.x - self.x
        dy = player.y - self.y
        distance = abs(dx) + abs(dy)  # Manhattan distance

        action_to_perform: GameAction | None = None

        # Default aggro/awareness radii - can be overridden by specific NPC types
        # if needed. These could also be properties of the NPC instance itself.
        aggro_radius = getattr(self, "aggro_radius", 8)
        # awareness_radius = getattr(self, "awareness_radius", 5)
        # personal_space_radius = getattr(self, "personal_space_radius", 3)
        # greeting_radius = getattr(self, "greeting_radius", 4)

        # --- AI Logic based on Disposition ---
        if self.disposition == Disposition.HOSTILE:
            if distance == 1:  # Adjacent
                controller.message_log.add_message(
                    f"{self.name} lunges at {player.name}!", colors.RED
                )
                action_to_perform = AttackAction(controller, self, player)
            elif distance <= aggro_radius:
                move_dx = 0
                move_dy = 0
                if dx != 0:
                    move_dx = 1 if dx > 0 else -1
                if dy != 0:
                    move_dy = 1 if dy > 0 else -1

                potential_moves = []
                if move_dx != 0 and move_dy != 0:
                    potential_moves.append((move_dx, move_dy))
                if move_dx != 0:
                    potential_moves.append((move_dx, 0))
                if move_dy != 0:
                    potential_moves.append((0, move_dy))

                moved_this_turn = False
                for test_dx, test_dy in potential_moves:
                    target_x, target_y = self.x + test_dx, self.y + test_dy
                    if not (
                        0 <= target_x < controller.gw.game_map.width
                        and 0 <= target_y < controller.gw.game_map.height
                    ):
                        continue
                    if controller.gw.game_map.tile_blocked[target_x, target_y]:
                        continue
                    blocking_entity = controller.gw.get_entity_at_location(
                        target_x, target_y
                    )
                    if (
                        blocking_entity
                        and blocking_entity != player
                        and blocking_entity.blocks_movement
                    ):
                        continue

                    controller.message_log.add_message(
                        f"{self.name} charges towards {player.name}.", colors.ORANGE
                    )
                    action_to_perform = MoveAction(controller, self, test_dx, test_dy)
                    moved_this_turn = True
                    break
                if not moved_this_turn:
                    controller.message_log.add_message(
                        f"{self.name} snarls, unable to reach {player.name}.",
                        colors.ORANGE,
                    )
            else:  # Hostile but player too far
                controller.message_log.add_message(
                    f"{self.name} prowls menacingly.", colors.ORANGE
                )

        elif self.disposition == Disposition.UNFRIENDLY:
            # if distance == 1:
            # controller.message_log.add_message(
            # f"'{player.name}? Keep your distance!' {self.name} growls.",
            # colors.ORANGE)
            # TODO: Consider a "shove" or "move away" action if possible,
            # or prepare to defend.
            # elif distance <= awareness_radius:
            # controller.message_log.add_message(
            # f"{self.name} eyes {player.name} with distaste.", colors.YELLOW)
            pass

        elif self.disposition == Disposition.WARY:
            pass

        elif self.disposition == Disposition.APPROACHABLE:
            # if distance <= greeting_radius and distance > 1: # Not directly adjacent
            # controller.message_log.add_message(
            #     f"{self.name} notices {player.name}.", colors.LIGHT_BLUE)
            # Might initiate dialogue or offer a quest if player gets closer (future).
            pass

        elif self.disposition == Disposition.FRIENDLY:
            # if distance <= greeting_radius:
            # controller.message_log.add_message(
            #     f"{self.name} smiles at {player.name}.", colors.GREEN)
            # Might follow player or offer assistance (future).
            pass

        elif self.disposition == Disposition.ALLY:
            # if distance <= greeting_radius:
            # controller.message_log.add_message(
            # f"{self.name} greets {player.name} warmly.", colors.CYAN)
            # Actively helps in combat, follows, etc. (future).
            pass

        # Default "wait" action if no other action was determined by disposition logic
        # Hostiles might patrol or search instead of just waiting
        if not action_to_perform and self.disposition not in [Disposition.HOSTILE]:
            # controller.message_log.add_message(f"{self.name} waits.", colors.GREY)
            pass

        if action_to_perform:
            controller.execute_action(action_to_perform)
