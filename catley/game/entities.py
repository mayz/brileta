from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from catley import colors

from .conditions import Condition
from .items import Item, ItemSize, Weapon

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.lighting import LightSource
    from catley.world.game_state import GameWorld

    from .actions import GameAction


class Entity:
    """An entity that can exist in the game world."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        model: GameWorld,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.ch = ch  # Character that represents the entity.
        self.color = color
        self.model = model
        self.light_source = light_source
        self.blocks_movement = blocks_movement
        if self.light_source and self.model:
            self.light_source.attach(self, self.model.lighting)
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
    """An entity that can take actions, have health, and participate in combat."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        max_hp: int,
        max_ap: int,
        model: GameWorld | None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        name: str = "<Unnamed Actor>",
        disposition: Disposition = Disposition.WARY,
    ) -> None:
        super().__init__(x, y, ch, color, model, light_source, blocks_movement)
        self.max_hp = max_hp
        self.hp = max_hp
        self.max_ap = max_ap
        self.ap = max_ap
        self.inventory: list[Item | Condition] = []
        self.equipped_weapon: Weapon | None = None
        self.name = name
        self.disposition = disposition

        # To show a visual "flash" when taking damage.
        self._flash_color: colors.Color | None = None
        self._flash_duration_frames: int = 0

    def take_damage(self, amount: int) -> None:
        """Handle damage to the actor, reducing AP first, then HP.

        Args:
            amount: Amount of damage to take
        """
        self._flash_color = colors.RED
        self._flash_duration_frames = 5

        # First reduce AP if any
        if self.ap > 0:
            ap_damage = min(amount, self.ap)
            self.ap -= ap_damage
            amount -= ap_damage

        # Apply remaining damage to HP
        if amount > 0:
            self.hp = max(0, self.hp - amount)

        if not self.is_alive():
            self.ch = "x"
            self.color = colors.DEAD
            self.blocks_movement = False
            # If this actor was selected, deselect it.
            if self.model and self.model.selected_entity == self:
                self.model.selected_entity = None

    def heal(self, amount: int) -> None:
        """Heal the actor by the specified amount, up to max_hp.

        Args:
            amount: Amount to heal
        """
        self.hp = min(self.max_hp, self.hp + amount)

    def is_alive(self) -> bool:
        """Return True if the actor is alive (HP > 0)."""
        return self.hp > 0

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

        if not self.is_alive() or not player.is_alive():
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


class CatleyActor(Actor):
    """An Actor that follows the Wastoid ruleset with seven core abilities."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        name: str,
        color: colors.Color,
        # Wastoid abilities
        weirdness: int = 0,
        agility: int = 0,
        strength: int = 0,
        toughness: int = 0,
        observation: int = 0,
        intelligence: int = 0,
        demeanor: int = 0,
        model: GameWorld | None = None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        disposition: Disposition = Disposition.WARY,
    ) -> None:
        # Initialize base Actor with calculated max_hp
        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            max_hp=toughness + 5,  # Wastoid HP calculation
            max_ap=3,  # Default AP, can be adjusted as needed
            model=model,
            light_source=light_source,
            blocks_movement=blocks_movement,
            name=name,
            disposition=disposition,
        )

        # Core Wastoid abilities
        self.weirdness = weirdness
        self.agility = agility
        self.strength = strength
        self.toughness = toughness
        self.observation = observation
        self.intelligence = intelligence
        self.demeanor = demeanor

        # Wastoid-specific attributes
        self.inventory_slots = strength + 5
        self.tricks: list = []  # Will hold Trick instances later

    @property
    def max_hp(self) -> int:
        """Calculate max HP based on Wastoid rules (toughness + 5)."""
        return self.toughness + 5

    @max_hp.setter
    def max_hp(self, value: int) -> None:
        """Override max_hp setter to prevent direct modification."""
        # Do nothing as max_hp is derived from toughness for WastoidActor
        pass

    def get_used_inventory_space(self) -> int:
        """Calculates the total number of inventory slots currently used,
        considering item sizes and conditions.
        """
        used_space = 0
        has_tiny_items = False
        # Ensure inventory is not None, though it's initialized as an empty list
        inventory_list = self.inventory or []

        for entity_in_slot in inventory_list:
            if isinstance(entity_in_slot, Item):
                item = entity_in_slot
                if item.size == ItemSize.TINY:
                    has_tiny_items = True
                elif item.size == ItemSize.NORMAL:
                    used_space += 1
                elif item.size == ItemSize.BIG:
                    used_space += 2
                elif item.size == ItemSize.HUGE:
                    used_space += 4
            elif isinstance(entity_in_slot, Condition):
                # Conditions take up one slot each
                used_space += 1

        if has_tiny_items:
            used_space += 1  # All tiny items collectively share one slot

        return used_space

    def can_add_to_inventory(self, item_to_add: Item | Condition) -> bool:
        """Checks if an item or condition can be added to the inventory
        based on available space and item sizes.
        """
        current_used_space = self.get_used_inventory_space()
        additional_space_needed = 0

        inventory_list = self.inventory or []  # For checking existing tiny items

        if isinstance(item_to_add, Item):
            item = item_to_add
            if item.size == ItemSize.TINY:
                # If no tiny items exist yet, adding this one will
                # 'create' the tiny slot.
                if not any(
                    isinstance(i, Item) and i.size == ItemSize.TINY
                    for i in inventory_list
                ):
                    additional_space_needed = 1
            elif item.size == ItemSize.NORMAL:
                additional_space_needed = 1
            elif item.size == ItemSize.BIG:
                additional_space_needed = 2
            elif item.size == ItemSize.HUGE:
                additional_space_needed = 4
        elif isinstance(item_to_add, Condition):
            additional_space_needed = 1  # Conditions always take 1 slot

        return (current_used_space + additional_space_needed) <= self.inventory_slots

    def get_inventory_slot_colors(self) -> list[colors.Color]:
        """
        Returns a list of colors representing the filled logical inventory slots.
        Each color corresponds to the item/condition occupying that slot.
        """
        slot_colors: list[colors.Color] = []
        has_processed_tiny_slot = False

        inventory_list = self.inventory or []

        for entity_in_slot in inventory_list:
            if isinstance(entity_in_slot, Item):
                item = entity_in_slot
                # Default color for items in the bar is WHITE
                item_bar_color = colors.WHITE

                if item.size == ItemSize.TINY:
                    if not has_processed_tiny_slot:
                        slot_colors.append(item_bar_color)
                        has_processed_tiny_slot = True
                elif item.size == ItemSize.NORMAL:
                    slot_colors.append(item_bar_color)
                elif item.size == ItemSize.BIG:
                    slot_colors.extend([item_bar_color] * 2)
                elif item.size == ItemSize.HUGE:
                    slot_colors.extend([item_bar_color] * 4)
            elif isinstance(entity_in_slot, Condition):
                slot_colors.append(entity_in_slot.display_color)
        return slot_colors
