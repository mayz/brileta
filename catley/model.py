from __future__ import annotations

import random
from enum import Enum, auto
from typing import TYPE_CHECKING

import colors
import items
from conditions import Condition
from items import Item, ItemSize, Weapon
from lighting import LightingSystem, LightSource

if TYPE_CHECKING:
    import tcod
    from actions import Action
    from controller import Controller


class Model:
    """
    Represents the complete state of the game world.

    Includes the game map, all entities (player, NPCs, items), their properties, and the
    core game rules that govern how these elements interact. Does not handle input,
    rendering, or high-level application flow. Its primary responsibility is to be
    the single source of truth for the game's state.
    """

    def __init__(self, map_width: int, map_height: int) -> None:
        self.mouse_tile_location_on_map: tuple[int, int] | None = None
        self.lighting = LightingSystem()
        self.selected_entity: Entity | None = None
        # Create player with a torch light source
        player_light = LightSource.create_torch()
        self.player = CatleyActor(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            model=self,
            light_source=player_light,
            toughness=30,
        )
        self.player.equipped_weapon = items.FISTS

        self.entities = [self.player]
        self.game_map = GameMap(map_width, map_height)

    def update_player_light(self) -> None:
        """Update player light source position"""
        if self.player.light_source:
            self.player.light_source.position = (self.player.x, self.player.y)

    def get_pickable_items_at_location(self, x: int, y: int) -> list[Item]:
        """Get all pickable items at the specified location.

        Currently, this includes items from dead actors' inventories and their
        equipped weapons.
        """
        items_found: list[Item] = []
        # Check items from dead actors at this location
        for entity in self.entities:
            if (
                entity.x == x
                and entity.y == y
                and isinstance(entity, Actor)
                and not entity.is_alive()  # Only from dead actors
            ):
                items_found.extend(entity.inventory)  # Add items from inventory
                if entity.equipped_weapon:
                    items_found.append(entity.equipped_weapon)  # Add equipped weapon
        # Future: Add items directly on the ground if we implement that
        # e.g., items_found.extend(self.game_map.get_items_on_ground(x,y))
        return items_found

    def get_entity_at_location(self, x: int, y: int) -> Actor | None:
        """Return the first entity found at the given location, or None."""
        for entity in self.entities:
            if entity.x == x and entity.y == y:
                return entity
        return None

    def has_pickable_items_at_location(self, x: int, y: int) -> bool:
        """Check if there are any pickable items at the specified location."""
        return bool(self.get_pickable_items_at_location(x, y))


class Entity:
    """An entity that can exist in the game world."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: tcod.Color,
        model: Model = None,
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
        color: tcod.Color,
        max_hp: int,
        max_ap: int,
        model: Model | None,
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

    def take_damage(self, amount: int) -> None:
        """Handle damage to the actor, reducing AP first, then HP.

        Args:
            amount: Amount of damage to take
        """
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

        player = controller.model.player
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

        from actions import AttackAction, MoveAction

        # Determine action based on proximity to player
        dx = player.x - self.x
        dy = player.y - self.y
        distance = abs(dx) + abs(dy)  # Manhattan distance

        action_to_perform: Action | None = None

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
                        0 <= target_x < controller.model.game_map.width
                        and 0 <= target_y < controller.model.game_map.height
                    ):
                        continue
                    if controller.model.game_map.tiles[target_x][target_y].blocked:
                        continue
                    blocking_entity = controller.model.get_entity_at_location(
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
        model: Model | None = None,
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


class Tile:
    """A tile in the game map."""

    def __init__(self, blocked: bool, blocks_sight: bool | None = None) -> None:
        self.blocked = blocked

        if blocks_sight is None:
            blocks_sight = blocked

        self.blocks_sight = blocks_sight


class Rect:
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h

    def center(self) -> tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    def intersects(self, other: Rect) -> bool:
        return (
            self.x1 <= other.x2
            and self.x2 >= other.x1
            and self.y1 <= other.y2
            and self.y2 >= other.y1
        )


class GameMap:
    """The game map."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        tiles = []
        for _ in range(width):
            tiles_line = []
            for _ in range(height):
                tiles_line.append(Tile(blocked=True))
            tiles.append(tiles_line)
        self.tiles = tiles

    def make_map(
        self,
        max_num_rooms: int,
        min_room_size: int,
        max_room_size: int,
        map_width: int,
        map_height: int,
    ) -> list[Rect]:
        rooms = []
        first_room = None

        for _ in range(max_num_rooms):
            w = random.randint(min_room_size, max_room_size)
            h = random.randint(min_room_size, max_room_size)

            x = random.randint(0, map_width - w - 1)
            y = random.randint(0, map_height - h - 1)

            new_room = Rect(x, y, w, h)

            # See if it intesects with any of the rooms we've already made.
            # If it does, toss it out.
            intersects = False
            for other in rooms:
                if new_room.intersects(other):
                    intersects = True
                    break

            if intersects:
                continue

            self._carve_room(new_room)

            new_x, new_y = new_room.center()
            if len(rooms) == 0:
                first_room = new_room
            else:
                # Connect it to the previous room with a tunnel.
                prev_x, prev_y = rooms[-1].center()

                if bool(random.getrandbits(1)):
                    # First move horizontally, then vertically.
                    self._carve_h_tunnel(prev_x, new_x, prev_y)
                    self._carve_v_tunnel(prev_y, new_y, new_x)
                else:
                    # First move vertically, then horizontally.
                    self._carve_v_tunnel(prev_y, new_y, prev_x)
                    self._carve_h_tunnel(prev_x, new_x, new_y)

            rooms.append(new_room)

        if first_room is None:
            raise ValueError("Need to make at least one room.")

        return rooms

    def _carve_room(self, room: Rect) -> None:
        for x in range(room.x1 + 1, room.x2):
            for y in range(room.y1 + 1, room.y2):
                self.tiles[x][y].blocked = False
                self.tiles[x][y].blocks_sight = False

    def _carve_h_tunnel(self, x1: int, x2: int, y: int) -> None:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.tiles[x][y].blocked = False
            self.tiles[x][y].blocks_sight = False

    def _carve_v_tunnel(self, y1: int, y2: int, x: int) -> None:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.tiles[x][y].blocked = False
            self.tiles[x][y].blocks_sight = False
