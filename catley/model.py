from __future__ import annotations

import dataclasses
import random
from abc import abstractmethod
from typing import TYPE_CHECKING

import colors
import items
from lighting import LightingSystem, LightSource

if TYPE_CHECKING:
    import tcod

    from catley.actions import Action
    from catley.controller import Controller

    # (for later, but good to think about)
    from catley.items import Item, Weapon  # Item is used here

    # (for later, e.g., "poisoned", "stunned")
    from catley.status_effects import StatusEffect


class Model:
    def __init__(self, map_width: int, map_height: int) -> None:
        self.lighting = LightingSystem()
        # Create player with a torch light source
        player_light = LightSource.create_torch()
        self.player = WastoidActor(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            model=self,
            light_source=player_light,
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

    def has_pickable_items_at_location(self, x: int, y: int) -> bool:
        """Check if there are any pickable items at the specified location."""
        return bool(self.get_pickable_items_at_location(x, y))


@dataclasses.dataclass
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

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
        # Update the light source position when entity moves
        if self.light_source:
            self.light_source.position = (self.x, self.y)


class Actor(Entity):
    """An entity that can take actions, have health, and participate in combat."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: tcod.Color,
        max_hp: int | 0,
        max_ap: int | 0,
        model: Model | None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        name: str = "",
    ) -> None:
        super().__init__(x, y, ch, color, model, light_source, blocks_movement)
        self.max_hp = max_hp
        self.hp = max_hp
        self.max_ap = max_ap
        self.ap = max_ap
        self.inventory: list[Item] = []
        self.effects: list[StatusEffect] = []
        self.equipped_weapon: Weapon | None = None
        self.name = name

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

    def heal(self, amount: int) -> None:
        """Heal the actor by the specified amount, up to max_hp.

        Args:
            amount: Amount to heal
        """
        self.hp = min(self.max_hp, self.hp + amount)

    def is_alive(self) -> bool:
        """Return True if the actor is alive (HP > 0)."""
        return self.hp > 0

    @abstractmethod
    def get_action(self, controller: Controller) -> Action | None:
        """Get the next action for this actor.

        Args:
            controller: The controller that manages input/decisions

        Returns:
            An Action to perform, or None for no action
        """
        pass


class WastoidActor(Actor):
    """An Actor that follows the Wastoid ruleset with seven core abilities."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        name: str,
        color: tcod.Color,
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
        # Do nothing as max_hp is derived from toughness
        pass


class Tile:
    """DOCME"""

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
    """DOCME"""

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
