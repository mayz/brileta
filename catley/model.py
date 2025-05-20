from __future__ import annotations

import dataclasses
import random

import tcod
from colors import PLAYER_COLOR
from lighting import LightingSystem, LightSource


class Model:
    def __init__(self, map_width: int, map_height: int):
        self.lighting = LightingSystem()
        # Create player with a light source
        player_light = LightSource(
            radius=10,  # Reduced radius
            color=(0.7, 0.5, 0.3),  # Warmer, dimmer torch color
            light_type="dynamic",
            flicker_enabled=True,
            min_brightness=0.5,  # Lower minimum brightness
            max_brightness=0.9   # Lower maximum brightness
        )
        self.player = Entity(
            0, 0,
            ord("@"),
            PLAYER_COLOR,
            model=self,
            light_source=player_light
        )

        self.entities = [self.player]
        self.game_map = GameMap(map_width, map_height)


@dataclasses.dataclass
class Entity:
    """An entity that can exist in the game world."""

    def __init__(
        self,
        x: int,
        y: int,
        ch: int,
        color: tcod.Color,
        model=None,
        light_source: LightSource | None = None
    ):
        self.x = x
        self.y = y
        self.ch = ch  # Ordinal value of the character that represents the entity.
        self.color = color
        self.model = model
        self.light_source = light_source
        if self.light_source and self.model:
            self.light_source.attach(self, self.model.lighting)

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy
        if self.light_source:
            self.light_source.position = (self.x, self.y)


class Tile:
    """DOCME"""

    def __init__(self, blocked: bool, blocks_sight: bool | None = None):
        self.blocked = blocked

        if blocks_sight is None:
            blocks_sight = blocked

        self.blocks_sight = blocks_sight


class Rect:
    def __init__(self, x: int, y: int, w: int, h: int):
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

    def __init__(self, width: int, height: int):
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

    def _carve_room(self, room: Rect):
        for x in range(room.x1 + 1, room.x2):
            for y in range(room.y1 + 1, room.y2):
                self.tiles[x][y].blocked = False
                self.tiles[x][y].blocks_sight = False

    def _carve_h_tunnel(self, x1: int, x2: int, y: int):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.tiles[x][y].blocked = False
            self.tiles[x][y].blocks_sight = False

    def _carve_v_tunnel(self, y1: int, y2: int, x: int):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.tiles[x][y].blocked = False
            self.tiles[x][y].blocks_sight = False
