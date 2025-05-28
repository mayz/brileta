from __future__ import annotations

import random

import numpy as np

from catley.world import tile_types


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

        # Initialize the map with walls. We'll carve out rooms later.
        # Store tiles using the types defined in `tile_types` for performance.
        self.tiles = np.full((width, height), fill_value=tile_types.WALL, order="F")

        # Field of view (FOV) state tracking.
        # Which tiles are currently visible.
        self.visible = np.full((width, height), fill_value=False, dtype=bool, order="F")

        # Which tiles have been previously explored
        # (whether they're currently visible or not).
        self.explored = np.full(
            (width, height), fill_value=False, dtype=bool, order="F"
        )

    @property
    def tile_blocked(self) -> np.ndarray:
        """Derive blocked tiles from the `walkable` property (see `tile_types`).

        Returns:
            Boolean array of shape (width, height) where True means the tile
            blocks movement (NOT walkable).
        """
        return ~self.tiles["walkable"]

    @property
    def tile_blocks_sight(self) -> np.ndarray:
        """
        Derive sight-blocking tiles from the `transparent` property (see `tile_types`).

        Returns:
            Boolean array of shape (width, height) where True means the tile
            blocks line of sight (NOT transparent).
        """
        return ~self.tiles["transparent"]

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
        self.tiles[room.x1 + 1 : room.x2, room.y1 + 1 : room.y2] = tile_types.FLOOR

    def _carve_h_tunnel(self, x1: int, x2: int, y: int) -> None:
        h_slice = slice(min(x1, x2), max(x1, x2) + 1)
        self.tiles[h_slice, y] = tile_types.FLOOR

    def _carve_v_tunnel(self, y1: int, y2: int, x: int) -> None:
        v_slice = slice(min(y1, y2), max(y1, y2) + 1)
        self.tiles[x, v_slice] = tile_types.FLOOR
