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

    def intersects(self, other: "Rect") -> bool:
        return (
            self.x1 <= other.x2
            and self.x2 >= other.x1
            and self.y1 <= other.y2
            and self.y2 >= other.y1
        )

    def __repr__(self) -> str:
        return f"Rect(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


class GameMap:
    """The game map."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        # Initialize the map with walls. We'll carve out rooms later.
        #
        # The map is represented as a NumPy array of tile type IDs.
        # np.uint8 allows for 256 tile types.
        self.tiles = np.full(
            (width, height),
            fill_value=tile_types.TILE_TYPE_ID_WALL,  # type: ignore[unresolved-attribute]
            dtype=np.uint8,
            order="F",
        )

        # Field of view (FOV) state tracking.
        # Which tiles are currently visible.
        self.visible = np.full((width, height), fill_value=False, dtype=bool, order="F")

        # Which tiles have been previously explored
        # (whether they're currently visible or not).
        self.explored = np.full(
            (width, height), fill_value=False, dtype=bool, order="F"
        )

        # Cached property arrays for performance.
        # These are populated on-demand by the respective properties.
        self._walkable_map_cache: np.ndarray | None = None
        self._transparent_map_cache: np.ndarray | None = None
        self._dark_appearance_map_cache: np.ndarray | None = None
        self._light_appearance_map_cache: np.ndarray | None = None

    def invalidate_property_caches(self) -> None:
        """Call this whenever `self.tiles` changes to clear cached property maps."""
        self._walkable_map_cache = None
        self._transparent_map_cache = None
        self._dark_appearance_map_cache = None
        self._light_appearance_map_cache = None

    @property
    def walkable(self) -> np.ndarray:
        """Boolean array of shape (width, height) where True means tile is walkable."""
        if self._walkable_map_cache is None:
            self._walkable_map_cache = tile_types.get_walkable_map(self.tiles)
        return self._walkable_map_cache

    @property
    def transparent(self) -> np.ndarray:
        """Boolean array of shape (width, height) where True means tile is transparent
        (for FOV)."""
        if self._transparent_map_cache is None:
            self._transparent_map_cache = tile_types.get_transparent_map(self.tiles)
        return self._transparent_map_cache

    @property
    def dark_appearance_map(self) -> np.ndarray:
        """A map of 'dark' TileTypeAppearance structs derived from self.tiles."""
        if self._dark_appearance_map_cache is None:
            self._dark_appearance_map_cache = tile_types.get_dark_appearance_map(
                self.tiles
            )
        return self._dark_appearance_map_cache

    @property
    def light_appearance_map(self) -> np.ndarray:
        """A map of 'light' TileTypeAppearance structs derived from self.tiles."""
        if self._light_appearance_map_cache is None:
            self._light_appearance_map_cache = tile_types.get_light_appearance_map(
                self.tiles
            )
        return self._light_appearance_map_cache

    def make_map(
        self,
        max_num_rooms: int,
        min_room_size: int,
        max_room_size: int,
    ) -> list[Rect]:
        rooms = []
        first_room = None

        for _ in range(max_num_rooms):
            w = random.randint(min_room_size, max_room_size)
            h = random.randint(min_room_size, max_room_size)

            x = random.randint(0, self.width - w - 1)
            y = random.randint(0, self.height - h - 1)

            new_room = Rect(x, y, w, h)

            # See if it intersects with any of the rooms we've already made.
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

        # After tunneling, place doors at room entrances.
        self._place_doors_at_room_entrances(rooms)
        for room in rooms:
            self._place_cover_in_room(room)

        # Invalidate caches once after all carving and door placement is complete.
        self.invalidate_property_caches()

        return rooms

    def _carve_room(self, room: Rect) -> None:
        # Assigns TILE_TYPE_ID_FLOOR to the inner area of the room.
        self.tiles[room.x1 + 1 : room.x2, room.y1 + 1 : room.y2] = (
            tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[unresolved-attribute]
        )

    def _carve_h_tunnel(self, x1: int, x2: int, y: int) -> None:
        h_slice = slice(min(x1, x2), max(x1, x2) + 1)
        self.tiles[h_slice, y] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[unresolved-attribute]

    def _carve_v_tunnel(self, y1: int, y2: int, x: int) -> None:
        v_slice = slice(min(y1, y2), max(y1, y2) + 1)
        self.tiles[x, v_slice] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[unresolved-attribute]

    def _place_door(self, x: int, y: int) -> None:
        if (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        ):
            self.tiles[x, y] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]

    def _place_doors_at_room_entrances(self, rooms: list[Rect]) -> None:
        """Place doors at the points where tunnels connect to rooms."""
        for room in rooms:
            # Top and bottom edges
            for x in range(room.x1 + 1, room.x2):
                # Top
                y = room.y1
                if (
                    y - 1 >= 0
                    and self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x, y - 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x, y + 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(x, y)
                # Bottom
                y = room.y2
                if (
                    y + 1 < self.height
                    and self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x, y + 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x, y - 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(x, y)

            # Left and right edges
            for y in range(room.y1 + 1, room.y2):
                # Left
                x = room.x1
                if (
                    x - 1 >= 0
                    and self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x - 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x + 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(x, y)
                # Right
                x = room.x2
                if (
                    x + 1 < self.width
                    and self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x + 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and self.tiles[x - 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(x, y)

    def _place_cover_in_room(self, room: Rect) -> None:
        """Randomly place a boulder providing cover inside the room."""
        if random.random() < 0.4:
            x = random.randint(room.x1 + 1, room.x2 - 2)
            y = random.randint(room.y1 + 1, room.y2 - 2)
            if self.tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR:  # type: ignore[attr-defined]
                self.tiles[x, y] = tile_types.TILE_TYPE_ID_BOULDER  # type: ignore[attr-defined]
