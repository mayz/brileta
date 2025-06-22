from __future__ import annotations

import abc
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from catley.environment import tile_types
from catley.environment.map import MapRegion
from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.util.coordinates import TileCoord


@dataclass
class GeneratedMapData:
    """A container for all raw data produced by a map generator."""

    tiles: np.ndarray
    regions: dict[int, MapRegion]
    tile_to_region_id: np.ndarray


class BaseMapGenerator(abc.ABC):
    """Abstract base class for map generation algorithms."""

    def __init__(self, map_width: TileCoord, map_height: TileCoord) -> None:
        self.map_width = map_width
        self.map_height = map_height

    @abc.abstractmethod
    def generate(self) -> GeneratedMapData:
        """Generate the map layout and its structural data."""
        raise NotImplementedError


class RoomsAndCorridorsGenerator(BaseMapGenerator):
    """Generates a map with rooms and connecting corridors."""

    def __init__(
        self,
        map_width: TileCoord,
        map_height: TileCoord,
        max_rooms: int,
        min_room_size: int,
        max_room_size: int,
    ) -> None:
        super().__init__(map_width, map_height)
        self.max_rooms = max_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size

    def _carve_room(self, tiles: np.ndarray, room: Rect) -> None:
        tiles[room.x1 + 1 : room.x2, room.y1 + 1 : room.y2] = (
            tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        )

    def _carve_h_tunnel(self, tiles: np.ndarray, x1: int, x2: int, y: int) -> None:
        h_slice = slice(min(x1, x2), max(x1, x2) + 1)
        tiles[h_slice, y] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]

    def _carve_v_tunnel(self, tiles: np.ndarray, y1: int, y2: int, x: int) -> None:
        v_slice = slice(min(y1, y2), max(y1, y2) + 1)
        tiles[x, v_slice] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]

    def _place_door(self, tiles: np.ndarray, x: int, y: int) -> None:
        if (
            0 <= x < self.map_width
            and 0 <= y < self.map_height
            and tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        ):
            tiles[x, y] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]

    def _place_doors_at_room_entrances(
        self, tiles: np.ndarray, rooms: list[Rect]
    ) -> None:
        for room in rooms:
            for x in range(room.x1 + 1, room.x2):
                y = room.y1
                if (
                    y - 1 >= 0
                    and tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x, y - 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x, y + 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(tiles, x, y)
                y = room.y2
                if (
                    y + 1 < self.map_height
                    and tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x, y + 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x, y - 1] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(tiles, x, y)

            for y in range(room.y1 + 1, room.y2):
                x = room.x1
                if (
                    x - 1 >= 0
                    and tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x - 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x + 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(tiles, x, y)
                x = room.x2
                if (
                    x + 1 < self.map_width
                    and tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x + 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tiles[x - 1, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                ):
                    self._place_door(tiles, x, y)

    def _place_cover_in_room(self, tiles: np.ndarray, room: Rect) -> None:
        if random.random() < 0.4:
            x = random.randint(room.x1 + 1, room.x2 - 2)
            y = random.randint(room.y1 + 1, room.y2 - 2)
            if tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR:  # type: ignore[attr-defined]
                tiles[x, y] = tile_types.TILE_TYPE_ID_BOULDER  # type: ignore[attr-defined]

    def generate(self) -> GeneratedMapData:
        tiles = np.full(
            (self.map_width, self.map_height),
            fill_value=tile_types.TILE_TYPE_ID_WALL,  # type: ignore[attr-defined]
            dtype=np.uint8,
            order="F",
        )

        regions: dict[int, MapRegion] = {}
        tile_to_region_id = np.full(
            (self.map_width, self.map_height), -1, dtype=np.int16, order="F"
        )

        rooms: list[Rect] = []
        next_region_id = 0
        first_room = None

        for _ in range(self.max_rooms):
            w = random.randint(self.min_room_size, self.max_room_size)
            h = random.randint(self.min_room_size, self.max_room_size)

            x = random.randint(0, self.map_width - w - 1)
            y = random.randint(0, self.map_height - h - 1)

            new_room = Rect(x, y, w, h)

            intersects = any(new_room.intersects(other) for other in rooms)
            if intersects:
                continue

            self._carve_room(tiles, new_room)

            room_region = MapRegion(
                id=next_region_id, region_type="room", bounds=[new_room]
            )
            regions[next_region_id] = room_region
            tile_to_region_id[
                new_room.x1 + 1 : new_room.x2, new_room.y1 + 1 : new_room.y2
            ] = next_region_id
            next_region_id += 1

            new_x, new_y = new_room.center()
            if not rooms:
                first_room = new_room
            else:
                prev_x, prev_y = rooms[-1].center()
                if bool(random.getrandbits(1)):
                    self._carve_h_tunnel(tiles, prev_x, new_x, prev_y)
                    self._carve_v_tunnel(tiles, prev_y, new_y, new_x)
                else:
                    self._carve_v_tunnel(tiles, prev_y, new_y, prev_x)
                    self._carve_h_tunnel(tiles, prev_x, new_x, new_y)

            rooms.append(new_room)

        if first_room is None:
            raise ValueError("Need to make at least one room.")

        self._place_doors_at_room_entrances(tiles, rooms)
        for room in rooms:
            self._place_cover_in_room(tiles, room)

        # Identify hallway regions using BFS/flood fill
        visited = np.full(
            (self.map_width, self.map_height), False, dtype=bool, order="F"
        )
        for x in range(self.map_width):
            for y in range(self.map_height):
                if (
                    tiles[x, y] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                    and tile_to_region_id[x, y] == -1
                    and not visited[x, y]
                ):
                    queue = deque([(x, y)])
                    visited[x, y] = True
                    coords = []
                    while queue:
                        cx, cy = queue.popleft()
                        coords.append((cx, cy))
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < self.map_width
                                and 0 <= ny < self.map_height
                                and not visited[nx, ny]
                                and tiles[nx, ny] == tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
                                and tile_to_region_id[nx, ny] == -1
                            ):
                                visited[nx, ny] = True
                                queue.append((nx, ny))

                    if coords:
                        xs, ys = zip(*coords, strict=False)
                        bounds = [
                            Rect(
                                min(xs),
                                min(ys),
                                max(xs) - min(xs) + 1,
                                max(ys) - min(ys) + 1,
                            )
                        ]
                        region = MapRegion(
                            id=next_region_id, region_type="hallway", bounds=bounds
                        )
                        regions[next_region_id] = region
                        for cx, cy in coords:
                            tile_to_region_id[cx, cy] = next_region_id
                        next_region_id += 1

        # Determine connections via doors
        for x in range(self.map_width):
            for y in range(self.map_height):
                if tiles[x, y] == tile_types.TILE_TYPE_ID_DOOR_CLOSED:  # type: ignore[attr-defined]
                    adjacent_ids = set()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                            rid = tile_to_region_id[nx, ny]
                            if rid != -1:
                                adjacent_ids.add(rid)
                    if len(adjacent_ids) == 2:
                        a, b = sorted(adjacent_ids)
                        regions[a].connections[b] = (x, y)
                        regions[b].connections[a] = (x, y)

        return GeneratedMapData(
            tiles=tiles, regions=regions, tile_to_region_id=tile_to_region_id
        )
