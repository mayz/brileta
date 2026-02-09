"""Dungeon-style map generation with rooms and corridors."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from brileta.environment.map import MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect

from .base import BaseMapGenerator, GeneratedMapData

if TYPE_CHECKING:
    from brileta.types import TileCoord, WorldTileCoord

_rng = rng.get("map.dungeon")


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
        tiles[room.x1 + 1 : room.x2, room.y1 + 1 : room.y2] = TileTypeID.FLOOR

    def _carve_h_tunnel(self, tiles: np.ndarray, x1: int, x2: int, y: int) -> None:
        h_slice = slice(min(x1, x2), max(x1, x2) + 1)
        tiles[h_slice, y] = TileTypeID.FLOOR

    def _carve_v_tunnel(self, tiles: np.ndarray, y1: int, y2: int, x: int) -> None:
        v_slice = slice(min(y1, y2), max(y1, y2) + 1)
        tiles[x, v_slice] = TileTypeID.FLOOR

    def _place_door(
        self, tiles: np.ndarray, x: WorldTileCoord, y: WorldTileCoord
    ) -> None:
        if (
            0 <= x < self.map_width
            and 0 <= y < self.map_height
            and tiles[x, y] == TileTypeID.FLOOR
        ):
            tiles[x, y] = TileTypeID.DOOR_CLOSED

    def _place_doors_at_room_entrances(
        self, tiles: np.ndarray, rooms: list[Rect]
    ) -> None:
        for room in rooms:
            for x in range(room.x1 + 1, room.x2):
                y = room.y1
                if (
                    y - 1 >= 0
                    and tiles[x, y] == TileTypeID.FLOOR
                    and tiles[x, y - 1] == TileTypeID.FLOOR
                    and tiles[x, y + 1] == TileTypeID.FLOOR
                ):
                    self._place_door(tiles, x, y)
                y = room.y2
                if (
                    y + 1 < self.map_height
                    and tiles[x, y] == TileTypeID.FLOOR
                    and tiles[x, y + 1] == TileTypeID.FLOOR
                    and tiles[x, y - 1] == TileTypeID.FLOOR
                ):
                    self._place_door(tiles, x, y)

            for y in range(room.y1 + 1, room.y2):
                x = room.x1
                if (
                    x - 1 >= 0
                    and tiles[x, y] == TileTypeID.FLOOR
                    and tiles[x - 1, y] == TileTypeID.FLOOR
                    and tiles[x + 1, y] == TileTypeID.FLOOR
                ):
                    self._place_door(tiles, x, y)
                x = room.x2
                if (
                    x + 1 < self.map_width
                    and tiles[x, y] == TileTypeID.FLOOR
                    and tiles[x + 1, y] == TileTypeID.FLOOR
                    and tiles[x - 1, y] == TileTypeID.FLOOR
                ):
                    self._place_door(tiles, x, y)

    def _place_cover_in_room(self, tiles: np.ndarray, room: Rect) -> None:
        if _rng.random() < 0.4:
            x = _rng.randint(room.x1 + 1, room.x2 - 2)
            y = _rng.randint(room.y1 + 1, room.y2 - 2)
            if tiles[x, y] == TileTypeID.FLOOR:
                tiles[x, y] = TileTypeID.BOULDER

    def generate(self) -> GeneratedMapData:
        tiles = np.full(
            (self.map_width, self.map_height),
            fill_value=TileTypeID.WALL,
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
            w = _rng.randint(self.min_room_size, self.max_room_size)
            h = _rng.randint(self.min_room_size, self.max_room_size)

            x = _rng.randint(0, self.map_width - w - 1)
            y = _rng.randint(0, self.map_height - h - 1)

            new_room = Rect(x, y, w, h)

            intersects = any(new_room.intersects(other) for other in rooms)
            if intersects:
                continue

            self._carve_room(tiles, new_room)

            # 20% chance for each room to be outdoor
            if _rng.random() < 0.2:
                room_region = MapRegion.create_outdoor_region(
                    map_region_id=next_region_id, region_type="room", bounds=[new_room]
                )
            else:
                room_region = MapRegion.create_indoor_region(
                    map_region_id=next_region_id, region_type="room", bounds=[new_room]
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
                if bool(_rng.getrandbits(1)):
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
                    tiles[x, y] == TileTypeID.FLOOR
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
                                and tiles[nx, ny] == TileTypeID.FLOOR
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
                if tiles[x, y] == TileTypeID.DOOR_CLOSED:
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
