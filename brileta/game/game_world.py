from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors, config
from brileta.config import PLAYER_BASE_STRENGTH, PLAYER_BASE_TOUGHNESS
from brileta.environment.generators import RoomsAndCorridorsGenerator
from brileta.environment.map import GameMap
from brileta.environment.tile_types import TileTypeID
from brileta.game.actors import NPC, PC, Actor, Character, ItemPile, create_bookcase
from brileta.game.actors.environmental import ContainedFire
from brileta.game.countables import CountableType
from brileta.game.enums import CreatureSize
from brileta.game.item_spawner import ItemSpawner
from brileta.game.items.item_core import Item
from brileta.game.items.item_types import (
    ALARM_CLOCK_TYPE,
    COMBAT_KNIFE_TYPE,
    FOOD_TYPE,
    HUNTING_SHOTGUN_TYPE,
    PISTOL_MAGAZINE_TYPE,
    PISTOL_TYPE,
    REVOLVER_TYPE,
    SHOTGUN_SHELLS_TYPE,
    SLEDGEHAMMER_TYPE,
)
from brileta.game.lights import DynamicLight, GlobalLight, LightSource
from brileta.game.outfit import LEATHER_ARMOR_TYPE
from brileta.types import ActorId, TileCoord, WorldTileCoord, WorldTilePos
from brileta.util import rng
from brileta.util.coordinates import Rect
from brileta.util.spatial import SpatialHashGrid, SpatialIndex
from brileta.view.render.lighting.base import LightingSystem

if TYPE_CHECKING:
    from brileta.environment.generators.buildings.building import Building

_npc_rng = rng.get("world.npc_placement")
_container_rng = rng.get("world.containers")


class GameWorld:
    """
    Represents the complete state of the game world.

    Includes the game map, all actors (player, NPCs, items), their properties, and the
    core game rules that govern how these elements interact. Does not handle input,
    rendering, or high-level application flow. Its primary responsibility is to be
    the single source of truth for the game's state.
    """

    MAX_MAP_REGENERATION_ATTEMPTS = 10

    def __init__(
        self,
        map_width: TileCoord,
        map_height: TileCoord,
        generator_type: str = "dungeon",  # "dungeon" or "settlement"
        seed: int | str | None = None,
    ) -> None:
        self.mouse_tile_location_on_map: WorldTilePos | None = None
        self.item_spawner = ItemSpawner(self)
        self.selected_actor: Actor | None = None

        self.lights: list[LightSource] = []  # All light sources in the world
        self.lighting_system: LightingSystem | None = None

        self.generator_type = generator_type
        self.seed = seed

        # Settlement-specific data (populated by _generate_map for settlements)
        self.buildings: list[Building] = []
        self.streets: list[Rect] = []

        # Attempt map generation with retry if no valid spawn point exists
        for attempt in range(self.MAX_MAP_REGENERATION_ATTEMPTS):
            self._init_actor_storage()
            self.lights = []

            self.game_map, rooms = self._generate_map(map_width, map_height)

            if not rooms:
                raise RuntimeError(
                    "Map generation produced no rooms - check generator parameters"
                )

            self.player = self._create_player()
            self.add_actor(self.player)

            try:
                self._position_player(rooms[0])
                break  # Success - valid spawn point found
            except ValueError:
                # No valid 3x3 spawn area, retry with new map
                if attempt == self.MAX_MAP_REGENERATION_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"Failed to generate map with valid spawn point after "
                        f"{self.MAX_MAP_REGENERATION_ATTEMPTS} attempts"
                    ) from None
                continue

        # Populate NPCs using generator-appropriate method
        if self.generator_type == "settlement":
            self._populate_settlement_npcs()
        else:
            self._populate_npcs(rooms)

        self._place_containers(rooms)

        if not config.IS_TEST_ENVIRONMENT:
            self._add_test_fire(rooms)
            self._add_starting_room_items(rooms[0])

    def add_actor(self, actor: Actor) -> None:
        """Adds an actor to the world and registers it with the spatial index."""
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)
        self._actor_id_registry[actor.actor_id] = actor

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the world and spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            # Actor was not in the list; ignore.
            pass
        # Always attempt to unregister from the id registry.
        self._actor_id_registry.pop(actor.actor_id, None)

    def add_light(self, light: LightSource) -> None:
        """Add a light source to the world and notify the lighting system.

        Args:
            light: The light source to add
        """
        self.lights.append(light)
        if self.lighting_system is not None:
            self.lighting_system.on_light_added(light)

    def remove_light(self, light: LightSource) -> None:
        """Remove a light source from the world and notify the lighting system.

        Args:
            light: The light source to remove
        """
        try:
            self.lights.remove(light)
            if self.lighting_system is not None:
                self.lighting_system.on_light_removed(light)
        except ValueError:
            # Light was not in the list; ignore.
            pass

    def get_global_lights(self) -> list[GlobalLight]:
        """Get all global lights (sun, moon, etc.) in the world.

        Returns:
            A list of all GlobalLight instances
        """
        return [light for light in self.lights if isinstance(light, GlobalLight)]

    def get_static_lights(self) -> list[LightSource]:
        """Get all static point lights in the world.

        Returns:
            A list of all static LightSource instances (excluding global lights)
        """
        return [
            light
            for light in self.lights
            if light.is_static() and not isinstance(light, GlobalLight)
        ]

    def set_region_sky_exposure(self, world_pos: WorldTilePos, exposure: float) -> bool:
        """Debug function to set sky exposure for a region at given position.

        Args:
            world_pos: World tile position to find the region
            exposure: Sky exposure value (0.0 = no sky, 1.0 = full sky)

        Returns:
            True if region was found and modified, False otherwise
        """
        if not self.game_map:
            return False

        region = self.game_map.get_region_at(world_pos)
        if region:
            region.sky_exposure = exposure
            # Invalidate lighting cache since global lighting conditions changed
            if self.lighting_system:
                self.lighting_system.on_global_light_changed()
            # Invalidate appearance caches since region properties changed
            if self.game_map:
                self.game_map.invalidate_appearance_caches()
            return True
        return False

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------

    def _init_actor_storage(self) -> None:
        """Initialize the collections used to track actors."""
        self.actors: list[Actor] = []
        self.actor_spatial_index: SpatialIndex[Actor] = SpatialHashGrid(cell_size=16)
        # Registry for O(1) actor lookup by actor_id.
        # Used by floating text system to track actor positions.
        self._actor_id_registry: dict[ActorId, Actor] = {}

    def _create_player(self) -> PC:
        """Instantiate the player character and starting inventory."""
        player = PC(
            x=0,
            y=0,
            ch="@",
            name="Player",
            color=colors.PLAYER_COLOR,
            game_world=self,
            strength=PLAYER_BASE_STRENGTH,
            toughness=PLAYER_BASE_TOUGHNESS,
            starting_weapon=PISTOL_TYPE.create(),
        )
        self._setup_player_inventory(player)

        return player

    def _setup_player_inventory(self, player: Character) -> None:
        """Equip the player's initial gear and ammunition."""
        player.inventory.equip_to_slot(HUNTING_SHOTGUN_TYPE.create(), 1)
        player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(PISTOL_MAGAZINE_TYPE.create())
        player.inventory.add_to_inventory(SHOTGUN_SHELLS_TYPE.create())
        player.inventory.add_to_inventory(SHOTGUN_SHELLS_TYPE.create())

        player.inventory.add_to_inventory(FOOD_TYPE.create())

        # Equip starting armor
        armor_item = LEATHER_ARMOR_TYPE.create()
        player.inventory.set_starting_outfit(armor_item)

        # Starting money
        player.inventory.add_countable(CountableType.COIN, 50)

    def _generate_map(
        self, map_width: int, map_height: int
    ) -> tuple[GameMap, list[Rect]]:
        """Create the game map and return it along with generated rooms/buildings."""
        generator_type = self.generator_type

        if generator_type == "settlement":
            # Pipeline-based settlement generator
            from brileta.environment.generators.pipeline import create_pipeline

            generator = create_pipeline(
                "settlement",
                map_width,
                map_height,
                seed=self.seed,
            )
            map_data = generator.generate()

            game_map = GameMap(map_width, map_height, map_data)
            game_map.gw = self

            # Store settlement data for NPC placement
            self.buildings = map_data.buildings
            self.streets = map_data.streets

            # For settlements, use building regions as spawn points
            building_regions = [
                r for r in map_data.regions.values() if r.region_type == "building"
            ]
            if building_regions:
                building_rects = [r.bounds[0] for r in building_regions if r.bounds]
            else:
                # Fallback: use exterior region or center of map
                exterior_regions = [
                    r for r in map_data.regions.values() if r.region_type == "exterior"
                ]
                if exterior_regions:
                    building_rects = [r.bounds[0] for r in exterior_regions if r.bounds]
                else:
                    building_rects = [
                        Rect(map_width // 2 - 2, map_height // 2 - 2, 4, 4)
                    ]

            return game_map, building_rects
        # Default: dungeon generator
        generator = RoomsAndCorridorsGenerator(
            map_width,
            map_height,
            max_rooms=config.MAX_NUM_ROOMS,
            min_room_size=config.MIN_ROOM_SIZE,
            max_room_size=config.MAX_ROOM_SIZE,
        )
        map_data = generator.generate()

        game_map = GameMap(map_width, map_height, map_data)
        game_map.gw = self

        room_regions = [r for r in map_data.regions.values() if r.region_type == "room"]
        room_rects = [r.bounds[0] for r in room_regions if r.bounds]

        return game_map, room_rects

    def _position_player(self, room: Rect) -> None:
        """Place the player at a valid spawn point within the room.

        Attempts to find a location with at least 3x3 walkable tiles around it.
        Falls back to the room center if no such location exists.
        """
        spawn_point = self._get_spawn_point(room)
        self.player.teleport(*spawn_point)

    def _get_spawn_point(self, room: Rect) -> WorldTilePos:
        """Find a spawn point with at least 3x3 walkable tiles around it.

        Searches within the room bounds first, then falls back to searching
        the entire map. This ensures the player spawns in a position where
        they can move in any direction.

        Args:
            room: The room rectangle to search within first.

        Returns:
            A (x, y) tuple for the best spawn point found.

        Raises:
            ValueError: If no position with a 3x3 walkable area exists on the map.
        """
        # First try within room bounds
        spawn = self._find_spawn_in_room(room)
        if spawn:
            return spawn

        # Fall back to searching the entire map
        spawn = self._find_spawn_on_map()
        if spawn:
            return spawn

        # No valid spawn point with 3x3 open area exists
        raise ValueError("No valid spawn point with 3x3 walkable area found")

    def _has_3x3_walkable(self, x: int, y: int) -> bool:
        """Check if position (x, y) has a 3x3 area of walkable tiles around it."""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if not (
                    0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height
                ):
                    return False
                if not self.game_map.walkable[nx, ny]:
                    return False
        return True

    def _find_spawn_in_room(self, room: Rect) -> WorldTilePos | None:
        """Search within room bounds for a spawn point with 3x3 open space."""
        center_x, center_y = room.center()

        # Check center first
        if self._has_3x3_walkable(center_x, center_y):
            return (center_x, center_y)

        # Spiral outward from center
        max_radius = max(room.width, room.height) // 2

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check perimeter of this radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    x = center_x + dx
                    y = center_y + dy

                    # Stay within room bounds
                    if not (room.x1 <= x < room.x2 and room.y1 <= y < room.y2):
                        continue

                    if self._has_3x3_walkable(x, y):
                        return (x, y)

        return None

    def _find_spawn_on_map(self) -> WorldTilePos | None:
        """Search the entire map for a spawn point with 3x3 open space."""
        width = self.game_map.width
        height = self.game_map.height
        center_x, center_y = width // 2, height // 2
        max_radius = max(width, height) // 2

        for radius in range(0, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if radius > 0 and abs(dx) != radius and abs(dy) != radius:
                        continue

                    x = center_x + dx
                    y = center_y + dy

                    if (
                        0 <= x < width
                        and 0 <= y < height
                        and self._has_3x3_walkable(x, y)
                    ):
                        return (x, y)

        return None

    def spawn_ground_item(
        self, item: Item, x: WorldTileCoord, y: WorldTileCoord, **kwargs
    ) -> Actor:
        """Spawn an item on the ground with smart placement and consolidation."""
        return self.item_spawner.spawn_item(item, x, y, **kwargs)

    def spawn_ground_items(self, items: list[Item], x: int, y: int) -> Actor:
        """Spawn multiple items efficiently as a single pile."""
        return self.item_spawner.spawn_multiple(items, x, y)

    def on_actor_moved(self, actor: Actor) -> None:
        """Notify the lighting system when an actor moves.

        This method should be called whenever an actor moves. It will:
        1. Update any dynamic lights owned by this actor
        2. Invalidate the shadow caster cache (actors cast shadows)

        Args:
            actor: The actor that moved
        """
        if self.lighting_system is not None:
            # Find and update any lights owned by this actor
            for light in self.lights:
                if isinstance(light, DynamicLight) and light.owner is actor:
                    light.position = (actor.x, actor.y)
                    self.lighting_system.on_light_moved(light)

            # Invalidate shadow caster cache since actor positions affect shadows
            self.lighting_system.on_actor_moved(actor)

    def get_pickable_items_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> list[Item]:
        """Get all pickable items at the specified location.

        Currently, this includes items from dead actors' inventories and their
        equipped weapons.
        """
        items_found: list[Item] = []
        # Check items from dead actors at this location
        for actor in self.actors:
            if (
                actor.x == x
                and actor.y == y
                and isinstance(actor, Character)
                and not actor.health.is_alive()  # Only from dead actors
            ):
                # Add all equipped items from all slots
                ready_slots = actor.inventory.ready_slots
                items_found.extend(item for item in ready_slots if item)

            elif (
                actor.x == x
                and actor.y == y
                and hasattr(actor, "inventory")
                and actor.inventory is not None
                and not isinstance(actor, Character)
            ):
                items_found.extend(
                    item for item in list(actor.inventory) if isinstance(item, Item)
                )
                # Only check ready_slots for CharacterInventory, not ContainerStorage
                ready_slots = getattr(actor.inventory, "ready_slots", None)
                if ready_slots is not None:
                    items_found.extend(item for item in ready_slots if item is not None)

        # Future: Add items directly on the ground if we implement that
        # e.g., items_found.extend(self.game_map.get_items_on_ground(x,y))
        return items_found

    def get_actor_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> Actor | None:
        """Return an actor at the given location using the spatial index.

        Prioritizes returning a blocking actor if multiple actors are present.
        """
        actors_at_point = self.actor_spatial_index.get_at_point(x, y)
        if not actors_at_point:
            return None

        # Prefer blocking actors (e.g., NPCs) so that selecting or targeting
        # chooses solid objects over items or corpses that may occupy the same
        # tile.
        for actor in actors_at_point:
            if actor.blocks_movement:
                return actor

        # If no blocking actor exists, return whichever actor is found first.
        return actors_at_point[0]

    def get_actor_by_id(self, actor_id: ActorId) -> Actor | None:
        """Look up an actor by its ``actor_id`` in O(1) time.

        Uses the internal actor registry for constant-time lookup.
        Used by floating text system to track actor positions.
        Returns None if actor no longer exists.
        """
        return self._actor_id_registry.get(actor_id)

    def has_pickable_items_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> bool:
        """Check if there are any pickable items or countables at the location."""
        if self.get_pickable_items_at_location(x, y):
            return True
        # Check for countables in ItemPiles at this location. Use the spatial
        # index directly since get_actor_at_location prioritizes blocking actors.
        for actor in self.actor_spatial_index.get_at_point(x, y):
            if isinstance(actor, ItemPile) and actor.inventory.countables:
                return True
        return False

    def _populate_npcs(
        self, rooms: list, num_npcs: int = 30, max_attempts_per_npc: int = 10
    ) -> None:
        """Add NPCs to random locations in rooms.

        Rooms must have width and height >= 4 to allow interior placement
        (avoiding walls). Smaller rooms are skipped.
        """
        # Filter to rooms large enough for NPC placement (need width/height >= 4)
        valid_rooms = [r for r in rooms if r.width >= 4 and r.height >= 4]
        if not valid_rooms:
            return  # No rooms large enough for NPCs

        for npc_index in range(num_npcs):
            placed = False

            for _ in range(max_attempts_per_npc):
                # Pick a random room from valid rooms
                room = _npc_rng.choice(valid_rooms)

                # Pick a random tile within the room (avoiding walls)
                npc_x = _npc_rng.randint(room.x1 + 1, room.x2 - 2)
                npc_y = _npc_rng.randint(room.y1 + 1, room.y2 - 2)

                # Check if tile is walkable and free
                if (
                    self.game_map.walkable[npc_x, npc_y]
                    and self.get_actor_at_location(npc_x, npc_y) is None
                ):
                    # Place the NPC - alternate between Trogs and Hackadoos
                    if npc_index % 2 == 0:
                        # Trog: hulking mutant with melee weapon
                        npc = NPC(
                            x=npc_x,
                            y=npc_y,
                            ch="T",
                            name=f"Trog {npc_index + 1}" if npc_index > 0 else "Trog",
                            color=colors.DARK_GREY,
                            game_world=self,
                            blocks_movement=True,
                            weirdness=3,
                            strength=3,
                            toughness=3,
                            intelligence=-3,
                            speed=80,
                            creature_size=CreatureSize.LARGE,
                            starting_weapon=SLEDGEHAMMER_TYPE.create(),
                            can_open_doors=True,
                        )
                    else:
                        # Hackadoo: standard ranged enemy
                        npc = NPC(
                            x=npc_x,
                            y=npc_y,
                            ch="H",
                            name=f"Hackadoo {npc_index + 1}"
                            if npc_index > 0
                            else "Hackadoo",
                            color=colors.DARK_GREY,
                            game_world=self,
                            blocks_movement=True,
                            weirdness=1,
                            intelligence=2,
                            starting_weapon=REVOLVER_TYPE.create(),
                            can_open_doors=True,
                        )
                    self.add_actor(npc)
                    placed = True
                    break

            if not placed:
                # NPC could not be placed after several attempts; skip it
                pass

    def _populate_settlement_npcs(self, max_attempts_per_npc: int = 15) -> None:
        """Populate a settlement with NPCs in buildings, near doors, and on streets.

        Uses the building data from map generation to place NPCs:
        - Indoor NPCs: Placed inside building rooms (scaled to building count)
        - Doorway NPCs: Placed just outside building entrances
        - Street NPCs: Placed on street tiles for outdoor activity

        The total NPC count scales with the number of buildings.
        """
        if not self.buildings:
            return

        num_buildings = len(self.buildings)

        # Scale NPC counts to building count
        # ~2 indoor NPCs per building, 1 near doors, 1-2 on streets
        indoor_npcs = num_buildings * 2
        doorway_npcs = num_buildings
        street_npcs = max(3, num_buildings // 2)

        npc_index = 0

        # 1. Place indoor NPCs in building rooms
        npc_index = self._place_indoor_npcs(
            indoor_npcs, npc_index, max_attempts_per_npc
        )

        # 2. Place NPCs near building doors
        npc_index = self._place_doorway_npcs(
            doorway_npcs, npc_index, max_attempts_per_npc
        )

        # 3. Place NPCs on streets
        self._place_street_npcs(street_npcs, npc_index, max_attempts_per_npc)

    def _place_indoor_npcs(
        self, count: int, start_index: int, max_attempts: int
    ) -> int:
        """Place NPCs inside building rooms.

        Args:
            count: Number of NPCs to place.
            start_index: Starting index for NPC naming.
            max_attempts: Max placement attempts per NPC.

        Returns:
            Next available NPC index.
        """
        npc_index = start_index

        # Collect all room bounds from buildings
        all_rooms: list[Rect] = [
            room.bounds
            for building in self.buildings
            for room in building.rooms
            if room.bounds.width >= 3 and room.bounds.height >= 3
        ]

        if not all_rooms:
            return npc_index

        for _ in range(count):
            placed = False

            for _ in range(max_attempts):
                room = _npc_rng.choice(all_rooms)

                # Pick a position inside the room (not on walls)
                npc_x = _npc_rng.randint(room.x1, room.x2 - 1)
                npc_y = _npc_rng.randint(room.y1, room.y2 - 1)

                if (
                    self.game_map.walkable[npc_x, npc_y]
                    and self.get_actor_at_location(npc_x, npc_y) is None
                ):
                    npc = self._create_settlement_npc(npc_x, npc_y, npc_index, "indoor")
                    self.add_actor(npc)
                    npc_index += 1
                    placed = True
                    break

            if not placed:
                pass  # Skip this NPC

        return npc_index

    def _place_doorway_npcs(
        self, count: int, start_index: int, max_attempts: int
    ) -> int:
        """Place NPCs just outside building doors.

        Args:
            count: Number of NPCs to place.
            start_index: Starting index for NPC naming.
            max_attempts: Max placement attempts per NPC.

        Returns:
            Next available NPC index.
        """
        npc_index = start_index

        # Collect all door positions
        door_positions: list[WorldTilePos] = []
        for building in self.buildings:
            door_positions.extend(building.door_positions)

        if not door_positions:
            return npc_index

        for _ in range(count):
            placed = False

            for _ in range(max_attempts):
                door_x, door_y = _npc_rng.choice(door_positions)

                # Find a position 1-2 tiles from the door (outside the building)
                # Try all 4 directions plus diagonals
                offsets = [
                    (0, -1),
                    (0, 1),
                    (-1, 0),
                    (1, 0),  # Cardinal
                    (0, -2),
                    (0, 2),
                    (-2, 0),
                    (2, 0),  # 2 tiles away
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                    (1, 1),  # Diagonal
                ]
                _npc_rng.shuffle(offsets)

                for dx, dy in offsets:
                    npc_x, npc_y = door_x + dx, door_y + dy

                    # Check bounds
                    if not (
                        0 <= npc_x < self.game_map.width
                        and 0 <= npc_y < self.game_map.height
                    ):
                        continue

                    # Must be walkable, outdoor, and empty
                    if (
                        self.game_map.walkable[npc_x, npc_y]
                        and self._is_outdoor_tile(npc_x, npc_y)
                        and self.get_actor_at_location(npc_x, npc_y) is None
                    ):
                        npc = self._create_settlement_npc(
                            npc_x, npc_y, npc_index, "doorway"
                        )
                        self.add_actor(npc)
                        npc_index += 1
                        placed = True
                        break

                if placed:
                    break

        return npc_index

    def _place_street_npcs(
        self, count: int, start_index: int, max_attempts: int
    ) -> int:
        """Place NPCs on street tiles.

        Args:
            count: Number of NPCs to place.
            start_index: Starting index for NPC naming.
            max_attempts: Max placement attempts per NPC.

        Returns:
            Next available NPC index.
        """
        npc_index = start_index

        # Collect walkable street positions
        street_positions: list[WorldTilePos] = [
            (x, y)
            for street in self.streets
            for x in range(max(0, street.x1), min(self.game_map.width, street.x2))
            for y in range(max(0, street.y1), min(self.game_map.height, street.y2))
            if self.game_map.walkable[x, y]
        ]

        if not street_positions:
            return npc_index

        for _ in range(count):
            for _ in range(max_attempts):
                npc_x, npc_y = _npc_rng.choice(street_positions)

                if self.get_actor_at_location(npc_x, npc_y) is None:
                    npc = self._create_settlement_npc(npc_x, npc_y, npc_index, "street")
                    self.add_actor(npc)
                    npc_index += 1
                    break

        return npc_index

    def _is_outdoor_tile(self, x: int, y: int) -> bool:
        """Check if a tile is an outdoor tile type."""
        outdoor_tiles = {
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
        }
        return self.game_map.tiles[x, y] in outdoor_tiles

    def _create_settlement_npc(
        self, x: int, y: int, index: int, location_type: str
    ) -> NPC:
        """Create an NPC appropriate for a settlement.

        Args:
            x: X position.
            y: Y position.
            index: NPC index for naming.
            location_type: Where the NPC is placed ("indoor", "doorway", "street").

        Returns:
            The created NPC.
        """
        # Alternate between Trogs and Hackadoos
        if index % 2 == 0:
            # Trog: hulking mutant with melee weapon
            return NPC(
                x=x,
                y=y,
                ch="T",
                name=f"Trog {index + 1}" if index > 0 else "Trog",
                color=colors.DARK_GREY,
                game_world=self,
                blocks_movement=True,
                weirdness=3,
                strength=3,
                toughness=3,
                intelligence=-3,
                speed=80,
                creature_size=CreatureSize.LARGE,
                starting_weapon=SLEDGEHAMMER_TYPE.create(),
                can_open_doors=True,
            )
        # Hackadoo: standard ranged enemy
        return NPC(
            x=x,
            y=y,
            ch="H",
            name=f"Hackadoo {index + 1}" if index > 0 else "Hackadoo",
            color=colors.DARK_GREY,
            game_world=self,
            blocks_movement=True,
            weirdness=1,
            intelligence=2,
            starting_weapon=REVOLVER_TYPE.create(),
            can_open_doors=True,
        )

    def _add_test_fire(self, rooms: list) -> None:
        """Add a test fire to demonstrate the fire system."""
        # Pick the first room to add a campfire
        if rooms:
            room = rooms[0]
            # Find a good spot for the fire (offset from center to avoid player)
            fire_x = room.x1 + 2  # Near left wall but not on it
            fire_y = room.y1 + 2  # Near top wall but not on it

            # Make sure the spot is walkable and free
            if (
                self.game_map.walkable[fire_x, fire_y]
                and self.get_actor_at_location(fire_x, fire_y) is None
            ):
                campfire = ContainedFire.create_campfire(fire_x, fire_y, self)
                self.add_actor(campfire)

            # Add both hazards for side-by-side comparison
            self._add_test_acid_pool(room)
            self._add_test_hot_coals(room)

    def _add_test_acid_pool(self, room) -> None:
        """Add a small acid pool to demonstrate hazardous terrain.

        Creates a 2x2 acid pool in the bottom-right corner of the room.
        """
        # Place acid pool in bottom-right corner (away from campfire)
        pool_x = room.x2 - 3  # 3 tiles from right wall
        pool_y = room.y2 - 3  # 3 tiles from bottom wall

        # Create a small 2x2 acid pool
        for dx in range(2):
            for dy in range(2):
                x, y = pool_x + dx, pool_y + dy
                # Only place on walkable tiles that are within bounds
                if (
                    0 <= x < self.game_map.width
                    and 0 <= y < self.game_map.height
                    and self.game_map.walkable[x, y]
                ):
                    self.game_map.tiles[x, y] = TileTypeID.ACID_POOL

    def _add_test_hot_coals(self, room) -> None:
        """Add a small patch of hot coals to demonstrate animated terrain.

        Creates a 2x2 hot coals area in the bottom-right corner of the room.
        """
        from brileta.environment.tile_types import TileTypeID

        # Place hot coals in bottom-left corner (opposite acid pool)
        coals_x = room.x1 + 2  # 2 tiles from left wall
        coals_y = room.y2 - 3  # 3 tiles from bottom wall

        # Create a small 2x2 hot coals area
        for dx in range(2):
            for dy in range(2):
                x, y = coals_x + dx, coals_y + dy
                # Only place on walkable tiles that are within bounds
                if (
                    0 <= x < self.game_map.width
                    and 0 <= y < self.game_map.height
                    and self.game_map.walkable[x, y]
                ):
                    self.game_map.tiles[x, y] = TileTypeID.HOT_COALS

    def _place_containers(
        self, rooms: list, num_containers: int = 5, max_attempts: int = 10
    ) -> None:
        """Place bookcases with random loot in rooms.

        Spawns bookcases throughout the dungeon, each containing random junk items.
        Bookcases use multi-character composition for a rich visual appearance.
        Rooms must have width and height >= 4 for interior placement.

        Args:
            rooms: List of Rect objects representing rooms
            num_containers: Number of bookcases to place
            max_attempts: Maximum attempts to find a valid spot for each bookcase
        """
        from brileta.game.items.junk_item_types import get_random_junk_type

        # Filter to rooms large enough for container placement
        valid_rooms = [r for r in rooms if r.width >= 4 and r.height >= 4]
        if not valid_rooms:
            return  # No rooms large enough for containers

        for _ in range(num_containers):
            placed = False

            for _ in range(max_attempts):
                # Pick a random room (skip first room where player spawns if possible)
                room = (
                    valid_rooms[0]
                    if len(valid_rooms) <= 1
                    else _container_rng.choice(valid_rooms[1:])
                )

                # Pick a random position within the room
                container_x = _container_rng.randint(room.x1 + 1, room.x2 - 2)
                container_y = _container_rng.randint(room.y1 + 1, room.y2 - 2)

                # Check if location is valid
                if (
                    self.game_map.walkable[container_x, container_y]
                    and self.get_actor_at_location(container_x, container_y) is None
                ):
                    # Generate random items for the container
                    num_items = _container_rng.randint(1, 4)
                    items = [get_random_junk_type().create() for _ in range(num_items)]

                    # Create a bookcase
                    container = create_bookcase(
                        x=container_x,
                        y=container_y,
                        items=items,
                        game_world=self,
                    )
                    self.add_actor(container)
                    placed = True
                    break

            if not placed:
                # Container could not be placed; skip it
                pass

    def _add_starting_room_items(self, room: Rect) -> None:
        """Place discoverable items in the starting room.

        Adds some junk items in a corner for the player to find and experiment with.
        """
        # Place in top-right corner (campfire is top-left)
        item_x = room.x2 - 2
        item_y = room.y1 + 2

        if (
            self.game_map.walkable[item_x, item_y]
            and self.get_actor_at_location(item_x, item_y) is None
        ):
            self.spawn_ground_item(ALARM_CLOCK_TYPE.create(), item_x, item_y)

        # Place a coin pile nearby for testing countables
        coin_x = item_x - 1
        coin_y = item_y
        if (
            self.game_map.walkable[coin_x, coin_y]
            and self.get_actor_at_location(coin_x, coin_y) is None
        ):
            self.item_spawner.spawn_ground_countable(
                (coin_x, coin_y), CountableType.COIN, 43
            )
