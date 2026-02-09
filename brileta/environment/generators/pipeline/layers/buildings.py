"""Building placement layer for settlement generation.

This layer places buildings using templates, creating rooms, walls,
doors, and interior spaces. It handles:
- Finding valid positions for buildings (using street zones when available)
- Carving building interiors
- Creating rooms and MapRegions
- Placing doors facing the nearest street
- Recording region connections via doors
"""

from __future__ import annotations

from brileta.environment.generators.buildings import Building, BuildingTemplate, Room
from brileta.environment.generators.buildings.templates import get_default_templates
from brileta.environment.generators.pipeline.context import GenerationContext
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.map import MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect

_rng = rng.get("map.buildings")


class BuildingPlacementLayer(GenerationLayer):
    """Places buildings on the map using templates.

    When street data is available (from StreetNetworkLayer), buildings are
    placed in organized zones with doors facing streets. Otherwise, falls
    back to random placement.

    The layer:
    1. Subdivides zones into building lots (if street data available)
    2. Selects buildings from templates
    3. Carves walls and floors into the tile map
    4. Creates MapRegions for each room
    5. Places doors facing nearest street and records region connections
    """

    def __init__(
        self,
        templates: list[BuildingTemplate] | None = None,
        min_spacing: int = 3,
        max_buildings: int = 10,
        lot_min_size: int = 18,
        lot_max_size: int = 30,
        building_density: float = 0.8,
    ) -> None:
        """Initialize the building placement layer.

        Args:
            templates: List of building templates to use. If None, uses defaults.
            min_spacing: Minimum spacing between buildings.
            max_buildings: Maximum number of buildings to place.
            lot_min_size: Minimum lot dimension for BSP subdivision.
            lot_max_size: Maximum lot dimension for BSP subdivision.
            building_density: Probability of placing a building in a lot (0.0-1.0).
        """
        self.templates = templates if templates is not None else get_default_templates()
        self.min_spacing = min_spacing
        self.max_buildings = max_buildings
        self.lot_min_size = lot_min_size
        self.lot_max_size = lot_max_size
        self.building_density = building_density
        self._next_building_id = 0

    def apply(self, ctx: GenerationContext) -> None:
        """Place buildings on the map.

        If street data is available, uses zone-based placement with BSP
        subdivision. Otherwise falls back to random placement.

        Args:
            ctx: The generation context to modify.
        """
        # Check if we have street data from StreetNetworkLayer
        if ctx.street_data.zones:
            self._place_buildings_in_zones(ctx)
        else:
            self._place_buildings_random(ctx)

    def _place_buildings_in_zones(self, ctx: GenerationContext) -> None:
        """Place buildings using zone-based lot subdivision.

        Subdivides each zone into lots using BSP, then places buildings
        in those lots with doors oriented toward streets.

        Args:
            ctx: The generation context.
        """
        placed_buildings: list[Building] = []
        template_counts: dict[str, int] = {}  # Track placed counts by template name

        # Subdivide zones into lots
        all_lots: list[Rect] = []
        for zone in ctx.street_data.zones:
            lots = self._subdivide_zone_into_lots(ctx, zone)
            all_lots.extend(lots)

        # Shuffle lots for variety
        _rng.shuffle(all_lots)

        # Place buildings in lots
        for lot in all_lots:
            if len(placed_buildings) >= self.max_buildings:
                break

            # Random chance to skip this lot
            if _rng.random() > self.building_density:
                continue

            # Pick a template that fits in the lot (respects weights and limits)
            template = self._pick_template_for_lot(ctx, lot, template_counts)
            if template is None:
                continue

            # Track this template
            template_counts[template.name] = template_counts.get(template.name, 0) + 1

            # Generate building size
            width, height = template.generate_size(_rng)

            # Shrink to fit in lot with margin
            margin = 1
            max_width = lot.width - 2 * margin
            max_height = lot.height - 2 * margin
            width = min(width, max_width)
            height = min(height, max_height)

            if width < 6 or height < 6:
                continue  # Too small for a building

            # Position building within lot (with margin)
            x = lot.x1 + margin + _rng.randint(0, max(0, max_width - width))
            y = lot.y1 + margin + _rng.randint(0, max(0, max_height - height))

            # Create and place the building
            building = self._create_building(ctx, template, (x, y), width, height)
            placed_buildings.append(building)
            ctx.buildings.append(building)

    def _subdivide_zone_into_lots(
        self, ctx: GenerationContext, zone: Rect
    ) -> list[Rect]:
        """Subdivide a zone into building lots using BSP.

        Args:
            ctx: The generation context.
            zone: The zone to subdivide.

        Returns:
            List of lot rectangles.
        """
        lots: list[Rect] = []
        self._bsp_subdivide(zone, lots, ctx, depth=0, max_depth=4)
        return lots

    def _bsp_subdivide(
        self,
        rect: Rect,
        lots: list[Rect],
        ctx: GenerationContext,
        depth: int,
        max_depth: int,
    ) -> None:
        """Binary space partition subdivision of a zone into lots."""
        min_dim = self.lot_min_size
        max_dim = self.lot_max_size

        # Stop if small enough or max depth reached
        if (rect.width <= max_dim and rect.height <= max_dim) or depth >= max_depth:
            if rect.width >= min_dim and rect.height >= min_dim:
                lots.append(rect)
            return

        # Check if we can split
        can_split_h = rect.height >= 2 * min_dim
        can_split_v = rect.width >= 2 * min_dim

        if not can_split_h and not can_split_v:
            if rect.width >= min_dim and rect.height >= min_dim:
                lots.append(rect)
            return

        # Choose split direction
        if can_split_h and can_split_v:
            split_h = rect.height > rect.width or (
                rect.height == rect.width and _rng.random() < 0.5
            )
        else:
            split_h = can_split_h

        if split_h:
            # Split horizontally
            split_range = rect.height - 2 * min_dim
            if split_range <= 0:
                split = rect.height // 2
            else:
                split = min_dim + _rng.randint(0, split_range)

            top = Rect(rect.x1, rect.y1, rect.width, split)
            bottom = Rect(rect.x1, rect.y1 + split, rect.width, rect.height - split)
            self._bsp_subdivide(top, lots, ctx, depth + 1, max_depth)
            self._bsp_subdivide(bottom, lots, ctx, depth + 1, max_depth)
        else:
            # Split vertically
            split_range = rect.width - 2 * min_dim
            if split_range <= 0:
                split = rect.width // 2
            else:
                split = min_dim + _rng.randint(0, split_range)

            left = Rect(rect.x1, rect.y1, split, rect.height)
            right = Rect(rect.x1 + split, rect.y1, rect.width - split, rect.height)
            self._bsp_subdivide(left, lots, ctx, depth + 1, max_depth)
            self._bsp_subdivide(right, lots, ctx, depth + 1, max_depth)

    def _pick_template_for_lot(
        self,
        ctx: GenerationContext,
        lot: Rect,
        template_counts: dict[str, int],
    ) -> BuildingTemplate | None:
        """Pick a template that fits in the given lot using weighted selection.

        Respects max_per_settlement limits and uses template weights for
        probabilistic selection. Houses (high weight) are more common than
        specialty buildings (low weight, max 1 per settlement).

        Args:
            ctx: The generation context.
            lot: The lot to fit a building into.
            template_counts: Dict tracking how many of each template placed so far.

        Returns:
            A suitable BuildingTemplate, or None if none fit or all are at limits.
        """
        margin = 1  # Building margin within lot
        available_width = lot.width - 2 * margin
        available_height = lot.height - 2 * margin

        # Filter templates that:
        # 1. Fit in the lot
        # 2. Haven't exceeded max_per_settlement
        available = []
        for t in self.templates:
            # Check size fit
            if t.min_width > available_width or t.min_height > available_height:
                continue

            # Check max_per_settlement limit
            if t.max_per_settlement is not None:
                current_count = template_counts.get(t.name, 0)
                if current_count >= t.max_per_settlement:
                    continue

            available.append(t)

        if not available:
            return None

        # Weighted random selection
        return self._weighted_choice(ctx, available)

    def _weighted_choice(
        self,
        ctx: GenerationContext,
        templates: list[BuildingTemplate],
    ) -> BuildingTemplate:
        """Select a template using weighted random selection.

        Args:
            ctx: The generation context (for RNG).
            templates: List of available templates.

        Returns:
            A randomly selected template, weighted by template.weight.
        """
        total_weight = sum(t.weight for t in templates)
        if total_weight <= 0:
            return _rng.choice(templates)

        r = _rng.random() * total_weight
        cumulative = 0.0
        for t in templates:
            cumulative += t.weight
            if r <= cumulative:
                return t

        return templates[-1]

    def _get_available_templates(
        self,
        template_counts: dict[str, int],
    ) -> list[BuildingTemplate]:
        """Get templates that haven't exceeded their max_per_settlement limit.

        Args:
            template_counts: Dict tracking how many of each template placed so far.

        Returns:
            List of templates that can still be placed.
        """
        available = []
        for t in self.templates:
            if t.max_per_settlement is not None:
                current_count = template_counts.get(t.name, 0)
                if current_count >= t.max_per_settlement:
                    continue
            available.append(t)
        return available

    def _place_buildings_random(self, ctx: GenerationContext) -> None:
        """Place buildings at random valid positions (fallback mode).

        Used when no street data is available.

        Args:
            ctx: The generation context.
        """
        placed_buildings: list[Building] = []
        template_counts: dict[str, int] = {}

        for _ in range(self.max_buildings):
            # Filter available templates
            available = self._get_available_templates(template_counts)
            if not available:
                break

            template = self._weighted_choice(ctx, available)
            width, height = template.generate_size(_rng)
            position = self._find_random_position(ctx, width, height, placed_buildings)

            if position is None:
                continue

            template_counts[template.name] = template_counts.get(template.name, 0) + 1

            building = self._create_building(ctx, template, position, width, height)
            placed_buildings.append(building)
            ctx.buildings.append(building)

    def _find_random_position(
        self,
        ctx: GenerationContext,
        width: int,
        height: int,
        placed_buildings: list[Building],
    ) -> tuple[int, int] | None:
        """Find a random valid position for a building.

        Args:
            ctx: The generation context.
            width: Width of the building.
            height: Height of the building.
            placed_buildings: Already-placed buildings.

        Returns:
            (x, y) position if found, None otherwise.
        """
        margin = 2
        max_attempts = 50

        for _ in range(max_attempts):
            x = _rng.randint(margin, ctx.width - width - margin)
            y = _rng.randint(margin, ctx.height - height - margin)

            candidate = Rect(x, y, width, height)
            valid = True

            for building in placed_buildings:
                expanded = Rect(
                    building.footprint.x1 - self.min_spacing,
                    building.footprint.y1 - self.min_spacing,
                    building.footprint.width + 2 * self.min_spacing,
                    building.footprint.height + 2 * self.min_spacing,
                )
                if candidate.intersects(expanded):
                    valid = False
                    break

            if valid:
                return x, y

        return None

    def _create_building(
        self,
        ctx: GenerationContext,
        template: BuildingTemplate,
        position: tuple[int, int],
        width: int,
        height: int,
    ) -> Building:
        """Create a building and carve it into the map.

        Args:
            ctx: The generation context.
            template: The building template.
            position: (x, y) position of the top-left corner.
            width: Width of the building.
            height: Height of the building.

        Returns:
            The created Building object.
        """
        building_id = self._next_building_id
        self._next_building_id += 1

        building = template.create_building(building_id, position, width, height)

        # Carve the building into the tile map
        self._carve_building(ctx, building)

        # Create rooms via subdivision
        # Scale room count based on actual interior area to avoid cramming too many
        # rooms into a small building
        interior = building.interior_bounds
        interior_area = interior.width * interior.height
        min_area_per_room = 25  # Each room needs at least ~5x5 of interior space

        template_room_count = template.generate_room_count(_rng)
        max_rooms_by_area = max(1, interior_area // min_area_per_room)
        room_count = min(template_room_count, max_rooms_by_area)

        rooms = self._create_rooms(ctx, building, template, room_count)
        building.rooms.extend(rooms)

        # Place exterior door (connecting to the first room)
        door_pos = self._place_door(ctx, building)
        if door_pos and rooms:
            building.door_positions.append(door_pos)
            # Record connection between interior and exterior
            self._record_door_connection(ctx, door_pos, rooms[0].region_id)

        return building

    def _carve_building(self, ctx: GenerationContext, building: Building) -> None:
        """Carve walls and floor into the tile map.

        Args:
            ctx: The generation context.
            building: The building to carve.
        """
        fp = building.footprint

        # Draw walls around the perimeter
        for x in range(fp.x1, fp.x2):
            if 0 <= x < ctx.width:
                if 0 <= fp.y1 < ctx.height:
                    ctx.tiles[x, fp.y1] = TileTypeID.WALL
                if 0 <= fp.y2 - 1 < ctx.height:
                    ctx.tiles[x, fp.y2 - 1] = TileTypeID.WALL

        for y in range(fp.y1, fp.y2):
            if 0 <= y < ctx.height:
                if 0 <= fp.x1 < ctx.width:
                    ctx.tiles[fp.x1, y] = TileTypeID.WALL
                if 0 <= fp.x2 - 1 < ctx.width:
                    ctx.tiles[fp.x2 - 1, y] = TileTypeID.WALL

        # Fill interior with floor
        for x in range(fp.x1 + 1, fp.x2 - 1):
            for y in range(fp.y1 + 1, fp.y2 - 1):
                if 0 <= x < ctx.width and 0 <= y < ctx.height:
                    ctx.tiles[x, y] = TileTypeID.FLOOR

    def _create_rooms(
        self,
        ctx: GenerationContext,
        building: Building,
        template: BuildingTemplate,
        room_count: int,
    ) -> list[Room]:
        """Create rooms by subdividing the building interior.

        Uses recursive binary subdivision to create the requested number
        of rooms. Each room gets internal walls and doors connecting
        to adjacent rooms.

        Args:
            ctx: The generation context.
            building: The building containing the rooms.
            template: The building template.
            room_count: Number of rooms to create.

        Returns:
            List of created Room objects.
        """
        interior = building.interior_bounds

        # Minimum room size (interior, not including walls)
        min_room_size = 4

        # Subdivide the interior into room bounds
        room_bounds = self._subdivide_space(ctx, interior, room_count, min_room_size)

        # Create Room objects and carve internal walls
        rooms: list[Room] = []
        room_types = list(template.room_types) if template.room_types else ["main"]

        for i, bounds in enumerate(room_bounds):
            room_type = room_types[i % len(room_types)]
            room = self._create_room(ctx, bounds, room_type)
            rooms.append(room)

        # Place internal doors between adjacent rooms
        self._place_internal_doors(ctx, rooms)

        return rooms

    def _subdivide_space(
        self,
        ctx: GenerationContext,
        bounds: Rect,
        target_count: int,
        min_size: int,
    ) -> list[Rect]:
        """Recursively subdivide a space into room-sized regions.

        Uses binary space partition (BSP) to split the space. Prefers
        splitting the longer axis for more natural room proportions.

        Args:
            ctx: The generation context (for RNG).
            bounds: The rectangular space to subdivide.
            target_count: Target number of subdivisions.
            min_size: Minimum dimension for a room.

        Returns:
            List of Rect bounds for each room.
        """
        # Base case: can't subdivide further or reached target
        if target_count <= 1:
            return [bounds]

        # Check if we can split
        can_split_h = bounds.height >= min_size * 2 + 1  # +1 for internal wall
        can_split_v = bounds.width >= min_size * 2 + 1

        if not can_split_h and not can_split_v:
            # Space too small to subdivide
            return [bounds]

        # Choose split direction based on aspect ratio (split longer axis)
        if can_split_h and can_split_v:
            split_horizontal = bounds.height > bounds.width
            if bounds.height == bounds.width:
                split_horizontal = _rng.choice([True, False])
        elif can_split_h:
            split_horizontal = True
        else:
            split_horizontal = False

        # Compute split position with some randomness
        if split_horizontal:
            # Split horizontally (creates top and bottom rooms)
            min_split = bounds.y1 + min_size
            max_split = bounds.y2 - min_size - 1
            if min_split > max_split:
                return [bounds]
            split_pos = _rng.randint(min_split, max_split)

            top = Rect(bounds.x1, bounds.y1, bounds.width, split_pos - bounds.y1)
            bottom = Rect(
                bounds.x1, split_pos + 1, bounds.width, bounds.y2 - split_pos - 1
            )

            # Carve the internal wall
            for x in range(bounds.x1, bounds.x2):
                if 0 <= x < ctx.width and 0 <= split_pos < ctx.height:
                    ctx.tiles[x, split_pos] = TileTypeID.WALL
        else:
            # Split vertically (creates left and right rooms)
            min_split = bounds.x1 + min_size
            max_split = bounds.x2 - min_size - 1
            if min_split > max_split:
                return [bounds]
            split_pos = _rng.randint(min_split, max_split)

            top = Rect(bounds.x1, bounds.y1, split_pos - bounds.x1, bounds.height)
            bottom = Rect(
                split_pos + 1, bounds.y1, bounds.x2 - split_pos - 1, bounds.height
            )

            # Carve the internal wall
            for y in range(bounds.y1, bounds.y2):
                if 0 <= split_pos < ctx.width and 0 <= y < ctx.height:
                    ctx.tiles[split_pos, y] = TileTypeID.WALL

        # Distribute target count between the two halves
        # Larger half gets more rooms
        area_top = top.width * top.height
        area_bottom = bottom.width * bottom.height
        total_area = area_top + area_bottom

        if total_area > 0:
            count_top = max(1, round((target_count - 1) * area_top / total_area))
            count_bottom = max(1, target_count - count_top)
        else:
            count_top = 1
            count_bottom = 1

        # Recurse
        result = []
        result.extend(self._subdivide_space(ctx, top, count_top, min_size))
        result.extend(self._subdivide_space(ctx, bottom, count_bottom, min_size))

        return result

    def _create_room(
        self,
        ctx: GenerationContext,
        bounds: Rect,
        room_type: str,
    ) -> Room:
        """Create a single room from bounds.

        Args:
            ctx: The generation context.
            bounds: The room bounds (walkable area).
            room_type: Semantic type for the room.

        Returns:
            The created Room object.
        """
        region_id = ctx.next_region_id()

        # Create MapRegion for this room
        map_region = MapRegion.create_indoor_region(
            map_region_id=region_id,
            region_type="building",
            bounds=[bounds],
            sky_exposure=0.0,
        )
        ctx.add_region(map_region)

        # Assign tiles to this region
        for x in range(bounds.x1, bounds.x2):
            for y in range(bounds.y1, bounds.y2):
                if 0 <= x < ctx.width and 0 <= y < ctx.height:
                    ctx.tile_to_region_id[x, y] = region_id

        return Room(
            region_id=region_id,
            room_type=room_type,
            bounds=bounds,
        )

    def _place_internal_doors(
        self,
        ctx: GenerationContext,
        rooms: list[Room],
    ) -> None:
        """Place doors between adjacent rooms.

        Finds rooms that share a wall and places a door connecting them.

        Args:
            ctx: The generation context.
            rooms: List of rooms in the building.
        """
        # Track which room pairs already have doors
        connected: set[tuple[int, int]] = set()

        for i, room_a in enumerate(rooms):
            for room_b in rooms[i + 1 :]:
                pair = (
                    min(room_a.region_id, room_b.region_id),
                    max(room_a.region_id, room_b.region_id),
                )
                if pair in connected:
                    continue

                # Find shared wall between rooms
                door_pos = self._find_shared_wall_door_position(
                    ctx, room_a.bounds, room_b.bounds
                )

                if door_pos:
                    x, y = door_pos
                    ctx.tiles[x, y] = TileTypeID.DOOR_CLOSED
                    connected.add(pair)

                    # Record bidirectional connection
                    if (
                        room_a.region_id in ctx.regions
                        and room_b.region_id in ctx.regions
                    ):
                        ctx.regions[room_a.region_id].connections[room_b.region_id] = (
                            door_pos
                        )
                        ctx.regions[room_b.region_id].connections[room_a.region_id] = (
                            door_pos
                        )

    def _find_shared_wall_door_position(
        self,
        ctx: GenerationContext,
        bounds_a: Rect,
        bounds_b: Rect,
    ) -> tuple[int, int] | None:
        """Find a valid door position on the wall between two rooms.

        Args:
            ctx: The generation context.
            bounds_a: Bounds of the first room.
            bounds_b: Bounds of the second room.

        Returns:
            (x, y) position for a door, or None if no shared wall.
        """
        candidates: list[tuple[int, int]] = []

        # Check for horizontal wall (rooms stacked vertically)
        # Room A above Room B: A.y2 == wall_y, B.y1 == wall_y + 1
        if bounds_a.y2 + 1 == bounds_b.y1:
            wall_y = bounds_a.y2
            x_start = max(bounds_a.x1, bounds_b.x1)
            x_end = min(bounds_a.x2, bounds_b.x2)
            if 0 <= wall_y < ctx.height:
                candidates.extend(
                    (x, wall_y)
                    for x in range(x_start, x_end)
                    if 0 <= x < ctx.width and ctx.tiles[x, wall_y] == TileTypeID.WALL
                )

        # Room B above Room A
        if bounds_b.y2 + 1 == bounds_a.y1:
            wall_y = bounds_b.y2
            x_start = max(bounds_a.x1, bounds_b.x1)
            x_end = min(bounds_a.x2, bounds_b.x2)
            if 0 <= wall_y < ctx.height:
                candidates.extend(
                    (x, wall_y)
                    for x in range(x_start, x_end)
                    if 0 <= x < ctx.width and ctx.tiles[x, wall_y] == TileTypeID.WALL
                )

        # Check for vertical wall (rooms side by side)
        # Room A left of Room B: A.x2 == wall_x, B.x1 == wall_x + 1
        if bounds_a.x2 + 1 == bounds_b.x1:
            wall_x = bounds_a.x2
            y_start = max(bounds_a.y1, bounds_b.y1)
            y_end = min(bounds_a.y2, bounds_b.y2)
            if 0 <= wall_x < ctx.width:
                candidates.extend(
                    (wall_x, y)
                    for y in range(y_start, y_end)
                    if 0 <= y < ctx.height and ctx.tiles[wall_x, y] == TileTypeID.WALL
                )

        # Room B left of Room A
        if bounds_b.x2 + 1 == bounds_a.x1:
            wall_x = bounds_b.x2
            y_start = max(bounds_a.y1, bounds_b.y1)
            y_end = min(bounds_a.y2, bounds_b.y2)
            if 0 <= wall_x < ctx.width:
                candidates.extend(
                    (wall_x, y)
                    for y in range(y_start, y_end)
                    if 0 <= y < ctx.height and ctx.tiles[wall_x, y] == TileTypeID.WALL
                )

        if candidates:
            return _rng.choice(candidates)
        return None

    def _has_entry_depth(
        self,
        ctx: GenerationContext,
        x: int,
        y: int,
        direction: str,
        fp_inner_bound: int,
        min_depth: int = 2,
    ) -> bool:
        """Check if a door position has adequate entry depth.

        Ensures there are at least min_depth walkable tiles (floor or door)
        in the entry direction before hitting a wall. This prevents awkward
        1-tile vestibules where you enter and immediately face a perpendicular
        interior wall.

        Args:
            ctx: The generation context.
            x: Door x position.
            y: Door y position.
            direction: Door direction ("N", "S", "E", "W").
            fp_inner_bound: The interior bound in the entry direction (opposite wall).
            min_depth: Minimum walkable tiles required (default 2).

        Returns:
            True if adequate entry depth exists.
        """
        walkable = {TileTypeID.FLOOR, TileTypeID.DOOR_CLOSED, TileTypeID.DOOR_OPEN}

        for d in range(1, min_depth + 1):
            if direction == "N":
                check_y = y + d
                if check_y >= fp_inner_bound:
                    break  # Reached opposite wall, that's fine
                if ctx.tiles[x, check_y] not in walkable:
                    return False
            elif direction == "S":
                check_y = y - d
                if check_y <= fp_inner_bound:
                    break
                if ctx.tiles[x, check_y] not in walkable:
                    return False
            elif direction == "W":
                check_x = x + d
                if check_x >= fp_inner_bound:
                    break
                if ctx.tiles[check_x, y] not in walkable:
                    return False
            elif direction == "E":
                check_x = x - d
                if check_x <= fp_inner_bound:
                    break
                if ctx.tiles[check_x, y] not in walkable:
                    return False

        return True

    def _place_door(
        self,
        ctx: GenerationContext,
        building: Building,
    ) -> tuple[int, int] | None:
        """Place a door on the wall facing the nearest street.

        When street data is available, orients the door toward the nearest
        street. Otherwise falls back to random wall selection.

        Args:
            ctx: The generation context.
            building: The building to add a door to.

        Returns:
            (x, y) position of the door, or None if placement failed.
        """
        fp = building.footprint

        # Determine which wall should have the door
        direction = self._find_nearest_street_direction(ctx, building)

        # Collect door positions on the chosen wall, filtering out positions
        # where the interior tile is blocked by an internal wall OR where
        # there's insufficient entry depth (would create awkward vestibule).
        candidates: list[tuple[int, int]] = []

        if direction == "N":
            # North wall (y = fp.y1), interior is y+1, opposite wall is fp.y2 - 1
            candidates = [
                (x, fp.y1)
                for x in range(fp.x1 + 1, fp.x2 - 1)
                if self._has_entry_depth(ctx, x, fp.y1, "N", fp.y2 - 1)
            ]
        elif direction == "S":
            # South wall (y = fp.y2 - 1), interior is y-1, opposite wall is fp.y1
            candidates = [
                (x, fp.y2 - 1)
                for x in range(fp.x1 + 1, fp.x2 - 1)
                if self._has_entry_depth(ctx, x, fp.y2 - 1, "S", fp.y1)
            ]
        elif direction == "W":
            # West wall (x = fp.x1), interior is x+1, opposite wall is fp.x2 - 1
            candidates = [
                (fp.x1, y)
                for y in range(fp.y1 + 1, fp.y2 - 1)
                if self._has_entry_depth(ctx, fp.x1, y, "W", fp.x2 - 1)
            ]
        elif direction == "E":
            # East wall (x = fp.x2 - 1), interior is x-1, opposite wall is fp.x1
            candidates = [
                (fp.x2 - 1, y)
                for y in range(fp.y1 + 1, fp.y2 - 1)
                if self._has_entry_depth(ctx, fp.x2 - 1, y, "E", fp.x1)
            ]

        if not candidates:
            return None

        # Find the largest contiguous segment of candidates.
        # Internal walls create gaps in the candidate list, so we want to place
        # the door in the center of the largest room's wall section, not at the
        # edge where two rooms meet.
        best_segment = self._find_largest_contiguous_segment(candidates, direction)

        # Pick a position near the center of the best segment
        mid_idx = len(best_segment) // 2
        # Add slight randomness around the center
        offset = _rng.randint(-1, 1) if len(best_segment) > 2 else 0
        idx = max(0, min(len(best_segment) - 1, mid_idx + offset))
        door_pos = best_segment[idx]
        x, y = door_pos

        # Place the door
        if 0 <= x < ctx.width and 0 <= y < ctx.height:
            ctx.tiles[x, y] = TileTypeID.DOOR_CLOSED
            return door_pos

        return None

    def _find_largest_contiguous_segment(
        self,
        candidates: list[tuple[int, int]],
        direction: str,
    ) -> list[tuple[int, int]]:
        """Find the largest contiguous segment of door candidates.

        When internal walls divide a building, candidates on the exterior wall
        are split into segments. This finds the largest segment so the door
        is placed in the center of a room, not at the edge where rooms meet.

        Args:
            candidates: List of (x, y) door candidate positions.
            direction: Door direction ("N", "S", "E", "W").

        Returns:
            The largest contiguous segment of candidates.
        """
        if len(candidates) <= 1:
            return candidates

        # Determine which coordinate varies (x for N/S walls, y for E/W walls)
        is_horizontal = direction in ("N", "S")

        # Sort by the varying coordinate
        if is_horizontal:
            sorted_candidates = sorted(candidates, key=lambda p: p[0])
        else:
            sorted_candidates = sorted(candidates, key=lambda p: p[1])

        # Find contiguous segments (adjacent positions differ by 1)
        segments: list[list[tuple[int, int]]] = []
        current_segment: list[tuple[int, int]] = [sorted_candidates[0]]

        for i in range(1, len(sorted_candidates)):
            prev = sorted_candidates[i - 1]
            curr = sorted_candidates[i]

            # Check if this position is adjacent to the previous
            if is_horizontal:
                is_adjacent = curr[0] == prev[0] + 1
            else:
                is_adjacent = curr[1] == prev[1] + 1

            if is_adjacent:
                current_segment.append(curr)
            else:
                segments.append(current_segment)
                current_segment = [curr]

        segments.append(current_segment)

        # Return the largest segment
        largest = segments[0]
        for segment in segments[1:]:
            if len(segment) > len(largest):
                largest = segment
        return largest

    def _find_nearest_street_direction(
        self, ctx: GenerationContext, building: Building
    ) -> str:
        """Determine which direction faces the nearest street.

        Args:
            ctx: The generation context.
            building: The building to check.

        Returns:
            Direction string: "N", "S", "E", or "W".
        """
        # If no street data, return a random direction
        if not ctx.street_data.streets:
            return _rng.choice(["N", "S", "E", "W"])

        bx, by = building.footprint.center()
        min_dist = float("inf")
        best_direction = "S"

        for street in ctx.street_data.streets:
            sx, sy = street.center()
            is_horizontal = street.width > street.height

            if is_horizontal:
                # Horizontal street - check if building is above or below
                if sy < by:
                    # Street is above building - door should face north
                    dist = building.footprint.y1 - street.y2
                    if dist >= 0 and dist < min_dist:
                        min_dist = dist
                        best_direction = "N"
                else:
                    # Street is below building - door should face south
                    dist = street.y1 - building.footprint.y2
                    if dist >= 0 and dist < min_dist:
                        min_dist = dist
                        best_direction = "S"
            else:
                # Vertical street - check if building is left or right
                if sx < bx:
                    # Street is to the left - door should face west
                    dist = building.footprint.x1 - street.x2
                    if dist >= 0 and dist < min_dist:
                        min_dist = dist
                        best_direction = "W"
                else:
                    # Street is to the right - door should face east
                    dist = street.x1 - building.footprint.x2
                    if dist >= 0 and dist < min_dist:
                        min_dist = dist
                        best_direction = "E"

        return best_direction

    def _record_door_connection(
        self,
        ctx: GenerationContext,
        door_pos: tuple[int, int],
        interior_region_id: int,
    ) -> None:
        """Record the connection between regions via this door.

        Args:
            ctx: The generation context.
            door_pos: (x, y) position of the door.
            interior_region_id: Region ID of the building interior.
        """
        x, y = door_pos

        # Find adjacent regions
        adjacent_ids: set[int] = set()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < ctx.width and 0 <= ny < ctx.height:
                rid = ctx.tile_to_region_id[nx, ny]
                if rid >= 0:
                    adjacent_ids.add(int(rid))

        # Record bidirectional connections
        for rid in adjacent_ids:
            if (
                rid != interior_region_id
                and rid in ctx.regions
                and interior_region_id in ctx.regions
            ):
                ctx.regions[rid].connections[interior_region_id] = door_pos
                ctx.regions[interior_region_id].connections[rid] = door_pos
