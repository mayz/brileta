"""Building templates for procedural building generation.

Templates define the constraints and properties for generating buildings.
Each template specifies size ranges, room counts, and building-specific
configuration.

Building variety is controlled via:
- weight: Higher weight = more likely to be picked (houses are common)
- max_per_settlement: Caps unique buildings (only 1 tavern, 1 blacksmith, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass

from brileta import config
from brileta.types import TileDimensions, WorldTilePos, saturate
from brileta.util.coordinates import Rect
from brileta.util.rng import RNG

from .building import (
    CHIMNEY_MAX_PROJECTED_HEIGHT,
    CHIMNEY_MAX_SHADOW_HEIGHT,
    CHIMNEY_MIN_PROJECTED_HEIGHT,
    CHIMNEY_MIN_SHADOW_HEIGHT,
    Building,
    RoofProfile,
    RoofStyle,
)


@dataclass
class BuildingTemplate:
    """Template defining constraints for building generation.

    Templates specify the size ranges, room configuration, and other
    properties for generating a specific type of building.

    Attributes:
        name: Human-readable name for this template.
        building_type: The semantic type to assign to generated buildings.
        min_width: Minimum building width (including walls).
        max_width: Maximum building width (including walls).
        min_height: Minimum building height (including walls).
        max_height: Maximum building height (including walls).
        room_count_min: Minimum number of rooms.
        room_count_max: Maximum number of rooms.
        door_count: Number of exterior doors (default 1).
        room_types: List of room type names to use for rooms.
        weight: Selection weight (higher = more common). Default 1.0.
        max_per_settlement: Max instances per settlement. None = unlimited.
        roof_style: Render-time roof style ("thatch", "shingle", or "tin").
        roof_profile_family: Allowed roof profile variants for this archetype.
        low_slope_flat_section_range: Quantized range for the central flat band
            ratio when a low-slope roof profile is selected.
        low_slope_flat_section_buckets: Number of deterministic buckets used when
            sampling low-slope flat-section ratios.
        floor_count: Number of floors (future multi-story support).
        has_chimney: Whether buildings of this type get a chimney. The
            position is randomized within the interior at creation time.
        l_shape_weight: Relative weight for L-shape when a compound shape is
            chosen. Templates with both weights at 0 never get a wing.
        t_shape_weight: Relative weight for T-shape when a compound shape is
            chosen. Templates with both weights at 0 never get a wing.
        wing_size_ratio: Min/max shared-wall fraction used for wing length.
        condition_range: (min, max) range for the building's weathering
            condition. Sampled uniformly per building during generation.
            0.0 = pristine, 1.0 = dilapidated.
    """

    name: str
    building_type: str
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    room_count_min: int = 1
    room_count_max: int = 1
    door_count: int = 1
    room_types: tuple[str, ...] = ("main",)
    weight: float = 1.0
    max_per_settlement: int | None = None
    roof_style: RoofStyle = "thatch"
    roof_profile_family: tuple[RoofProfile, ...] = ("gable",)
    low_slope_flat_section_range: tuple[float, float] = (0.45, 0.75)
    low_slope_flat_section_buckets: int = 4
    floor_count: int = 1
    has_chimney: bool = False
    l_shape_weight: float = 0.0
    t_shape_weight: float = 0.0
    wing_size_ratio: tuple[float, float] = (0.4, 0.6)
    condition_range: tuple[float, float] = (0.0, 0.5)

    def generate_size(self, rng: RNG) -> TileDimensions:
        """Generate random dimensions within the template's constraints.

        Args:
            rng: Random number generator.

        Returns:
            A tuple of (width, height) for the building.
        """
        width = rng.randint(self.min_width, self.max_width)
        height = rng.randint(self.min_height, self.max_height)
        return width, height

    def generate_room_count(self, rng: RNG) -> int:
        """Generate random room count within the template's constraints.

        Args:
            rng: Random number generator.

        Returns:
            Number of rooms to generate.
        """
        return rng.randint(self.room_count_min, self.room_count_max)

    def create_building(
        self,
        building_id: int,
        position: WorldTilePos,
        width: int,
        height: int,
        rng: RNG | None = None,
    ) -> Building:
        """Create a Building object with the given position and size.

        This creates the Building shell - rooms and doors are added later
        by the BuildingPlacementLayer.

        Args:
            building_id: Unique ID for this building.
            position: (x, y) position of the top-left corner.
            width: Width of the building.
            height: Height of the building.
            rng: Random number generator for chimney placement and roof-profile
                variant selection. Required for full procedural variation.

        Returns:
            A new Building object with the specified footprint.
        """
        x, y = position
        footprint = Rect(x, y, width, height)
        wing = self._generate_wing_rect(footprint, rng)

        # Randomize chimney position within the interior, staying at least
        # 2 tiles from each edge (1 wall + 1 margin) so the chimney sits
        # comfortably on the roof surface away from eaves.
        chimney_offset = None
        chimney_projected_height = CHIMNEY_MIN_PROJECTED_HEIGHT
        chimney_shadow_height = CHIMNEY_MIN_SHADOW_HEIGHT
        if self.has_chimney and rng is not None:
            margin = 2
            chimney_offset = (
                rng.randint(margin, width - margin - 1),
                rng.randint(margin, height - margin - 1),
            )
            # Coupled height factor: a single random value drives both the
            # visible projected height and the physical shadow height so
            # taller-looking chimneys cast proportionally longer shadows.
            height_factor = rng.random()
            chimney_projected_height = CHIMNEY_MIN_PROJECTED_HEIGHT + height_factor * (
                CHIMNEY_MAX_PROJECTED_HEIGHT - CHIMNEY_MIN_PROJECTED_HEIGHT
            )
            chimney_shadow_height = CHIMNEY_MIN_SHADOW_HEIGHT + height_factor * (
                CHIMNEY_MAX_SHADOW_HEIGHT - CHIMNEY_MIN_SHADOW_HEIGHT
            )

        # Roof profile selection is constrained by archetype (template) and
        # sampled deterministically from the generation RNG.
        profiles = self.roof_profile_family or ("gable",)
        if rng is None or len(profiles) == 1:
            roof_profile: RoofProfile = profiles[0]
        else:
            roof_profile = profiles[rng.randint(0, len(profiles) - 1)]

        flat_section_ratio = 0.0
        if roof_profile == "low_slope":
            min_ratio, max_ratio = self.low_slope_flat_section_range
            min_ratio = saturate(float(min_ratio))
            max_ratio = max(min_ratio, min(1.0, float(max_ratio)))
            bucket_count = max(1, int(self.low_slope_flat_section_buckets))
            if rng is None or bucket_count == 1:
                flat_section_ratio = (min_ratio + max_ratio) * 0.5
            else:
                bucket = rng.randint(0, bucket_count - 1)
                t = bucket / float(bucket_count - 1)
                flat_section_ratio = min_ratio + (max_ratio - min_ratio) * t
        elif roof_profile == "flat":
            flat_section_ratio = 1.0

        # Building condition (weathering/degradation) sampled uniformly
        # within the template's configured range.
        min_cond, max_cond = self.condition_range
        if rng is None or min_cond >= max_cond:
            condition = (min_cond + max_cond) * 0.5
        else:
            condition = rng.uniform(min_cond, max_cond)

        return Building(
            id=building_id,
            building_type=self.building_type,
            footprint=footprint,
            wing=wing,
            roof_style=self.roof_style,
            roof_profile=roof_profile,
            flat_section_ratio=flat_section_ratio,
            floor_count=self.floor_count,
            chimney_offset=chimney_offset,
            chimney_projected_height=chimney_projected_height,
            chimney_shadow_height=chimney_shadow_height,
            condition=condition,
        )

    def _generate_wing_rect(self, footprint: Rect, rng: RNG | None) -> Rect | None:
        """Optionally generate an L-shape or T-shape wing rectangle."""
        if rng is None:
            return None

        lw = max(0.0, float(self.l_shape_weight))
        tw = max(0.0, float(self.t_shape_weight))
        total_weight = lw + tw
        if total_weight <= 0.0:
            return None

        # Global compound-shape probability gate.
        if rng.random() >= float(config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE):
            return None

        # Pick L vs T using the template's relative weights.
        shape_kind = "l" if rng.random() < (lw / total_weight) else "t"

        min_ratio, max_ratio = self.wing_size_ratio
        min_ratio = saturate(float(min_ratio))
        max_ratio = max(min_ratio, min(1.0, float(max_ratio)))

        # Wings need full wall-interior-wall thickness (>=5) in both dimensions.
        min_wing_dim = 5
        if footprint.width < min_wing_dim or footprint.height < min_wing_dim:
            return None

        side = rng.choice(("north", "south", "west", "east"))
        shared_dim = footprint.width if side in ("north", "south") else footprint.height
        if shared_dim < min_wing_dim:
            return None

        shared_ratio = rng.uniform(min_ratio, max_ratio)
        shared_len = max(min_wing_dim, round(shared_dim * shared_ratio))
        shared_len = min(shared_len, shared_dim)
        if shared_len < min_wing_dim:
            return None

        perpendicular_dim = (
            footprint.height if side in ("north", "south") else footprint.width
        )
        wing_depth_ratio = rng.uniform(0.45, 0.65)
        wing_depth = max(min_wing_dim, round(perpendicular_dim * wing_depth_ratio))
        if wing_depth < min_wing_dim:
            return None

        # L-shapes attach at one end of the chosen side; T-shapes center.
        if side in ("north", "south"):
            if shape_kind == "t":
                shared_start = footprint.x1 + (shared_dim - shared_len) // 2
            else:
                shared_start = (
                    footprint.x1
                    if rng.choice((True, False))
                    else footprint.x2 - shared_len
                )
        else:
            if shape_kind == "t":
                shared_start = footprint.y1 + (shared_dim - shared_len) // 2
            else:
                shared_start = (
                    footprint.y1
                    if rng.choice((True, False))
                    else footprint.y2 - shared_len
                )

        if side == "north":
            return Rect.from_bounds(
                shared_start,
                footprint.y1 - wing_depth,
                shared_start + shared_len,
                footprint.y1 + 1,
            )
        if side == "south":
            return Rect.from_bounds(
                shared_start,
                footprint.y2 - 1,
                shared_start + shared_len,
                footprint.y2 + wing_depth,
            )
        if side == "west":
            return Rect.from_bounds(
                footprint.x1 - wing_depth,
                shared_start,
                footprint.x1 + 1,
                shared_start + shared_len,
            )

        return Rect.from_bounds(
            footprint.x2 - 1,
            shared_start,
            footprint.x2 + wing_depth,
            shared_start + shared_len,
        )


# =============================================================================
# Residential Buildings
# =============================================================================

SMALL_HOUSE_TEMPLATE = BuildingTemplate(
    name="small_house",
    building_type="house",
    min_width=12,
    max_width=16,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("living_room", "bedroom", "kitchen"),
    weight=2.0,  # Common - many small houses in a settlement
    roof_style="thatch",
    has_chimney=True,
    condition_range=(0.0, 0.65),  # Wide range - some pristine, some worn
)

MEDIUM_HOUSE_TEMPLATE = BuildingTemplate(
    name="medium_house",
    building_type="house",
    min_width=18,
    max_width=22,
    min_height=14,
    max_height=18,
    room_count_min=2,
    room_count_max=4,
    room_types=("living_room", "bedroom", "kitchen", "storage"),
    weight=2.0,  # Less common than small houses
    roof_style="thatch",
    has_chimney=True,
    l_shape_weight=0.7,
    t_shape_weight=0.3,
    condition_range=(0.0, 0.55),  # Some pristine, slightly better than small houses
)

# =============================================================================
# Commercial Buildings
# =============================================================================

GENERAL_STORE_TEMPLATE = BuildingTemplate(
    name="general_store",
    building_type="general_store",
    min_width=14,
    max_width=18,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("shop_floor", "storage", "office"),
    weight=1.0,
    max_per_settlement=1,  # Only one general store per town
    roof_style="shingle",
    roof_profile_family=("gable", "low_slope"),
    condition_range=(0.0, 0.45),  # Commercial - kept presentable, some pristine
)

BUTCHER_TEMPLATE = BuildingTemplate(
    name="butcher",
    building_type="butcher",
    min_width=10,
    max_width=14,
    min_height=8,
    max_height=12,
    room_count_min=2,
    room_count_max=2,
    room_types=("shop_floor", "cold_storage"),
    weight=0.8,
    max_per_settlement=1,
    roof_style="shingle",
    roof_profile_family=("gable", "low_slope"),
    condition_range=(0.0, 0.5),  # Commercial - moderate upkeep
)

BLACKSMITH_TEMPLATE = BuildingTemplate(
    name="blacksmith",
    building_type="blacksmith",
    min_width=14,
    max_width=18,
    min_height=12,
    max_height=16,
    room_count_min=2,
    room_count_max=3,
    room_types=("forge", "workshop", "storage"),
    weight=0.8,
    max_per_settlement=1,
    roof_style="tin",
    roof_profile_family=("gable", "low_slope"),
    has_chimney=True,
    l_shape_weight=0.7,
    t_shape_weight=0.3,
    condition_range=(0.05, 0.65),  # Industrial - heat and soot, some newer
)

# =============================================================================
# Public Buildings
# =============================================================================

TAVERN_TEMPLATE = BuildingTemplate(
    name="tavern",
    building_type="tavern",
    min_width=22,
    max_width=28,
    min_height=18,
    max_height=22,
    room_count_min=4,
    room_count_max=6,
    room_types=("main_hall", "kitchen", "storage", "bedroom", "cellar"),
    weight=1.0,
    max_per_settlement=1,  # One tavern per settlement
    roof_style="shingle",
    has_chimney=True,
    l_shape_weight=0.7,
    t_shape_weight=0.3,
    condition_range=(0.0, 0.4),  # Well-maintained public building
)

INN_TEMPLATE = BuildingTemplate(
    name="inn",
    building_type="inn",
    min_width=20,
    max_width=26,
    min_height=16,
    max_height=20,
    room_count_min=4,
    room_count_max=6,
    room_types=("common_room", "kitchen", "guest_room", "guest_room", "storage"),
    weight=0.8,
    max_per_settlement=1,  # One inn per settlement
    roof_style="shingle",
    has_chimney=True,
    l_shape_weight=0.7,
    t_shape_weight=0.3,
    condition_range=(0.0, 0.4),  # Well-maintained public building
)

LIBRARY_TEMPLATE = BuildingTemplate(
    name="library",
    building_type="library",
    min_width=16,
    max_width=22,
    min_height=14,
    max_height=18,
    room_count_min=2,
    room_count_max=4,
    room_types=("reading_room", "stacks", "study", "archive"),
    weight=0.5,  # Rare - not every town has a library
    max_per_settlement=1,
    roof_style="shingle",
    roof_profile_family=("gable", "low_slope"),
    condition_range=(0.0, 0.3),  # Best-maintained building in the settlement
)

# =============================================================================
# Small Commercial (can have multiples)
# =============================================================================

SHOP_TEMPLATE = BuildingTemplate(
    name="shop",
    building_type="shop",
    min_width=12,
    max_width=16,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("shop_floor", "back_room", "storage"),
    weight=1.5,  # Generic shops are fairly common
    max_per_settlement=3,  # Can have a few different shops
    roof_style="shingle",
    roof_profile_family=("gable", "low_slope"),
    condition_range=(0.0, 0.45),  # Commercial - kept reasonably presentable
)

WAREHOUSE_TEMPLATE = BuildingTemplate(
    name="warehouse",
    building_type="warehouse",
    min_width=16,
    max_width=24,
    min_height=12,
    max_height=18,
    room_count_min=2,
    room_count_max=3,
    room_types=("storage", "loading_area", "office"),
    weight=0.6,
    max_per_settlement=2,
    roof_style="tin",
    roof_profile_family=("low_slope", "flat"),
    condition_range=(0.0, 0.6),  # Utilitarian - function over appearance
)


def get_default_templates() -> list[BuildingTemplate]:
    """Get the default set of building templates.

    Returns:
        A list of BuildingTemplate objects for common building types.
    """
    return [
        # Residential (common)
        SMALL_HOUSE_TEMPLATE,
        MEDIUM_HOUSE_TEMPLATE,
        # Commercial (unique)
        GENERAL_STORE_TEMPLATE,
        BUTCHER_TEMPLATE,
        BLACKSMITH_TEMPLATE,
        # Public (unique)
        TAVERN_TEMPLATE,
        INN_TEMPLATE,
        LIBRARY_TEMPLATE,
        # Small commercial (can have multiples)
        SHOP_TEMPLATE,
        WAREHOUSE_TEMPLATE,
    ]
