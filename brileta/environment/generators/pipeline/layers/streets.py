"""Street network layer for settlement generation.

This layer creates the street infrastructure that organizes building placement:
- Creates street rectangles based on style (single, cross, grid)
- Defines building zones adjacent to streets
- Carves street tiles into the map
- Applies gravel/dirt margins along street edges for organic transitions
- Stores street data for use by BuildingPlacementLayer
"""

from __future__ import annotations

import numpy as np

from brileta import config
from brileta.environment.generators.pipeline.context import (
    GenerationContext,
    StreetData,
)
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect
from brileta.util.noise import FractalType, NoiseGenerator, NoiseType

# Street width constant - matches settlement.py
STREET_WIDTH = 3

_street_margin_rng = rng.get("map.street_margins")


class StreetNetworkLayer(GenerationLayer):
    """Creates a street network and building zones for settlements.

    The street network provides organization for building placement:
    - Streets are carved as walkable paths through the settlement
    - Gravel/dirt margins are placed along street edges for organic transitions
    - Building zones are defined in the areas between streets
    - Door placement can then orient toward the nearest street

    Street styles:
    - "single": One horizontal street through the middle
    - "cross": Cross intersection with horizontal and vertical streets
    - "grid": Regular grid of streets (for larger maps)
    """

    def __init__(
        self,
        style: str = "cross",
        street_width: int = STREET_WIDTH,
        grid_spacing: int = 20,
        margin_max: int = config.STREET_MARGIN_MAX,
        margin_noise_frequency: float = config.STREET_MARGIN_NOISE_FREQUENCY,
        margin_gravel_bias: float = config.STREET_MARGIN_GRAVEL_BIAS,
    ) -> None:
        """Initialize the street network layer.

        Args:
            style: Street layout style ("single", "cross", "grid").
            street_width: Width of streets in tiles.
            grid_spacing: Spacing between streets for grid style.
            margin_max: Maximum margin width in tiles (noise varies 0 to this).
            margin_noise_frequency: Noise frequency for margin width variation.
            margin_gravel_bias: Probability that a margin tile becomes gravel
                vs dirt. Higher values make road edges more gravelly.
        """
        self.style = style
        self.street_width = street_width
        self.grid_spacing = grid_spacing
        self.margin_max = margin_max
        self.margin_noise_frequency = margin_noise_frequency
        self.margin_gravel_bias = margin_gravel_bias

    def apply(self, ctx: GenerationContext) -> None:
        """Create the street network and building zones.

        Args:
            ctx: The generation context to modify.
        """
        # Create streets based on style
        streets = self._create_streets(ctx)

        # Carve streets into the tile map
        for street in streets:
            self._carve_street(ctx, street)

        # Apply gravel/dirt margins along street edges before zone definition,
        # so the margin tiles are part of the landscape when buildings are placed.
        self._apply_street_margins(ctx, streets)

        # Define building zones between streets
        zones = self._define_zones(ctx, streets)

        # Store street data in context for other layers
        ctx.street_data = StreetData(
            streets=streets,
            zones=zones,
            style=self.style,
        )

    def _create_streets(self, ctx: GenerationContext) -> list[Rect]:
        """Create street rectangles based on the configured style.

        Args:
            ctx: The generation context.

        Returns:
            List of Rect objects representing street areas.
        """
        streets: list[Rect] = []
        half_width = self.street_width // 2

        if self.style == "single":
            # Single horizontal street through the middle
            street_y = ctx.height // 2 - half_width
            street = Rect(0, street_y, ctx.width, self.street_width)
            streets.append(street)

        elif self.style == "cross":
            # Cross intersection - horizontal and vertical streets
            h_y = ctx.height // 2 - half_width
            v_x = ctx.width // 2 - half_width
            h_street = Rect(0, h_y, ctx.width, self.street_width)
            v_street = Rect(v_x, 0, self.street_width, ctx.height)
            streets.extend([h_street, v_street])

        elif self.style == "grid":
            # Grid of streets with configurable spacing
            # Horizontal streets
            y = self.grid_spacing
            while y < ctx.height - self.grid_spacing:
                street = Rect(0, y - half_width, ctx.width, self.street_width)
                streets.append(street)
                y += self.grid_spacing

            # Vertical streets
            x = self.grid_spacing
            while x < ctx.width - self.grid_spacing:
                street = Rect(x - half_width, 0, self.street_width, ctx.height)
                streets.append(street)
                x += self.grid_spacing

        return streets

    def _carve_street(self, ctx: GenerationContext, street: Rect) -> None:
        """Carve a street into the tile map.

        Streets overwrite whatever natural terrain exists with COBBLESTONE,
        reflecting the settlement being built on top of the pre-existing
        landscape.

        Args:
            ctx: The generation context.
            street: The street rectangle to carve.
        """
        for x in range(max(0, street.x1), min(ctx.width, street.x2)):
            for y in range(max(0, street.y1), min(ctx.height, street.y2)):
                ctx.tiles[x, y] = TileTypeID.COBBLESTONE

    @staticmethod
    def _chebyshev_distance_to_rect(x: int, y: int, rect: Rect) -> int:
        """Compute Chebyshev (chessboard) distance from a point to a rectangle.

        Returns 0 if the point is inside or on the edge of the rectangle.
        """
        # Horizontal distance: 0 if x is within [x1, x2), else distance to nearest edge
        dx = max(rect.x1 - x, 0, x - (rect.x2 - 1))
        # Vertical distance: 0 if y is within [y1, y2), else distance to nearest edge
        dy = max(rect.y1 - y, 0, y - (rect.y2 - 1))
        return max(dx, dy)

    def _apply_street_margins(
        self, ctx: GenerationContext, streets: list[Rect]
    ) -> None:
        """Place irregular gravel/dirt margins along street edges.

        Noise drives both the margin width and the material choice per tile,
        so the border is organic - gravel spills further in some spots, grass
        reclaims the edge in others, and the gravel/dirt mix varies.

        Both GRASS and DIRT tiles are eligible for conversion.
        Buildings, walls, cobblestone, and other non-terrain tiles are
        left untouched.

        Args:
            ctx: The generation context to modify.
            streets: List of street rectangles.
        """
        if self.margin_max <= 0:
            return

        # Two noise fields with different seeds:
        # - width_noise: controls how far the margin extends at each position
        # - material_noise: controls whether a margin tile becomes gravel or dirt
        width_noise = NoiseGenerator(
            seed=_street_margin_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=self.margin_noise_frequency,
            fractal_type=FractalType.NONE,
            octaves=1,
        )
        material_noise = NoiseGenerator(
            seed=_street_margin_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=self.margin_noise_frequency * 1.5,  # Slightly higher freq
            fractal_type=FractalType.NONE,
            octaves=1,
        )

        # Collect candidate tiles: expand each street by margin_max and
        # gather unique (x, y) positions that are within map bounds.
        candidates: set[tuple[int, int]] = set()
        for street in streets:
            x_lo = max(0, street.x1 - self.margin_max)
            x_hi = min(ctx.width, street.x2 + self.margin_max)
            y_lo = max(0, street.y1 - self.margin_max)
            y_hi = min(ctx.height, street.y2 + self.margin_max)
            for x in range(x_lo, x_hi):
                for y in range(y_lo, y_hi):
                    candidates.add((x, y))

        # Filter candidates: remove tiles inside streets and non-terrain tiles.
        filtered: list[tuple[int, int, int]] = []  # (x, y, dist)
        for x, y in candidates:
            # Skip tiles inside any street (cobblestone stays).
            if any(
                street.x1 <= x < street.x2 and street.y1 <= y < street.y2
                for street in streets
            ):
                continue

            # Only convert natural terrain tiles (grass and dirt). Buildings,
            # walls, cobblestone, and other non-terrain tiles are untouched.
            if ctx.tiles[x, y] not in (TileTypeID.GRASS, TileTypeID.DIRT):
                continue

            # Chebyshev distance to nearest street edge.
            dist = min(
                self._chebyshev_distance_to_rect(x, y, street) for street in streets
            )
            filtered.append((x, y, dist))

        if not filtered:
            return

        # Batch-sample both noise fields for all candidates at once.
        cxs = np.array([c[0] for c in filtered], dtype=np.float32)
        cys = np.array([c[1] for c in filtered], dtype=np.float32)
        dists = np.array([c[2] for c in filtered], dtype=np.float32)

        width_vals = width_noise.sample_array(cxs, cys)
        mat_vals = material_noise.sample_array(cxs, cys)

        # Noise varies the effective margin width from 0 to margin_max.
        # This means some tiles right next to the street can stay as grass
        # (where noise dips low), while others extend 3 tiles out.
        effective_margins = (width_vals + 1.0) * 0.5 * self.margin_max
        in_margin = dists <= effective_margins

        # Material choice: noise biased by distance. Tiles closer to the
        # street are more likely gravel, tiles further out more likely dirt.
        mat_01 = (mat_vals + 1.0) * 0.5
        distance_factors = 1.0 - (dists - 1) / max(self.margin_max - 1, 1)
        gravel_thresholds = self.margin_gravel_bias * distance_factors
        is_gravel = mat_01 < gravel_thresholds

        # Apply tile changes.
        for i in range(len(filtered)):
            if not in_margin[i]:
                continue
            x, y = filtered[i][0], filtered[i][1]
            if is_gravel[i]:
                ctx.tiles[x, y] = TileTypeID.GRAVEL
            else:
                ctx.tiles[x, y] = TileTypeID.DIRT

    def _define_zones(self, ctx: GenerationContext, streets: list[Rect]) -> list[Rect]:
        """Define building zones in areas between streets.

        Zones are rectangular areas where buildings can be placed.
        They are created by subdividing the space between streets.

        Args:
            ctx: The generation context.
            streets: List of street rectangles.

        Returns:
            List of Rect objects representing building zones.
        """
        zones: list[Rect] = []
        margin = 1  # Margin from map edges
        min_zone_size = 8  # Minimum zone dimension

        if self.style == "single":
            # Two zones: above and below the street
            street = streets[0]

            # Zone above street
            if street.y1 > margin + min_zone_size:
                zones.append(
                    Rect(margin, margin, ctx.width - 2 * margin, street.y1 - margin)
                )

            # Zone below street
            if ctx.height - street.y2 > margin + min_zone_size:
                zones.append(
                    Rect(
                        margin,
                        street.y2,
                        ctx.width - 2 * margin,
                        ctx.height - street.y2 - margin,
                    )
                )

        elif self.style == "cross":
            # Four quadrants around the intersection
            h_street = streets[0]
            v_street = streets[1]

            # Top-left quadrant
            if (
                v_street.x1 > margin + min_zone_size
                and h_street.y1 > margin + min_zone_size
            ):
                zones.append(
                    Rect(margin, margin, v_street.x1 - margin, h_street.y1 - margin)
                )

            # Top-right quadrant
            if (
                ctx.width - v_street.x2 > margin + min_zone_size
                and h_street.y1 > margin + min_zone_size
            ):
                zones.append(
                    Rect(
                        v_street.x2,
                        margin,
                        ctx.width - v_street.x2 - margin,
                        h_street.y1 - margin,
                    )
                )

            # Bottom-left quadrant
            if (
                v_street.x1 > margin + min_zone_size
                and ctx.height - h_street.y2 > margin + min_zone_size
            ):
                zones.append(
                    Rect(
                        margin,
                        h_street.y2,
                        v_street.x1 - margin,
                        ctx.height - h_street.y2 - margin,
                    )
                )

            # Bottom-right quadrant
            if (
                ctx.width - v_street.x2 > margin + min_zone_size
                and ctx.height - h_street.y2 > margin + min_zone_size
            ):
                zones.append(
                    Rect(
                        v_street.x2,
                        h_street.y2,
                        ctx.width - v_street.x2 - margin,
                        ctx.height - h_street.y2 - margin,
                    )
                )

        elif self.style == "grid":
            # Zones are the cells between grid streets
            h_streets = [s for s in streets if s.width >= ctx.width - 2]
            v_streets = [s for s in streets if s.height >= ctx.height - 2]
            h_streets.sort(key=lambda s: s.y1)
            v_streets.sort(key=lambda s: s.x1)

            # Build boundary lists
            x_boundaries = [margin] + [s.x1 for s in v_streets] + [ctx.width - margin]
            y_boundaries = [margin] + [s.y1 for s in h_streets] + [ctx.height - margin]

            for i in range(len(x_boundaries) - 1):
                for j in range(len(y_boundaries) - 1):
                    x1, x2 = x_boundaries[i], x_boundaries[i + 1]
                    y1, y2 = y_boundaries[j], y_boundaries[j + 1]

                    # Adjust for street widths
                    if i > 0:
                        x1 = v_streets[i - 1].x2
                    if j > 0:
                        y1 = h_streets[j - 1].y2

                    width = x2 - x1
                    height = y2 - y1

                    if width >= min_zone_size and height >= min_zone_size:
                        zones.append(Rect(x1, y1, width, height))

        return zones
