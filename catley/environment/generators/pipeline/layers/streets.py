"""Street network layer for settlement generation.

This layer creates the street infrastructure that organizes building placement:
- Creates street rectangles based on style (single, cross, grid)
- Defines building zones adjacent to streets
- Carves street tiles into the map
- Stores street data for use by BuildingPlacementLayer
"""

from __future__ import annotations

from catley.environment.generators.pipeline.context import GenerationContext, StreetData
from catley.environment.generators.pipeline.layer import GenerationLayer
from catley.environment.tile_types import TileTypeID
from catley.util.coordinates import Rect

# Street width constant - matches settlement.py
STREET_WIDTH = 3


class StreetNetworkLayer(GenerationLayer):
    """Creates a street network and building zones for settlements.

    The street network provides organization for building placement:
    - Streets are carved as walkable paths through the settlement
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
    ) -> None:
        """Initialize the street network layer.

        Args:
            style: Street layout style ("single", "cross", "grid").
            street_width: Width of streets in tiles.
            grid_spacing: Spacing between streets for grid style.
        """
        self.style = style
        self.street_width = street_width
        self.grid_spacing = grid_spacing

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

        Streets are marked with COBBLESTONE to distinguish
        them from other terrain. The WFC layer can then use this as a
        constraint for natural transitions.

        Args:
            ctx: The generation context.
            street: The street rectangle to carve.
        """
        for x in range(max(0, street.x1), min(ctx.width, street.x2)):
            for y in range(max(0, street.y1), min(ctx.height, street.y2)):
                ctx.tiles[x, y] = TileTypeID.COBBLESTONE

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
