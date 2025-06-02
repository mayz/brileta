class CoordinateConverter:
    """Handles pixel-to-tile coordinate conversion for rendering."""

    def __init__(
        self,
        console_width: int,
        console_height: int,
        tile_width: int,
        tile_height: int,
        renderer_width: int,
        renderer_height: int,
    ) -> None:
        """Initialize coordinate converter with console and renderer dimensions."""
        self.console_width = console_width
        self.console_height = console_height

        self.tile_width = tile_width
        self.tile_height = tile_height

        expected_width = console_width * tile_width
        expected_height = console_height * tile_height

        self.scale_x = expected_width / renderer_width
        self.scale_y = expected_height / renderer_height

    def pixel_to_tile(self, pixel_x: int, pixel_y: int) -> tuple[int, int]:
        """Convert pixel coordinates to tile coordinates."""
        # Apply scaling if needed
        scaled_x = pixel_x * self.scale_x
        scaled_y = pixel_y * self.scale_y

        # Convert to tile coordinates
        tile_x = int(scaled_x // self.tile_width)
        tile_y = int(scaled_y // self.tile_height)

        # Clamp to valid range
        tile_x = max(0, min(tile_x, self.console_width - 1))
        tile_y = max(0, min(tile_y, self.console_height - 1))

        return tile_x, tile_y
