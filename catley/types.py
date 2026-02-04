from __future__ import annotations

from enum import Enum
from typing import Literal, NewType

# =============================================================================
# TILE-BASED COORDINATE SYSTEMS (Always integers)
# =============================================================================


TileCoord = int  # Always integer tile position

# Game world coordinates - absolute positions on the game map
WorldTileCoord = TileCoord  # Example: x=5, y=3
WorldTilePos = tuple[
    WorldTileCoord, WorldTileCoord
]  # Example: (5, 3) = tile 5,3 on map

# Viewport coordinates - relative to the visible area
ViewportTileCoord = TileCoord  # Example: vp_x=0, vp_y=0
ViewportTilePos = tuple[
    ViewportTileCoord, ViewportTileCoord
]  # Example: (0, 0) = top-left of viewport

# Root console coordinates - UI view positioning
RootConsoleTileCoord = TileCoord  # Example: root_x=10, root_y=5
RootConsoleTilePos = tuple[
    RootConsoleTileCoord, RootConsoleTileCoord
]  # Example: (10, 5) = tile 10,5 on UI

# Float-capable view offset for smooth scrolling (camera fractional offset)
ViewOffset = tuple[float, float]

# =============================================================================
# PIXEL-BASED COORDINATE SYSTEMS (Can be float in SDL3)
# =============================================================================

# Raw SDL pixel coordinates (SDL3-ready for float precision)
PixelCoord = int | float  # Example: px_x=123.5
PixelPos = tuple[PixelCoord, PixelCoord]  # Example: (123.5, 456.7)
PixelRect = tuple[int, int, int, int]  # Example: (x1, y1, x2, y2) for hit areas

# =============================================================================
# UTILITY TYPES
# =============================================================================

# Individual tile dimensions in pixels (for working with tilesets)
TileDimensions = tuple[int, int]  # Example: (16, 16) = 16x16 pixel tiles


# =============================================================================
# TIME-RELATED TYPES
# =============================================================================

# Represents the real-world time elapsed between two rendered frames.
# This is variable and used for visual effects and animations that should
# appear smooth regardless of the underlying game logic speed.
DeltaTime = NewType("DeltaTime", float)

# Represents the interpolation factor between two fixed game logic steps.
# It is a value between 0.0 and 1.0, used to smoothly render movement
# and other state changes that occur at a fixed rate.
InterpolationAlpha = NewType("InterpolationAlpha", float)

# Represents the fixed duration of a single game logic update.
# This is a constant value (e.g., 1/60th of a second) and ensures that
# the game simulation is deterministic and independent of the rendering framerate.
FixedTimestep = NewType("FixedTimestep", float)

# =============================================================================
# RENDERING-RELATED TYPES
# =============================================================================

# Represents the opacity/transparency level for visual rendering effects.
# It is a value between 0.0 (fully transparent) and 1.0 (fully opaque),
# used for transparency effects like highlights, overlays, and particle alpha.
Opacity = NewType("Opacity", float)

# =============================================================================
# GAME-RELATED TYPES
# =============================================================================

# Unique identifier for sound definitions (e.g., "fire_ambient", "waterfall_ambient")
SoundId = str

# Random seed for deterministic generation (map generation, etc.)
# Can be an int for numeric seeds or a descriptive string like "burrito1".
RandomSeed = int | str | None

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Literal types for type-checked backend attributes.
AppBackend = Literal["tcod", "glfw"]
GraphicsBackend = Literal["tcod", "moderngl", "wgpu"]
LightingBackend = Literal["moderngl", "wgpu"]


class BackendConfig(Enum):
    """Configuration for which backends to use for app, graphics, and lighting.

    Options:
    - MODERNGL: Fast startup (~350ms). Recommended for now.
    - WGPU: Future-proof (WebGPU/Vulkan/Metal) but slow startup (~1200ms).
    - TCOD_MODERNGL / TCOD_WGPU: Legacy tcod app backend with GPU lighting.
    """

    TCOD_MODERNGL = ("tcod", "tcod", "moderngl")
    TCOD_WGPU = ("tcod", "tcod", "wgpu")
    MODERNGL = ("glfw", "moderngl", "moderngl")
    WGPU = ("glfw", "wgpu", "wgpu")

    app: AppBackend
    graphics: GraphicsBackend
    lighting: LightingBackend

    def __init__(
        self, app: AppBackend, graphics: GraphicsBackend, lighting: LightingBackend
    ) -> None:
        self.app = app
        self.graphics = graphics
        self.lighting = lighting
