from __future__ import annotations

from enum import Enum
from typing import Literal, NewType

# =============================================================================
# SPATIAL TYPES
# =============================================================================

type TileCoord = int  # Always integer tile position

# World coordinates - absolute positions on the game map
type WorldTileCoord = TileCoord  # Example: x=5, y=3
type WorldTilePos = tuple[
    WorldTileCoord, WorldTileCoord
]  # Example: (5, 3) = tile 5,3 on map

# Directions - discrete grid steps and continuous vectors
type UnitStep = Literal[-1, 0, 1]
type Direction = tuple[UnitStep, UnitStep]  # Example: (-1, 0) = westward step
type Heading = tuple[float, float]  # Example: (1.0, 0.3) = continuous drift direction

# Viewport coordinates - relative to the visible area
type ViewportTileCoord = TileCoord  # Example: vp_x=0, vp_y=0
type ViewportTilePos = tuple[
    ViewportTileCoord, ViewportTileCoord
]  # Example: (0, 0) = top-left of viewport

# Root console coordinates - UI view positioning
type RootConsoleTileCoord = TileCoord  # Example: root_x=10, root_y=5
type RootConsoleTilePos = tuple[
    RootConsoleTileCoord, RootConsoleTileCoord
]  # Example: (10, 5) = tile 10,5 on UI

# View/camera offset for smooth scrolling (fractional tile offset)
type ViewOffset = tuple[float, float]

# Pixel coordinates
type PixelCoord = int | float  # Example: px_x=123.5
type PixelPos = tuple[PixelCoord, PixelCoord]  # Example: (123.5, 456.7)
type PixelRect = tuple[int, int, int, int]  # Example: (x1, y1, x2, y2) for hit areas

# Dimensions
type TileDimensions = tuple[int, int]  # Example: (16, 16) = 16x16 pixel tiles

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

# Float RGB color in 0.0-1.0 space (GPU-facing color math).
type ColorRGBf = tuple[float, float, float]

# Float RGBA color in 0.0-1.0 space (GPU vertex colors).
type ColorRGBAf = tuple[float, float, float, float]

# UV rectangle for texture coordinates: (u1, v1, u2, v2).
type UVRect = tuple[float, float, float, float]

# =============================================================================
# GAME-RELATED TYPES
# =============================================================================

# Unique identifier for an Actor in the game world. Assigned sequentially
# and used as keys in registries, disposition maps, and goal tracking.
ActorId = NewType("ActorId", int)

# Unique identifier for sound definitions (e.g., "fire_ambient", "waterfall_ambient")
type SoundId = str

# Random seed for deterministic generation (map generation, etc.)
# Can be an int for numeric seeds or a descriptive string like "burrito1".
type RandomSeed = int | str | None

# =============================================================================
# UTILITY TYPES
# =============================================================================

# Generic min/max float range (e.g., speed/lifetime jitter ranges)
type FloatRange = tuple[float, float]

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Literal types for type-checked backend attributes.
# Note: legacy backends (ModernGL, TCOD) were removed in favor of WGPU.
# To recover them, see git tag "pre-moderngl-removal".
type AppBackend = Literal["glfw"]
type GraphicsBackend = Literal["wgpu"]
type LightingBackend = Literal["wgpu"]


class BackendConfig(Enum):
    """Configuration for which backends to use for app, graphics, and lighting.

    WGPU uses native GPU APIs (Metal on macOS, Vulkan on Linux, D3D12 on Windows)
    for future-proof cross-platform rendering.

    Historical note: legacy backends (ModernGL, TCOD) were removed.
    See git tag "pre-moderngl-removal" to recover them if needed.
    """

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
