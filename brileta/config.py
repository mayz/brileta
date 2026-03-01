"""
Configuration constants.

Centralizes all magic numbers and configuration values used throughout the codebase.
Organized by functional area for easy maintenance.
"""

import sys
from pathlib import Path
from typing import NamedTuple

from brileta.game.enums import GeneratorType
from brileta.types import BackendConfig, RandomSeed

# =============================================================================
# GENERAL
# =============================================================================

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

# Master seed for all randomness (map generation, gameplay, etc.)
# Set to None for non-deterministic behavior.
RANDOM_SEED: RandomSeed = None

# Test environment detection
IS_TEST_ENVIRONMENT = "pytest" in sys.modules

# =============================================================================
# DISPLAY & RENDERING
# =============================================================================

# Main window
WINDOW_TITLE = "Brileta"
INITIAL_WINDOW_WIDTH_TILES: int = 80
INITIAL_WINDOW_HEIGHT_TILES: int = 50
# Minimum resizable window dimensions in physical pixels.
WINDOW_MIN_WIDTH_PX: int = 640
WINDOW_MIN_HEIGHT_PX: int = 480
# Allowed window aspect-ratio range while resizing.
# Keep this as a range (not a fixed ratio) so players can still choose
# reasonably wide/tall layouts without extreme distortion cases.
WINDOW_MIN_ASPECT_RATIO: float = 1.2
WINDOW_MAX_ASPECT_RATIO: float = 2.4

# User zoom multiplier for tile size.
# Final display tile size is derived as:
#     display_tile = native_tile_from_tileset * content_scale * TILE_ZOOM
# where content_scale is auto-detected from framebuffer/window dimensions.
TILE_ZOOM: float = 1.0

# Discrete runtime zoom levels. ExploreMode scroll input steps through these.
ZOOM_STOPS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
DEFAULT_ZOOM_INDEX: int = 4

# Level-of-detail threshold for zoom-dependent rendering.
# Below this zoom, tiles are 16px or smaller and fine detail (terrain glyph
# variation, color jitter, edge feathering, shadow wall-clipping) is
# imperceptible. Expensive detail work is skipped at lower zoom stops.
LOD_DETAIL_ZOOM_THRESHOLD: float = 0.8

# Safety floor for dynamic console dimensions after DPI + zoom are applied.
# If a requested TILE_ZOOM would shrink below this, the renderer reduces the
# effective zoom until these minimums are met.
MIN_CONSOLE_WIDTH: int = 40
MIN_CONSOLE_HEIGHT: int = 25

# Toggle environmental overlay effects
ENVIRONMENTAL_EFFECTS_ENABLED = True
# Toggle atmospheric effects (cloud shadows, ground mist)
ATMOSPHERIC_EFFECTS_ENABLED = True
# Toggle rain effect (GPU procedural line segments)
RAIN_ENABLED: bool = False
RAIN_INTENSITY: float = 0.4
RAIN_ANGLE: float = 0.15
# Rain tilt from vertical (radians). Keep below pi/2 so drops always fall down.
RAIN_ANGLE_MAX_ABS_RAD: float = 0.7
RAIN_WIND_DRIZZLE_MAX_ABS_RAD: float = 0.10
RAIN_WIND_DOWNPOUR_MAX_ABS_RAD: float = 0.30
# Continuous micro-budging around baseline angle.
RAIN_WIND_MICRO_MAX_ABS_RAD: float = 0.0
# Scale applied to micro variation by rain density, as
# (drizzle_scale, downpour_scale).
RAIN_WIND_MICRO_DENSITY_SCALE_RANGE: tuple[float, float] = (1.0, 0.35)
# Gust cadence and envelope timing.
RAIN_WIND_GUSTS_ENABLED: bool = False
RAIN_WIND_GUST_INTERVAL_SEC_RANGE: tuple[float, float] = (18.0, 45.0)
RAIN_WIND_GUST_DURATION_SEC_RANGE: tuple[float, float] = (0.8, 2.2)
# Probability a gust keeps the previous gust's direction.
RAIN_WIND_GUST_KEEP_DIRECTION_PROB: float = 0.8
# Additional smoothing for gust contribution to the render angle.
# Lower values = softer/eased gust transitions, higher values = snappier.
RAIN_WIND_GUST_OFFSET_RESPONSE_RATE: float = 3.5
# How quickly render angle tracks the baseline+wind target.
RAIN_WIND_ANGLE_RESPONSE_RATE: float = 6.0
RAIN_DROP_LENGTH: float = 0.8
RAIN_DROP_SPEED: float = 25.0
RAIN_DROP_SPACING: float = 1.35
RAIN_STREAM_SPACING: float = 0.33
RAIN_COLOR: tuple[int, int, int] = (180, 200, 220)
# Rain-driven sunlight tuning:
# - Dimming multiplies the base (pre-rain) sun intensity.
# - Color blend lerps from base sun color toward a cooler overcast target.
# - Cloud coverage controls atmospheric overcast response.
# Ranges are (drizzle_value, downpour_value).
RAIN_SUN_DIM_RANGE: tuple[float, float] = (0.85, 0.45)
RAIN_SUN_COOL_COLOR: tuple[int, int, int] = (160, 175, 200)
RAIN_SUN_COLOR_BLEND_RANGE: tuple[float, float] = (0.10, 0.35)
RAIN_CLOUD_COVERAGE_RANGE: tuple[float, float] = (0.40, 0.95)
# Rain-spacing presets (tile units). These are reference ranges used by the
# dev-console ``rain <preset>`` command to pick randomized values.


class RainPreset(NamedTuple):
    """Randomization ranges for a rain intensity preset."""

    stream_spacing: tuple[float, float]
    drop_spacing: tuple[float, float]
    speed: tuple[float, float]
    intensity: tuple[float, float]
    angle: tuple[float, float]


RAIN_PRESETS: dict[str, RainPreset] = {
    "downpour": RainPreset(
        stream_spacing=(0.16, 0.3),
        drop_spacing=(0.75, 1.5),
        speed=(27.0, 34.0),
        intensity=(0.28, 0.40),
        angle=(-0.45, 0.45),
    ),
    "regular": RainPreset(
        stream_spacing=(0.28, 0.45),
        drop_spacing=(1.2, 2.4),
        speed=(22.0, 28.0),
        intensity=(0.28, 0.40),
        angle=(-0.30, 0.30),
    ),
    "drizzle": RainPreset(
        stream_spacing=(0.52, 0.6),
        drop_spacing=(3.2, 4.0),
        speed=(17.0, 20.5),
        intensity=(0.28, 0.35),
        angle=(-0.15, 0.15),
    ),
}

# Shake effect
# Set to False to disable screen shake
SCREEN_SHAKE_ENABLED = True

# Audio
# Set to False to disable audio entirely
AUDIO_ENABLED = True

# Debug flags for rendering troubleshooting
DEBUG_DISABLE_BACKGROUND_CACHE = False  # Re-render background every frame
DEBUG_DISABLE_LIGHT_OVERLAY = False  # Show only dark/unlit background

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

BACKEND = BackendConfig.WGPU

# ============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# --- Performance & Debug Toggles ---
# These are independent flags to control debugging features.

# Set to True for uncapped FPS to identify performance bottlenecks.
PERFORMANCE_PROFILING = False

# Set to True to print metrics about the "tap vs. hold" input system.
# This measures the real-world time between player-perceived moves, which is
# dominated by the KEY_REPEAT_INTERVAL in the MovementInputHandler.
# Useful for tuning game feel, not for measuring raw CPU performance.
SHOW_ACTION_PROCESSING_METRICS = False

# Set to True to initially watch the ``dev.fps`` variable, causing
# the FPS counter to appear in the debug stats overlay at startup.
# This setting is independent of PERFORMANCE_PROFILING.
SHOW_FPS = False

# Set to True to draw colored outlines around each View for debugging layouts.
DEBUG_DRAW_VIEW_OUTLINES = False
# Set to True to render a debug tile grid overlay on the world view.
DEBUG_SHOW_TILE_GRID: bool = False

# --- Begin Engine Settings (Derived from flags above) - DO NOT CHANGE DIRECTLY ---

# Let VSYNC (or lack thereof) control frame timing.
# Set TARGET_FPS = 60 for battery-friendly mode on laptops
# (sacrifices smoothness on >60Hz displays).
TARGET_FPS = None

# VSYNC off for profiling (uncapped FPS), on otherwise (sync to monitor refresh rate).
VSYNC = not PERFORMANCE_PROFILING

# --- End Engine Settings ---

# =============================================================================
# GAMEPLAY MECHANICS
# =============================================================================

# Action economy
ACTION_COST = 100  # Energy cost for standard actions
DEFAULT_ACTOR_SPEED = 100  # Default speed for actors (energy gained per round)

# Set to False to make hostile NPCs passive (won't attack or pursue)
# NPCs will still perform reactive behaviors like escaping hazards.
HOSTILE_AI_ENABLED = True

# Field of view
FOV_RADIUS = 50  # Player's sight radius (large enough to cover full viewport)

# Action context
# Radius for "nearby actor" queries in action discovery and combat context.
ACTION_CONTEXT_RADIUS = 15

# Combat & health
PLAYER_BASE_STRENGTH = 3
PLAYER_BASE_TOUGHNESS = 30  # Player's starting toughness score


# =============================================================================
# MAP GENERATION
# =============================================================================

# Map size
MAP_WIDTH = 120
MAP_HEIGHT = 80

# Generator type: DUNGEON (legacy rooms+corridors) or SETTLEMENT (pipeline-based)
# TODO: Add WILDERNESS once CellularAutomataTerrainLayer is implemented
MAP_GENERATOR_TYPE = GeneratorType.SETTLEMENT

# Room generation (for dungeon generator)
MAX_ROOM_SIZE = 20
MIN_ROOM_SIZE = 6
MAX_NUM_ROOMS = 3

# Settlement generation (for settlement generator)
SETTLEMENT_STREET_STYLE = "cross"  # "single", "cross", or "grid"
SETTLEMENT_LOT_MIN_SIZE = (
    14  # Min lot dimension for BSP subdivision (must fit smallest template + margin)
)
SETTLEMENT_LOT_MAX_SIZE = (
    26  # Max lot dimension (larger lots fit bigger buildings like taverns)
)
SETTLEMENT_BUILDING_DENSITY = (
    0.85  # Probability of placing a building in each lot (0.0-1.0)
)
SETTLEMENT_MAX_BUILDINGS = 12  # Maximum buildings per settlement

# Tree placement
# Wild trees: forested borders around the settlement. Noise modulation creates
# groves and clearings so the player has natural paths through the trees.
SETTLEMENT_WILD_TREE_DENSITY = 0.05  # Probability per eligible tile in wild areas
# Yard trees: sparse, deliberate placement near buildings. Settlements are
# cleared land - a tree in a yard here and there, not an obstacle course.
SETTLEMENT_YARD_TREE_DENSITY = 0.015  # Probability per eligible tile near buildings

# Boulder placement
# Wild boulders provide most outdoor hard-cover and visual texture.
SETTLEMENT_WILD_BOULDER_DENSITY = 0.008
# Near roads/buildings, boulders should be very rare because cleared
# settlement space should stay readable and navigable.
SETTLEMENT_SETTLEMENT_BOULDER_DENSITY = 0.0005
# Distance from streets considered "settlement" for boulder density split.
SETTLEMENT_BOULDER_STREET_BUFFER = 12

# Noise-modulated boulder clustering.
# Compared to trees, boulder modulation is deliberately subtle so we get small
# natural clusters and sparse gaps without large dramatic bands.
BOULDER_DENSITY_NOISE_FREQUENCY = 0.015
BOULDER_DENSITY_NOISE_OCTAVES = 2

# Natural terrain generation.
# The pre-settlement landscape uses a single noise field over a dirt base.
# Noise controls where grass grows. Gravel is not a natural terrain type -
# it's a transitional material placed by feature layers (street margins,
# future riverbeds, cliff bases).
#
# Grass noise: controls where grass patches grow over the dirt base.
# The threshold controls coverage: lower values = more grass. At -0.1
# roughly 55% of the map becomes grass.
GRASS_NOISE_FREQUENCY = 0.02  # ~50-tile features for grass patch scale
GRASS_NOISE_OCTAVES = 3  # Coarse shape + medium detail + fine irregularity
GRASS_NOISE_THRESHOLD = -0.1  # Noise values above this become grass
#
# Grass islands: a second, high-frequency noise field that creates small
# isolated patches where the terrain type flips. The high frequency produces
# small 1-3 tile blobby shapes, and the high threshold means only the noise
# peaks poke through, keeping patches sparse and isolated.
#
# Asymmetric thresholds reflect ecology: grass colonizes dirt more easily
# (seeds blow in, take root in favorable pockets) than dirt survives inside
# grass (requires something actively preventing growth - compaction, poor
# drainage, heavy shade). So grass-in-dirt islands are more common.
GRASS_ISLAND_FREQUENCY = 0.15  # High frequency for small (~6-7 tile) features
GRASS_ISLAND_OCTAVES = 2  # Just enough detail for organic blob shapes
GRASS_ISLAND_THRESHOLD = 0.72  # Noise peaks above this flip dirt to grass
GRASS_ISLAND_BARE_THRESHOLD = 0.9  # Higher bar to flip grass to bare dirt

# Street margin buffer.
# After streets are carved, a margin of cleared ground is placed along the
# edges so cobblestone doesn't cut a hard right-angle against grass. Noise
# drives both the margin width and the material choice per tile, so the
# border is irregular - gravel spills further in some spots, grass reclaims
# the edge in others.
STREET_MARGIN_MAX = 3  # Maximum margin width in tiles (noise varies 0 to this)
STREET_MARGIN_NOISE_FREQUENCY = 0.12  # Per-tile irregularity for margin width
STREET_MARGIN_GRAVEL_BIAS = 0.85  # Probability that a margin tile is gravel vs dirt

# Noise-modulated tree clustering.
# A low-frequency noise field scales placement probability per tile, creating
# organic groves (high noise → 3x density) and clearings (low noise → 0 density).
# A separate noise channel with different seed/frequency biases the deciduous vs
# conifer ratio spatially, producing "pine groves" and "hardwood groves".
TREE_DENSITY_NOISE_FREQUENCY = 0.02  # ~50-tile features for grove/clearing scale
TREE_DENSITY_NOISE_OCTAVES = 3  # Coarse shape + medium detail + fine irregularity
TREE_SPECIES_NOISE_FREQUENCY = 0.008  # ~125-tile biome zones (larger than groves)
TREE_SPECIES_NOISE_OCTAVES = 2
# XOR salt to derive species noise seed from decoration_seed, so the two noise
# fields are uncorrelated.
TREE_SPECIES_SEED_XOR = 0xA5A5_A5A5


# =============================================================================
# LIGHTING SYSTEM
# =============================================================================

# Torch preset values (used in LightSource.create_torch())
TORCH_RADIUS = 10
TORCH_COLOR = (179, 128, 77)  # Warm orange/yellow
TORCH_FLICKER_SPEED = 3.0
TORCH_MIN_BRIGHTNESS = 1.15
TORCH_MAX_BRIGHTNESS = 1.35

# Lighting system defaults
AMBIENT_LIGHT_LEVEL = 0.2  # Base light level for all areas

# Global sunlight configuration
SUN_ENABLED = True  # Enable sunlight illumination
SUN_COLOR = (255, 243, 204)  # Warm sunlight color (RGB 0-255)
SUN_ELEVATION_DEGREES = 45.0  # Sun elevation above horizon (0-90 degrees)
SUN_AZIMUTH_DEGREES = 135.0  # Sun direction (0=North, 90=East, 180=South, 270=West)
SUN_INTENSITY = 0.75  # Base sun intensity multiplier
SKY_EXPOSURE_POWER = 1.5  # Non-linear sky exposure curve (higher = more contrast)

# Shadow system
SHADOWS_ENABLED: bool = True  # Set to False to disable shadows
SHADOW_INTENSITY = 0.25  # Point light shadow intensity (indoor torches, etc.)
SUN_SHADOW_INTENSITY = 0.55  # Directional/sun shadow intensity
SHADOW_MAX_LENGTH = 6  # Maximum shadow march distance (accommodates tallest terrain)
SHADOW_FALLOFF = True  # Shadows get lighter with distance
ACTOR_SHADOW_ALPHA: float = 0.65  # Base alpha for projected actor shadow quads
ACTOR_SHADOW_FADE_TIP: bool = True  # Fade actor shadow alpha to zero at the tip
ACTOR_SHADOW_BLUR_RADIUS: float = 2.0  # Gaussian blur radius for shadow atlas
TERRAIN_GLYPH_SHADOW_ALPHA: float = 0.55  # Alpha for terrain glyph shadows (boulders)

# Tile emission (glowing hazard tiles like acid pools, hot coals)
TILE_EMISSION_ENABLED = True

# =============================================================================
# INPUT & CONTROLS
# =============================================================================

# Movement
MOVEMENT_KEY_REPEAT_DELAY = 0.25
MOVEMENT_KEY_REPEAT_INTERVAL = 0.07

# Unified action timing: duration controls both animation and pacing.
# Duration for held-key movement (ms). Fast, responsive. Matches current feel.
HELD_KEY_MOVE_DURATION_MS = 70
# Duration for autopilot movement (ms). Slightly slower, can vary for approach.
AUTOPILOT_MOVE_DURATION_MS = 100

# =============================================================================
# ASSET PATHS
# =============================================================================

ASSETS_BASE_DIR = PROJECT_ROOT_PATH / "assets"

BASE_MOUSE_CURSOR_PATH = ASSETS_BASE_DIR / "cursors"

# UI Font Configuration
UI_FONT_PATH = ASSETS_BASE_DIR / "fonts" / "CozetteVector.ttf"

# Per-view font sizes
MESSAGE_LOG_FONT_SIZE = 36
ACTION_PANEL_FONT_SIZE = 48
PLAYER_STATUS_FONT_SIZE = 48
EQUIPMENT_FONT_SIZE = 48  # Same size as action panel and player status
MENU_FONT_SIZE = 36  # Slightly smaller than action panel to fit more content

# Menu line spacing multiplier. At 1.0, line height = ascent + descent.
# Increase slightly (e.g., 1.02) if box-drawing chars show gaps between rows.
MENU_LINE_SPACING = 1.0

# Tileset
TILESET_PATH = ASSETS_BASE_DIR / "tilesets" / "Taffer_20x20.png"
TILESET_COLUMNS = 16
TILESET_ROWS = 16

# =============================================================================
# PROBABILITY DESCRIPTORS
# =============================================================================

# Probability ranges and their descriptors
# Format: (max_probability, descriptor, color_name)
PROBABILITY_DESCRIPTORS = [
    (0.20, "Long Shot", "red"),
    (0.40, "Unlikely", "orange"),
    (0.60, "Even Odds", "yellow"),
    (0.80, "Very Likely", "light_green"),
    (1.00, "Almost Certain", "green"),
]

# =============================================================================
# MESSAGE LOG
# =============================================================================

SHOW_MESSAGE_SEQUENCE_NUMBERS = False
PRINT_MESSAGES_TO_CONSOLE = False
