"""
Configuration constants.

Centralizes all magic numbers and configuration values used throughout the codebase.
Organized by functional area for easy maintenance.
"""

import sys
from pathlib import Path
from typing import Literal

import tcod.constants

from catley.types import Opacity

# =============================================================================
# GENERAL
# =============================================================================

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

# RANDOM_SEED = None
RANDOM_SEED = "burrito1"

# Test environment detection
IS_TEST_ENVIRONMENT = "pytest" in sys.modules

# =============================================================================
# DISPLAY & RENDERING
# =============================================================================

# Main window
WINDOW_TITLE = "Catley Prototype"

# Screen dimensions
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 50

# Viewport defaults used when initializing views before they are resized.
DEFAULT_VIEWPORT_WIDTH = SCREEN_WIDTH
DEFAULT_VIEWPORT_HEIGHT = 40  # Initial height before layout adjustments

# UI Layout
HELP_HEIGHT = 1  # Lines reserved for help text at top

# Use "smooth" rendering for actors (sub-tile render coordinates).
SMOOTH_ACTOR_RENDERING_ENABLED = True
# Toggle environmental overlay effects
ENVIRONMENTAL_EFFECTS_ENABLED = True

# Rendering effects
PULSATION_PERIOD = 2.0  # Seconds for full pulsation cycle (selected actor)
PULSATION_MAX_BLEND_ALPHA = Opacity(0.5)  # Maximum alpha for pulsation blending
LUMINANCE_THRESHOLD = 127.5  # For determining light vs dark colors

# Combat outline shimmer effect (shimmering glyph outlines on targetable enemies)
COMBAT_OUTLINE_SHIMMER_PERIOD = 2.4  # Seconds for full shimmer cycle
COMBAT_OUTLINE_MIN_ALPHA = Opacity(0.4)  # Minimum alpha during shimmer
COMBAT_OUTLINE_MAX_ALPHA = Opacity(0.85)  # Maximum alpha during shimmer

# Contextual target outline (exploration mode)
CONTEXTUAL_OUTLINE_ALPHA = Opacity(0.70)  # Solid outline opacity

# Shake effect
# Set to False to disable screen shake
SCREEN_SHAKE_ENABLED = True

# Audio
# Set to False to disable audio entirely
AUDIO_ENABLED = True

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Application backend selection
APP_BACKEND: Literal["tcod", "glfw"] = "glfw"

# Graphics context selection
# - "moderngl":
#   Fast startup (~350ms). Recommended for now.
# - "wgpu":
#   Ready to go and future-proof (WebGPU/Vulkan/Metal).
#   BUT: slow startup (~1200ms) due to pipeline state object creation overhead.
#   The first pipeline creation triggers Metal shader compiler init (~165ms) and
#   adapter/device enumeration (~290ms). wgpu-py doesn't yet support Apple's
#   Metal Binary Archives for caching compiled pipelines.
#
#   When switching to "wgpu" as a default, consider implementing a loading screen
#   to mask the ~1.2s init time, or just accept the startup cost.
GRAPHICS_BACKEND: Literal["tcod", "moderngl", "wgpu"] = "moderngl"

# Lighting system selection (independent of graphics backend)
# Same tradeoffs as GRAPHICS_BACKEND above.
LIGHTING_BACKEND: Literal["moderngl", "wgpu"] = "moderngl"

# These are the valid combinations of
# APP_BACKEND, GRAPHICS_BACKEND, and LIGHTING_BACKEND.
VALID_BACKEND_CONFIG_COMBINATIONS = [
    ("tcod", "tcod", "moderngl"),
    ("tcod", "tcod", "wgpu"),
    ("glfw", "moderngl", "moderngl"),
    ("glfw", "wgpu", "wgpu"),
]

# ============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

FPS_SAMPLE_SIZE = 256  # Number of frame time samples to track

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
FOV_RADIUS = 15  # Player's sight radius
FOV_ALGORITHM = tcod.constants.FOV_SYMMETRIC_SHADOWCAST
FOV_LIGHT_WALLS = True

# Combat & health
PLAYER_BASE_STRENGTH = 3
PLAYER_BASE_TOUGHNESS = 30  # Player's starting toughness score


# =============================================================================
# MAP GENERATION
# =============================================================================

# Map size
MAP_WIDTH = 80
MAP_HEIGHT = 43

# Room generation
MAX_ROOM_SIZE = 20
MIN_ROOM_SIZE = 6
MAX_NUM_ROOMS = 3


# =============================================================================
# LIGHTING SYSTEM
# =============================================================================


# Generic light source defaults
DEFAULT_LIGHT_COLOR = (255, 255, 255)  # Pure white
DEFAULT_FLICKER_SPEED = 3.0
DEFAULT_MIN_BRIGHTNESS = 1.15
DEFAULT_MAX_BRIGHTNESS = 1.35

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
SUN_INTENSITY = 0.4  # Base sun intensity multiplier
SKY_EXPOSURE_POWER = 1.5  # Non-linear sky exposure curve (higher = more contrast)

# Shadow system
SHADOWS_ENABLED: bool = True  # Set to False to disable shadows
SHADOW_INTENSITY = 0.17  # How dark shadows are (0.0 = no shadow, 1.0 = completely dark)
SHADOW_MAX_LENGTH = 5  # Maximum shadow length in tiles
SHADOW_FALLOFF = True  # Shadows get lighter with distance

# Tile emission (glowing hazard tiles like acid pools, hot coals)
TILE_EMISSION_ENABLED = True

# =============================================================================
# INPUT & CONTROLS
# =============================================================================

# Mouse and selection
MOUSE_HIGHLIGHT_ALPHA = Opacity(0.6)  # Alpha blending for mouse cursor highlight

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

# Alternative flavor sets (easily swappable)
PROBABILITY_DESCRIPTORS_POST_APOCALYPTIC = [
    (0.20, "Desperate Gamble", "red"),
    (0.40, "Risky", "orange"),
    (0.60, "Fighting Chance", "yellow"),
    (0.80, "Good Shot", "light_green"),
    (1.00, "Sure Thing", "green"),
]

PROBABILITY_DESCRIPTORS_MILITARY = [
    (0.20, "Low Confidence", "red"),
    (0.40, "Poor Odds", "orange"),
    (0.60, "Fifty-Fifty", "yellow"),
    (0.80, "High Confidence", "light_green"),
    (1.00, "Mission Critical", "green"),
]

# =============================================================================
# MESSAGE LOG
# =============================================================================

SHOW_MESSAGE_SEQUENCE_NUMBERS = False
PRINT_MESSAGES_TO_CONSOLE = False

# =============================================================================
# Supporting functions
# =============================================================================


def validate_backend_configuration() -> None:
    """Validate the backend configuration.

    Raises:
        ValueError: If the backend combination is not valid.
    """
    if (
        APP_BACKEND,
        GRAPHICS_BACKEND,
        LIGHTING_BACKEND,
    ) not in VALID_BACKEND_CONFIG_COMBINATIONS:
        valid_combos = "\n  ".join(str(c) for c in VALID_BACKEND_CONFIG_COMBINATIONS)
        raise ValueError(
            f"Invalid backend configuration: "
            f"APP_BACKEND={APP_BACKEND}, GRAPHICS_BACKEND={GRAPHICS_BACKEND}, "
            f"LIGHTING_BACKEND={LIGHTING_BACKEND}.\n"
            f"Valid combinations are:\n  {valid_combos}"
        )
