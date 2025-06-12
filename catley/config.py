"""
Configuration constants.

Centralizes all magic numbers and configuration values used throughout the codebase.
Organized by functional area for easy maintenance.
"""

from pathlib import Path

import tcod.constants

# =============================================================================
# GENERAL
# =============================================================================

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

RANDOM_SEED = None
# RANDOM_SEED = "burrito burrito"

# =============================================================================
# DISPLAY & RENDERING
# =============================================================================

# Main window
WINDOW_TITLE = "Catley Prototype"

# Screen dimensions
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 50

# Viewport defaults used when initializing panels before they are resized.
DEFAULT_VIEWPORT_WIDTH = SCREEN_WIDTH
DEFAULT_VIEWPORT_HEIGHT = 40  # Initial height before layout adjustments

# UI Layout
HELP_HEIGHT = 1  # Lines reserved for help text at top

# Rendering effects
PULSATION_PERIOD = 2.0  # Seconds for full pulsation cycle (selected actor)
PULSATION_MAX_BLEND_ALPHA = 0.5  # Maximum alpha for pulsation blending
LUMINANCE_THRESHOLD = 127.5  # For determining light vs dark colors

# Shake effect
# Set to False to disable screen shake
SCREEN_SHAKE_ENABLED = True

# Scale all shake intensity (0.5 = half strength, 2.0 = double)
SCREEN_SHAKE_INTENSITY_MULTIPLIER = 0.2

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

FPS_SAMPLE_SIZE = 256  # Number of frame time samples to track

PERFORMANCE_TESTING = False
if PERFORMANCE_TESTING:
    # Shows true uncapped performance for bottleneck identification
    TARGET_FPS = None
    VSYNC = False
    SHOW_FPS = True
else:
    # Release Build & Daily Development (battery-friendly)
    TARGET_FPS = 60
    VSYNC = True
    SHOW_FPS = False

# =============================================================================
# GAMEPLAY MECHANICS
# =============================================================================

# Action economy
ACTION_COST = 100  # Energy cost for standard actions
DEFAULT_ACTOR_SPEED = 100  # Default speed for actors (energy gained per round)

# Field of view
FOV_RADIUS = 15  # Player's sight radius
FOV_ALGORITHM = tcod.constants.FOV_SYMMETRIC_SHADOWCAST
FOV_LIGHT_WALLS = True

# Combat & health
DEFAULT_MAX_ARMOR = 3  # Default maximum armor points
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
AMBIENT_LIGHT_LEVEL = 0.1  # Base light level for all areas

# Shadow system
SHADOWS_ENABLED = True  # Set to False to disable shadows
SHADOW_INTENSITY = 0.17  # How dark shadows are (0.0 = no shadow, 1.0 = completely dark)
SHADOW_MAX_LENGTH = 3  # Maximum shadow length in tiles
SHADOW_FALLOFF = True  # Shadows get lighter with distance

# =============================================================================
# INPUT & CONTROLS
# =============================================================================

# Mouse and selection
MOUSE_HIGHLIGHT_ALPHA = 0.6  # Alpha blending for mouse cursor highlight
SELECTION_HIGHLIGHT_ALPHA = 0.6  # Alpha blending for selected actor highlight


# =============================================================================
# ASSET PATHS
# =============================================================================

ASSETS_BASE_DIR = PROJECT_ROOT_PATH / "assets"

BASE_MOUSE_CURSOR_PATH = ASSETS_BASE_DIR / "cursors"

MESSAGE_LOG_FONT_PATH = ASSETS_BASE_DIR / "fonts" / "SourceSans3-Medium.ttf"
MESSAGE_LOG_FONT_SIZE = 20

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
