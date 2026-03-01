# Type alias for RGB colors
Color = tuple[int, int, int]
ColorRGBA = tuple[int, int, int, int]

# Basic colors
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
RED: Color = (255, 0, 0)
GREEN: Color = (0, 255, 0)
BLUE: Color = (0, 0, 255)
YELLOW: Color = (255, 255, 0)
CYAN: Color = (0, 255, 255)
ORANGE: Color = (255, 165, 0)
DEEP_ORANGE: Color = (255, 100, 0)
LIGHT_ORANGE: Color = (255, 200, 100)
LIGHT_GREEN: Color = (144, 238, 144)
GREY: Color = (128, 128, 128)
MEDIUM_GREY: Color = (170, 170, 170)
LIGHT_GREY: Color = (200, 200, 200)
DARK_GREY: Color = (50, 50, 50)
BROWN: Color = (139, 69, 19)
TAN: Color = (180, 150, 80)
LIGHT_BLUE: Color = (173, 216, 230)
MAGENTA: Color = (255, 0, 255)

# UI specific colors
MENU_HOVER_BG: Color = (40, 40, 40)  # Dark grey for hover background

# Item category colors (for inventory UI prefixes)
CATEGORY_WEAPON: Color = (220, 60, 60)  # Desaturated red
CATEGORY_OUTFIT: Color = (100, 140, 220)  # Steel blue
CATEGORY_CONSUMABLE: Color = (80, 200, 120)  # Soft green
CATEGORY_JUNK: Color = (160, 120, 180)  # Muted purple
CATEGORY_MUNITIONS: Color = (220, 180, 60)  # Brass yellow

# Map colors
DARK_WALL: Color = BLACK
DARK_GROUND: Color = (0, 0, 100)
LIGHT_WALL: Color = (130, 110, 50)
LIGHT_GROUND: Color = (200, 180, 50)

# Outdoor map colors
OUTDOOR_DARK_GROUND: Color = (60, 45, 30)  # Dark earth/dirt
OUTDOOR_LIGHT_GROUND: Color = (120, 100, 70)  # Natural dirt/sand color
OUTDOOR_DARK_WALL: Color = (40, 30, 20)  # Dark rock/stone
OUTDOOR_LIGHT_WALL: Color = (80, 60, 40)  # Sunlit rock/stone

# Building roof colors (Phase 1 roof cutaway rendering)
ROOF_THATCH_DARK: Color = (55, 47, 30)  # Muted straw/brown in shadow
ROOF_THATCH_LIGHT: Color = (138, 118, 78)  # Sunlit straw/thatch, less saturated
ROOF_SHINGLE_DARK: Color = (38, 42, 48)  # Cool slate in shadow
ROOF_SHINGLE_LIGHT: Color = (92, 102, 116)  # Sunlit slate shingles

# Chimney colors (stone chimney rising through roof)
CHIMNEY_STONE_DARK: Color = (70, 65, 55)  # Warm grey fieldstone in shadow
CHIMNEY_STONE_LIGHT: Color = (135, 125, 110)  # Sunlit fieldstone
CHIMNEY_FLUE_DARK: Color = (20, 18, 16)  # Sooty interior in shadow
CHIMNEY_FLUE_LIGHT: Color = (35, 30, 26)  # Sooty interior in daylight
# Chimney body: south-facing side visible in 3/4 perspective, darker than
# the top surface since it faces away from the sky.
CHIMNEY_BODY_DARK: Color = (55, 50, 42)  # Fieldstone south face in shadow
CHIMNEY_BODY_LIGHT: Color = (105, 95, 82)  # Fieldstone south face in daylight

# Wall face colors: south-facing wall exposed by perspective roof offset.
# Main wall face uses warm timber/plaster tones distinct from terrain.
WALL_FACE_DARK: Color = (58, 48, 36)  # Warm timber/plaster in shadow
WALL_FACE_LIGHT: Color = (125, 108, 82)  # Sunlit timber/plaster
# Eave shadow: darker band right under the roof overhang.
WALL_EAVE_SHADOW_DARK: Color = (36, 30, 22)  # Deep shadow under eave
WALL_EAVE_SHADOW_LIGHT: Color = (80, 68, 50)  # Lit eave shadow

# Actor colors
# Player uses bright saturated gold to stand out against terrain
PLAYER_COLOR: Color = (255, 220, 80)
NPC_COLOR: Color = YELLOW
HOVER_OUTLINE: Color = (255, 240, 200)  # Warm white for hover feedback
SELECTION_OUTLINE: Color = (255, 200, 60)  # Golden outline for click-selected targets

DEAD: Color = (128, 128, 128)

# Combat targeting
COMBAT_OUTLINE: ColorRGBA = (255, 50, 50, 255)  # Red outline for targetable enemies

# Debugging colors
DEBUG_COLORS: list[Color] = [
    (0, 255, 0),  # Lime Green
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 255, 0),  # Yellow
]


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


def lerp(start: float, end: float, t: float) -> float:
    """Linearly interpolate between two scalar values (clamped to 0..1)."""
    return start + (end - start) * max(0.0, min(1.0, t))


def lerp_color(base: Color, target: Color, blend: float) -> Color:
    """Blend two RGB colors with clamped interpolation factor."""
    t = max(0.0, min(1.0, float(blend)))
    return (
        round(base[0] + (target[0] - base[0]) * t),
        round(base[1] + (target[1] - base[1]) * t),
        round(base[2] + (target[2] - base[2]) * t),
    )
