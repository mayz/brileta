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
LIGHT_ORANGE: Color = (255, 200, 100)
LIGHT_GREEN: Color = (144, 238, 144)
GREY: Color = (128, 128, 128)
MEDIUM_GREY: Color = (170, 170, 170)
LIGHT_GREY: Color = (200, 200, 200)
DARK_GREY: Color = (50, 50, 50)
LIGHT_BLUE: Color = (173, 216, 230)
MAGENTA: Color = (255, 0, 255)

# UI specific colors
MENU_HOVER_BG: Color = (40, 40, 40)  # Dark grey for hover background

# Item category colors (for inventory UI prefixes)
CATEGORY_WEAPON: Color = (220, 60, 60)  # Desaturated red
CATEGORY_ARMOR: Color = (100, 140, 220)  # Steel blue
CATEGORY_CONSUMABLE: Color = (80, 200, 120)  # Soft green
CATEGORY_JUNK: Color = (128, 128, 128)  # Gray
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

# Actor colors
PLAYER_COLOR: Color = WHITE
NPC_COLOR: Color = YELLOW
SELECTED_HIGHLIGHT: Color = WHITE

DEAD: Color = (128, 128, 128)

# Debugging colors
DEBUG_COLORS: list[Color] = [
    (0, 255, 0),  # Lime Green
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 255, 0),  # Yellow
]
