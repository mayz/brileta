# Type alias for RGB colors
Color = tuple[int, int, int]

# Basic colors
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
RED: Color = (255, 0, 0)
GREEN: Color = (0, 255, 0)
BLUE: Color = (0, 0, 255)
YELLOW: Color = (255, 255, 0)
CYAN: Color = (0, 255, 255)
ORANGE: Color = (255, 165, 0)
GREY: Color = (128, 128, 128)
LIGHT_GREY: Color = (200, 200, 200)
DARK_GREY: Color = (50, 50, 50)
LIGHT_BLUE: Color = (173, 216, 230)

# Map colors
DARK_WALL: Color = BLACK
DARK_GROUND: Color = (0, 0, 100)
LIGHT_WALL: Color = (130, 110, 50)
LIGHT_GROUND: Color = (200, 180, 50)

# Entity colors
PLAYER_COLOR: Color = WHITE
NPC_COLOR: Color = YELLOW
SELECTED_HIGHLIGHT: Color = WHITE

DEAD: Color = (128, 128, 128)
