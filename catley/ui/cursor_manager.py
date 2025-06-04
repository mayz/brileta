from dataclasses import dataclass

import numpy as np
import tcod.sdl.render
from PIL import Image as PILImage

from catley.config import BASE_MOUSE_CURSOR_PATH


@dataclass
class CursorData:
    """Stores the texture and hotspot for a single cursor type."""

    texture: tcod.sdl.render.Texture
    hotspot: tuple[int, int]  # (hot_x, hot_y) from top-left of image
    name: str
    filename: str


class CursorManager:
    """Manages loading, storing, providing, and drawing mouse cursor
    textures and hotspots."""

    def __init__(
        self,
        sdl_renderer: tcod.sdl.render.Renderer,
        base_asset_path: str = "assets/cursors/",
    ) -> None:
        self.sdl_renderer = sdl_renderer
        self.base_asset_path = base_asset_path
        self.cursors: dict[str, CursorData] = {}

        self.mouse_pixel_x: int = 0
        self.mouse_pixel_y: int = 0
        self.active_cursor_type: str = "arrow"

        self._load_default_cursors()

    def update_mouse_position(self, x: int, y: int) -> None:
        """Updates the stored mouse pixel coordinates."""
        self.mouse_pixel_x = x
        self.mouse_pixel_y = y

    def set_active_cursor_type(self, cursor_type_name: str) -> None:
        """Sets the active cursor type to be drawn."""
        if cursor_type_name in self.cursors or cursor_type_name == "arrow":
            self.active_cursor_type = cursor_type_name
        else:
            print(
                f"Warning: Attempted to set unknown cursor type '{cursor_type_name}'. "
                f"Defaulting to 'arrow'."
            )
            self.active_cursor_type = "arrow"

    def draw_cursor(self) -> None:
        """Draws the active cursor at the current mouse position."""
        cursor_data = self.cursors.get(self.active_cursor_type)

        if not cursor_data:
            # This might happen if active_cursor_type was set to something not loaded
            return

        texture = cursor_data.texture
        hotspot_x, hotspot_y = cursor_data.hotspot

        dest_rect = (
            self.mouse_pixel_x - hotspot_x,
            self.mouse_pixel_y - hotspot_y,
            texture.width,
            texture.height,
        )
        self.sdl_renderer.copy(texture, dest=dest_rect)

    def get_cursor_data(self, cursor_name: str) -> CursorData | None:
        """Retrieves the CursorData object for a given cursor name."""
        return self.cursors.get(cursor_name)

    def get_hotspot(self, cursor_name: str) -> tuple[int, int]:
        """Retrieves the hotspot for a given cursor name."""
        cursor_data = self.cursors.get(cursor_name)
        return cursor_data.hotspot if cursor_data else (0, 0)

    def _load_cursor(
        self, cursor_name: str, filename: str, hotspot: tuple[int, int]
    ) -> None:
        try:
            full_path = BASE_MOUSE_CURSOR_PATH / filename

            if not full_path.exists():
                print(f"ERROR: Cursor file not found at {full_path.resolve()}")
                return

            img_pil = PILImage.open(str(full_path))
            img_pil = img_pil.convert("RGBA")

            pixels_rgba = np.array(img_pil, dtype=np.uint8)
            pixels_rgba = np.ascontiguousarray(pixels_rgba)

            if pixels_rgba.ndim != 3 or pixels_rgba.shape[2] != 4:
                print(
                    f"ERROR: Image '{filename}' not in RGBA format"
                    f" (shape: {pixels_rgba.shape})"
                )
                return

            texture = self.sdl_renderer.upload_texture(pixels_rgba)
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND

            # Create and store the CursorData object
            self.cursors[cursor_name] = CursorData(
                texture=texture, hotspot=hotspot, name=cursor_name, filename=filename
            )

        except Exception as e:
            print(
                f"ERROR: Failed to load/upload cursor '{cursor_name}' "
                f"from '{full_path}'. Exception: {e!s}"
            )
            import traceback

            traceback.print_exc()

    def _load_default_cursors(self) -> None:
        self._load_cursor(cursor_name="arrow", filename="arrow1.png", hotspot=(6, 3))
        # self._load_cursor(
        # cursor_name="crosshair",
        # filename="crosshair1.png",
        # hotspot=(15, q5))
