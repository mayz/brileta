from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image as PILImage

from brileta.config import BASE_MOUSE_CURSOR_PATH
from brileta.util.coordinates import PixelCoord


@dataclass
class CursorData:
    """Stores the data and lazily-loaded texture for a single cursor type."""

    name: str
    filename: str
    hotspot: tuple[int, int]  # (hot_x, hot_y) from top-left of image

    # This will hold the raw pixel data loaded from the file.
    # It's renderer-agnostic.
    pixels: np.ndarray

    # This will cache the backend-specific texture object once it's created
    # by a renderer. The `Any` type is crucial here.
    texture: Any = field(default=None, repr=False)


class CursorManager:
    """Manages loading, storing, providing, and drawing mouse cursor
    textures and hotspots."""

    def __init__(
        self,
        base_asset_path: str = "assets/cursors/",
    ) -> None:
        self.base_asset_path = base_asset_path
        self.cursors: dict[str, CursorData] = {}

        self.mouse_pixel_x: PixelCoord = 0
        self.mouse_pixel_y: PixelCoord = 0
        self.active_cursor_type: str = "arrow"

        self._load_default_cursors()

    def update_mouse_position(self, x: PixelCoord, y: PixelCoord) -> None:
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
        """
        Loads cursor image data from a file and stores it.

        This method no longer creates a texture. It only loads the raw RGBA
        pixel data into a NumPy array, which is stored in a CursorData object.
        The actual texture creation is deferred to the active renderer.
        """
        full_path = BASE_MOUSE_CURSOR_PATH / filename
        try:
            if not full_path.exists():
                print(f"ERROR: Cursor file not found at {full_path.resolve()}")
                return

            # Load the image using Pillow and convert to a standard RGBA format
            img_pil = PILImage.open(str(full_path))
            img_pil = img_pil.convert("RGBA")

            # Convert the Pillow image to a NumPy array
            pixels_rgba = np.array(img_pil, dtype=np.uint8)

            # Ensure the array is C-contiguous for compatibility with graphics libraries
            pixels_rgba = np.ascontiguousarray(pixels_rgba)

            if pixels_rgba.ndim != 3 or pixels_rgba.shape[2] != 4:
                print(
                    f"ERROR: Image '{filename}' is not in a valid RGBA format. "
                    f"Shape is {pixels_rgba.shape}."
                )
                return

            # Create and store the new, renderer-agnostic CursorData object.
            # It contains the pixel data but no texture yet.
            self.cursors[cursor_name] = CursorData(
                name=cursor_name,
                filename=filename,
                hotspot=hotspot,
                pixels=pixels_rgba,
            )

        except Exception as e:
            print(
                f"ERROR: Failed to load cursor data for '{cursor_name}' "
                f"from '{full_path}'. Exception: {e!s}"
            )
            import traceback

            traceback.print_exc()

    def _load_default_cursors(self) -> None:
        self._load_cursor(cursor_name="arrow", filename="arrow1.png", hotspot=(6, 3))
        self._load_cursor(
            cursor_name="crosshair", filename="crosshair1.png", hotspot=(15, 15)
        )
