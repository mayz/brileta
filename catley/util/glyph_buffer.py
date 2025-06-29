from __future__ import annotations

import numpy as np

from catley import colors

# This is the public, backend-agnostic data type for a single cell.
# Views and other game logic can create arrays of this type.
GLYPH_DTYPE = np.dtype(
    [
        ("ch", np.int32),  # Character code
        ("fg", "4B"),  # Foreground RGBA (4 unsigned bytes)
        ("bg", "4B"),  # Background RGBA (4 unsigned bytes)
    ]
)


class GlyphBuffer:
    """
    A backend-agnostic 2D grid of glyphs, foreground, and background colors.

    This class provides a simple, NumPy-based drawing surface for views. It holds
    the semantic information for a scene (what character and colors go where)
    without any knowledge of how it will be rendered. It is a pure data model,
    intended to be passed to a concrete GraphicsContext for rendering.
    """

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")
        self.width = width
        self.height = height
        self.data: np.ndarray = np.zeros((height, width), dtype=GLYPH_DTYPE)
        self.clear()

    def clear(
        self,
        ch: int = ord(" "),
        fg: colors.ColorRGBA = (0, 0, 0, 0),
        bg: colors.ColorRGBA = (0, 0, 0, 0),
    ) -> None:
        """Fills the entire buffer with a single glyph."""
        self.data["ch"] = ch
        self.data["fg"] = fg
        self.data["bg"] = bg

    def put_char(
        self, x: int, y: int, ch: int, fg: colors.ColorRGBA, bg: colors.ColorRGBA
    ) -> None:
        """Places a single character at a given coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y, x] = (ch, fg, bg)

    def print(
        self,
        x: int,
        y: int,
        text: str,
        fg: colors.ColorRGBA,
        bg: colors.ColorRGBA | None = None,
    ) -> None:
        """Draws a string of text. Vectorized for performance."""
        if not (0 <= y < self.height and x < self.width):
            return

        # Truncate text if it goes off the right edge
        if x < 0:
            text = text[-x:]
            x = 0
        if x + len(text) > self.width:
            text = text[: self.width - x]

        if not text:
            return

        target_slice = self.data[y, x : x + len(text)]

        char_array = np.frombuffer(text.encode("utf-32-le"), dtype=np.int32)
        target_slice["ch"] = char_array
        target_slice["fg"] = fg
        if bg is not None:
            target_slice["bg"] = bg

    def draw_frame(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        fg: colors.ColorRGBA,
        bg: colors.ColorRGBA,
        title: str = "",
        clear: bool = True,
    ) -> None:
        """Draws a framed box using NumPy slicing for performance."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        x1, y1 = x, y
        x2, y2 = min(x1 + width, self.width), min(y1 + height, self.height)

        target_slice = self.data[y1:y2, x1:x2]

        if clear:
            target_slice["ch"] = ord(" ")

        target_slice["bg"] = bg

        # Draw borders on the slice
        target_slice[0, :]["ch"] = 196  # Top
        target_slice[-1, :]["ch"] = 196  # Bottom
        target_slice[:, 0]["ch"] = 179  # Left
        target_slice[:, -1]["ch"] = 179  # Right

        # Draw corners
        target_slice[0, 0]["ch"] = 218
        target_slice[0, -1]["ch"] = 191
        target_slice[-1, 0]["ch"] = 192
        target_slice[-1, -1]["ch"] = 217

        target_slice["fg"] = fg

        if title:
            # Title is drawn with fg color on bg color, inverted from the frame
            # We need to print relative to the frame's corner
            self.print(x1 + 2, y1, f" {title} ", fg=bg, bg=fg)
