from __future__ import annotations

import numpy as np

from brileta import colors

# This is the public, backend-agnostic data type for a single cell.
# Views and other game logic can create arrays of this type.
GLYPH_DTYPE = np.dtype(
    [
        ("ch", np.int32),  # Character code
        ("fg", "4B"),  # Foreground RGBA (4 unsigned bytes)
        ("bg", "4B"),  # Background RGBA (4 unsigned bytes)
        ("noise", np.float32),  # Sub-tile noise amplitude (0.0 = no noise)
        ("noise_pattern", np.uint8),  # Sub-tile noise pattern ID (0 = default blocks)
        ("edge_neighbor_mask", np.uint8),  # Cardinal diff mask (W/N/S/E bits)
        ("edge_blend", np.float32),  # Organic edge feathering amplitude (0.0-1.0)
        (
            "edge_neighbor_bg",
            np.uint8,
            (4, 3),
        ),  # Cardinal-neighbor background RGB (W, N, S, E) for edge blending
        # Sub-tile split fields for perspective offset wall/roof boundary tiles.
        # split_y in [0, 1]: 0 = no split (whole tile primary), >0 = above threshold
        # is primary appearance, below threshold uses split_bg/split_fg/split_noise.
        ("split_y", np.float32),
        ("split_bg", "4B"),  # Background RGBA for the below-split portion
        ("split_fg", "4B"),  # Foreground RGBA for the below-split portion
        ("split_noise", np.float32),  # Noise amplitude for the below-split portion
        ("split_noise_pattern", np.uint8),  # Noise pattern for the below-split portion
        # Packed wear/aging data for per-pixel shader effects on roofs.
        # Bits 0-7: material (0=none, 1=thatch, 2=shingle, 3=tin),
        # bits 8-15: condition (0-255), bits 16-23: edge proximity (0-255).
        ("wear_pack", np.uint32),
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
        self.data: np.ndarray = np.zeros((width, height), dtype=GLYPH_DTYPE)
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
        self.data["noise"] = 0.0
        self.data["noise_pattern"] = 0
        self.data["edge_neighbor_mask"] = 0
        self.data["edge_blend"] = 0.0
        self.data["edge_neighbor_bg"] = 0
        self.data["split_y"] = 0.0
        self.data["split_bg"] = (0, 0, 0, 0)
        self.data["split_fg"] = (0, 0, 0, 0)
        self.data["split_noise"] = 0.0
        self.data["split_noise_pattern"] = 0
        self.data["wear_pack"] = 0

    def put_char(
        self,
        x: int,
        y: int,
        ch: int,
        fg: colors.ColorRGBA,
        bg: colors.ColorRGBA,
        noise: float = 0.0,
        noise_pattern: int = 0,
    ) -> None:
        """Places a single character at a given coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data["ch"][x, y] = ch
            self.data["fg"][x, y] = fg
            self.data["bg"][x, y] = bg
            self.data["noise"][x, y] = noise
            self.data["noise_pattern"][x, y] = noise_pattern
            self.data["edge_neighbor_mask"][x, y] = 0
            self.data["edge_blend"][x, y] = 0.0
            self.data["edge_neighbor_bg"][x, y] = 0
            self.data["split_y"][x, y] = 0.0
            self.data["split_bg"][x, y] = (0, 0, 0, 0)
            self.data["split_fg"][x, y] = (0, 0, 0, 0)
            self.data["split_noise"][x, y] = 0.0
            self.data["split_noise_pattern"][x, y] = 0
            self.data["wear_pack"][x, y] = 0

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

        target_slice = self.data[x : x + len(text), y]

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

        target_slice = self.data[x1:x2, y1:y2]

        if clear:
            target_slice["ch"] = ord(" ")

        target_slice["bg"] = bg

        # Draw borders on the slice using Unicode box-drawing characters
        target_slice[:, 0]["ch"] = 9472  # ─ Top
        target_slice[:, -1]["ch"] = 9472  # ─ Bottom
        target_slice[0, :]["ch"] = 9474  # │ Left
        target_slice[-1, :]["ch"] = 9474  # │ Right

        # Draw corners
        target_slice[0, 0]["ch"] = 9484  # ┌ Top-left
        target_slice[-1, 0]["ch"] = 9488  # ┐ Top-right
        target_slice[0, -1]["ch"] = 9492  # └ Bottom-left
        target_slice[-1, -1]["ch"] = 9496  # ┘ Bottom-right

        target_slice["fg"] = fg

        if title:
            # Title is drawn with fg color on bg color, inverted from the frame
            # We need to print relative to the frame's corner
            self.print(x1 + 2, y1, f" {title} ", fg=bg, bg=fg)
