"""UI utilities for composing complex UI elements from Canvas primitives.

Includes:
- Text wrapping utilities with word-boundary awareness
- Keycap drawing for keyboard hints
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from catley import colors
from catley.util.coordinates import PixelCoord

if TYPE_CHECKING:
    from catley.view.render.canvas import Canvas


def wrap_text_by_words(
    text: str,
    fits_in_width: Callable[[str], bool],
) -> list[str]:
    """Wrap text at word boundaries using a width-checking function.

    This is the shared word-wrapping algorithm used by all backends. Each backend
    provides its own width measurement function (pixel-based for Pillow,
    character-count-based for tile-based backends).

    Args:
        text: Text to wrap
        fits_in_width: Function that returns True if a string fits in the target width

    Returns:
        List of wrapped lines. Never returns an empty list - always at least one line.
    """
    words = text.split()
    if not words:
        return [text] if text else [""]

    lines: list[str] = []
    current_line = ""
    for word in words:
        test = word if not current_line else f"{current_line} {word}"
        if fits_in_width(test):
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines if lines else [text]


def draw_keycap(
    canvas: Canvas,
    pixel_x: PixelCoord,
    pixel_y: PixelCoord,
    key: str,
    keycap_size: int | None = None,
    bg_color: colors.Color = colors.DARK_GREY,
    border_color: colors.Color = colors.GREY,
    text_color: colors.Color = colors.WHITE,
) -> int:
    """Draw a keyboard-style keycap at the specified position.

    Args:
        canvas: The canvas to draw on
        pixel_x: X position in pixels
        pixel_y: Y position in pixels (baseline of text line)
        key: The key character to display (will be converted to uppercase)
        keycap_size: Size of the keycap square (defaults to 85% of line height)
        bg_color: Background color of the keycap
        border_color: Border color of the keycap
        text_color: Color of the key character

    Returns:
        The width in pixels consumed by the keycap (including padding)
    """
    # Get font metrics to determine default size
    ascent, descent = canvas.get_font_metrics()
    line_height = ascent + descent

    if keycap_size is None:
        keycap_size = int(line_height * 0.85)

    # Calculate vertical position to align keycap with text
    # pixel_y is the TOP of the text line
    # Position keycap slightly above the text line for proper text centering
    keycap_y = pixel_y - 2

    # Use smaller font for keycap text (65% of keycap size for readability)
    keycap_font_size = max(8, int(keycap_size * 0.65))

    # Get text metrics for dynamic sizing with smaller font
    key_upper = key.upper()
    text_width, text_height, _ = canvas.get_text_metrics(
        key_upper, font_size=keycap_font_size
    )

    # Calculate dynamic width: use larger of fixed size or text width + padding
    internal_padding = 12  # Internal padding for text
    keycap_width = max(keycap_size, text_width + internal_padding)
    keycap_height = keycap_size  # Keep height fixed for consistency

    # Draw background with dynamic width
    canvas.draw_rect(
        pixel_x=pixel_x,
        pixel_y=keycap_y,
        width=keycap_width,
        height=keycap_height,
        color=bg_color,
        fill=True,
    )

    # Draw border with dynamic width (1px inset to ensure it's visible)
    canvas.draw_rect(
        pixel_x=pixel_x,
        pixel_y=keycap_y,
        width=keycap_width - 1,
        height=keycap_height - 1,
        color=border_color,
        fill=False,
    )

    # Center the text in the dynamic keycap
    text_x = pixel_x + (keycap_width - text_width) // 2
    text_y = keycap_y + (keycap_height - text_height) // 2

    # Draw the key character with smaller font
    canvas.draw_text(
        pixel_x=text_x,
        pixel_y=text_y,
        text=key_upper,
        color=text_color,
        font_size=keycap_font_size,
    )

    # Return width consumed (dynamic keycap + padding)
    return keycap_width + 12  # Dynamic width + 12px padding after keycap
