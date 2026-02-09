import numpy as np
import pytest

from brileta.util.glyph_buffer import GLYPH_DTYPE, GlyphBuffer


def test_glyph_buffer_init():
    buf = GlyphBuffer(10, 5)
    assert buf.width == 10
    assert buf.height == 5
    assert buf.data.shape == (10, 5)
    assert buf.data.dtype == GLYPH_DTYPE


def test_glyph_buffer_init_invalid():
    with pytest.raises(ValueError):
        GlyphBuffer(0, 10)
    with pytest.raises(ValueError):
        GlyphBuffer(10, 0)
    with pytest.raises(ValueError):
        GlyphBuffer(-1, 10)
    with pytest.raises(ValueError):
        GlyphBuffer(10, -1)


def test_clear_default():
    buf = GlyphBuffer(2, 2)
    buf.clear()
    expected = np.zeros((2, 2), dtype=GLYPH_DTYPE)
    expected["ch"] = ord(" ")
    expected["fg"] = (0, 0, 0, 0)
    expected["bg"] = (0, 0, 0, 0)
    np.testing.assert_array_equal(buf.data, expected)


def test_clear_custom():
    buf = GlyphBuffer(2, 2)
    fg_color = (255, 0, 0, 255)
    bg_color = (0, 255, 0, 255)
    buf.clear(ord("A"), fg_color, bg_color)
    expected = np.zeros((2, 2), dtype=GLYPH_DTYPE)
    expected["ch"] = ord("A")
    expected["fg"] = fg_color
    expected["bg"] = bg_color
    np.testing.assert_array_equal(buf.data, expected)


def test_put_char_in_bounds():
    buf = GlyphBuffer(3, 3)
    fg_color = (255, 255, 255, 255)
    bg_color = (0, 0, 0, 255)
    buf.put_char(1, 1, ord("@"), fg_color, bg_color)
    assert buf.data[1, 1]["ch"] == ord("@")
    np.testing.assert_array_equal(buf.data[1, 1]["fg"], fg_color)
    np.testing.assert_array_equal(buf.data[1, 1]["bg"], bg_color)


def test_put_char_out_of_bounds():
    buf = GlyphBuffer(3, 3)
    original_data = buf.data.copy()
    buf.put_char(3, 3, ord("X"), (255, 0, 0, 255), (0, 0, 0, 255))
    np.testing.assert_array_equal(buf.data, original_data)


def test_print_basic():
    buf = GlyphBuffer(10, 1)
    fg = (255, 255, 255, 255)
    buf.print(0, 0, "test", fg)
    assert "".join(chr(c) for c in buf.data[0:4, 0]["ch"]) == "test"
    for i in range(4):
        np.testing.assert_array_equal(buf.data[i, 0]["fg"], fg)


def test_print_with_bg():
    buf = GlyphBuffer(10, 1)
    fg = (255, 255, 255, 255)
    bg = (128, 0, 128, 255)
    buf.print(0, 0, "test", fg, bg)
    assert "".join(chr(c) for c in buf.data[0:4, 0]["ch"]) == "test"
    for i in range(4):
        np.testing.assert_array_equal(buf.data[i, 0]["fg"], fg)
        np.testing.assert_array_equal(buf.data[i, 0]["bg"], bg)


def test_print_off_screen():
    buf = GlyphBuffer(5, 5)
    original_data = buf.data.copy()
    buf.print(10, 10, "offscreen", (255, 255, 255, 255))
    np.testing.assert_array_equal(buf.data, original_data)


def test_print_truncate_right():
    buf = GlyphBuffer(5, 1)
    fg = (255, 255, 255, 255)
    buf.print(3, 0, "hello", fg)
    assert "".join(chr(c) for c in buf.data[3:5, 0]["ch"]) == "he"


def test_print_truncate_left():
    buf = GlyphBuffer(5, 1)
    fg = (255, 255, 255, 255)
    buf.print(-2, 0, "hello", fg)
    assert "".join(chr(c) for c in buf.data[0:3, 0]["ch"]) == "llo"


def test_print_empty_string():
    buf = GlyphBuffer(5, 5)
    original_data = buf.data.copy()
    buf.print(0, 0, "", (255, 255, 255, 255))
    np.testing.assert_array_equal(buf.data, original_data)


def test_draw_frame():
    buf = GlyphBuffer(10, 8)
    fg = (255, 255, 255, 255)
    bg = (0, 0, 0, 255)
    buf.draw_frame(1, 1, 8, 6, fg, bg)

    # Check corners
    assert buf.data[1, 1]["ch"] == 9484  # ┌ Top-left
    assert buf.data[8, 1]["ch"] == 9488  # ┐ Top-right
    assert buf.data[1, 6]["ch"] == 9492  # └ Bottom-left
    assert buf.data[8, 6]["ch"] == 9496  # ┘ Bottom-right

    # Check borders
    assert all(c == 9472 for c in buf.data[2:8, 1]["ch"])  # ─ Top
    assert all(c == 9472 for c in buf.data[2:8, 6]["ch"])  # ─ Bottom
    assert all(c == 9474 for c in buf.data[1, 2:6]["ch"])  # │ Left
    assert all(c == 9474 for c in buf.data[8, 2:6]["ch"])  # │ Right

    # Check background and foreground
    for y in range(1, 7):
        for x in range(1, 9):
            np.testing.assert_array_equal(buf.data[x, y]["bg"], bg)
            np.testing.assert_array_equal(buf.data[x, y]["fg"], fg)


def test_draw_frame_with_title():
    buf = GlyphBuffer(20, 10)
    fg = (255, 255, 255, 255)
    bg = (0, 0, 0, 255)
    title = "My Title"
    buf.draw_frame(2, 2, 16, 8, fg, bg, title=title)

    # Title text should be inverted fg/bg
    title_slice = buf.data[4 : 4 + len(title) + 2, 2]
    assert "".join(chr(c) for c in title_slice["ch"]) == f" {title} "
    for cell in title_slice:
        np.testing.assert_array_equal(cell["fg"], bg)
        np.testing.assert_array_equal(cell["bg"], fg)


def test_draw_frame_clipping():
    buf = GlyphBuffer(5, 5)
    fg = (255, 255, 255, 255)
    bg = (0, 0, 0, 255)
    buf.draw_frame(3, 3, 5, 5, fg, bg)

    # Check that it drew a clipped frame
    assert buf.data[3, 3]["ch"] == 9484  # ┌ Top-left
    assert buf.data[4, 3]["ch"] == 9488  # ┐ Top-right
    assert buf.data[3, 4]["ch"] == 9492  # └ Bottom-left
    assert buf.data[4, 4]["ch"] == 9496  # ┘ Bottom-right


def test_noise_defaults_to_zero():
    """Noise field should default to 0.0 after init and clear."""
    buf = GlyphBuffer(3, 3)
    assert buf.data["noise"][0, 0] == 0.0
    assert np.all(buf.data["noise"] == 0.0)


def test_put_char_writes_noise():
    """put_char should accept and store a noise amplitude value."""
    buf = GlyphBuffer(3, 3)
    buf.put_char(1, 1, ord("@"), (255, 255, 255, 255), (0, 0, 0, 255), noise=0.04)
    assert buf.data[1, 1]["noise"] == pytest.approx(0.04)


def test_put_char_noise_default():
    """put_char without noise arg should default to 0.0 (backwards compat)."""
    buf = GlyphBuffer(3, 3)
    buf.put_char(1, 1, ord("@"), (255, 255, 255, 255), (0, 0, 0, 255))
    assert buf.data[1, 1]["noise"] == 0.0


def test_draw_frame_no_clear():
    buf = GlyphBuffer(10, 10)
    buf.print(3, 4, "test", (255, 255, 255, 255))

    fg = (255, 0, 0, 255)
    bg = (0, 255, 0, 255)
    buf.draw_frame(2, 2, 6, 6, fg, bg, clear=False)

    # Check that the text is still there
    assert "".join(chr(c) for c in buf.data[3:7, 4]["ch"]) == "test"

    # Check that the border was drawn
    assert buf.data[2, 2]["ch"] == 9484  # ┌ Top-left
