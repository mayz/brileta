"""Tests for key handling and KeySym conversion.

These tests verify that:
1. The Keys class constants are valid KeySym values
2. GLFW key-to-tcod conversion produces valid KeySyms
3. Key constants match between different parts of the codebase
"""

import pytest
import tcod.event

from catley.input_handler import Keys


class TestKeysClassValidity:
    """Test that the Keys class constants are valid KeySym values."""

    @pytest.mark.parametrize(
        "key_name,key_value",
        [
            ("KEY_H", Keys.KEY_H),
            ("KEY_J", Keys.KEY_J),
            ("KEY_K", Keys.KEY_K),
            ("KEY_L", Keys.KEY_L),
            ("KEY_Q", Keys.KEY_Q),
            ("KEY_I", Keys.KEY_I),
            ("KEY_T", Keys.KEY_T),
            ("KEY_R", Keys.KEY_R),
        ],
    )
    def test_key_constant_is_valid_keysym(
        self, key_name: str, key_value: tcod.event.KeySym
    ) -> None:
        """Each Keys constant should be a valid, non-zero KeySym."""
        # KeySym(invalid_value) returns 0, so check it's not 0
        assert key_value != 0, (
            f"Keys.{key_name} is 0 (invalid KeySym). "
            f"This likely means tcod.event.KeySym() was called with an invalid value."
        )
        # Verify it's actually a KeySym instance
        assert isinstance(key_value, tcod.event.KeySym), (
            f"Keys.{key_name} should be a KeySym, got {type(key_value)}"
        )

    @pytest.mark.parametrize(
        "key_name,expected_keysym",
        [
            ("KEY_H", tcod.event.KeySym.h),
            ("KEY_J", tcod.event.KeySym.j),
            ("KEY_K", tcod.event.KeySym.k),
            ("KEY_L", tcod.event.KeySym.l),
            ("KEY_Q", tcod.event.KeySym.q),
            ("KEY_I", tcod.event.KeySym.i),
            ("KEY_T", tcod.event.KeySym.t),
            ("KEY_R", tcod.event.KeySym.r),
        ],
    )
    def test_key_constant_matches_tcod_keysym(
        self, key_name: str, expected_keysym: tcod.event.KeySym
    ) -> None:
        """Keys constants should match the corresponding tcod.event.KeySym values."""
        actual = getattr(Keys, key_name)
        assert actual == expected_keysym, (
            f"Keys.{key_name} ({actual}) doesn't match "
            f"tcod.event.KeySym.{key_name[-1].lower()} ({expected_keysym})"
        )


class TestGlfwKeyConversion:
    """Test GLFW key-to-tcod KeySym conversion.

    Note: These tests import the GLFW app module which requires glfw to be available.
    """

    @pytest.fixture
    def glfw_key_converter(self):
        """Get the GLFW key conversion function and glfw module."""
        # Import here to avoid issues if glfw isn't available
        try:
            import glfw

            from catley.backends.glfw.app import GlfwApp

            # Return the unbound method and glfw module
            return GlfwApp._glfw_key_to_tcod, glfw
        except ImportError:
            pytest.skip("GLFW not available")

    @pytest.mark.parametrize(
        "letter",
        list("abcdefghijklmnopqrstuvwxyz"),
    )
    def test_letter_key_conversion_produces_valid_keysym(
        self, glfw_key_converter, letter: str
    ) -> None:
        """Converting GLFW letter keys should produce valid, non-zero KeySyms."""
        converter, _glfw = glfw_key_converter
        # GLFW key codes for A-Z are 65-90 (same as ASCII uppercase)
        glfw_key = ord(letter.upper())

        # Call the converter (it's an instance method, so pass None as self)
        result = converter(None, glfw_key)

        assert result != 0, (
            f"GLFW key {letter.upper()} (code {glfw_key}) converted to KeySym 0. "
            f"KeySym values for letters should be lowercase ASCII (97-122)."
        )
        assert isinstance(result, tcod.event.KeySym), (
            f"Converter should return KeySym, got {type(result)}"
        )

    @pytest.mark.parametrize(
        "letter",
        list("abcdefghijklmnopqrstuvwxyz"),
    )
    def test_letter_key_conversion_matches_tcod_keysym(
        self, glfw_key_converter, letter: str
    ) -> None:
        """GLFW letter key conversion should match tcod.event.KeySym values."""
        converter, _glfw = glfw_key_converter
        glfw_key = ord(letter.upper())

        result = converter(None, glfw_key)
        expected = getattr(tcod.event.KeySym, letter)

        assert result == expected, (
            f"GLFW key {letter.upper()} converted to {result}, "
            f"expected tcod.event.KeySym.{letter} ({expected})"
        )

    @pytest.mark.parametrize(
        "digit",
        list("0123456789"),
    )
    def test_number_key_conversion_produces_valid_keysym(
        self, glfw_key_converter, digit: str
    ) -> None:
        """Converting GLFW number keys should produce valid KeySyms."""
        converter, _glfw = glfw_key_converter
        glfw_key = ord(digit)

        result = converter(None, glfw_key)

        assert result != 0, f"GLFW key {digit} converted to KeySym 0 (invalid)"
        assert isinstance(result, tcod.event.KeySym)

    def test_arrow_key_conversion(self, glfw_key_converter) -> None:
        """Arrow keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_UP) == tcod.event.KeySym.UP
        assert converter(None, glfw.KEY_DOWN) == tcod.event.KeySym.DOWN
        assert converter(None, glfw.KEY_LEFT) == tcod.event.KeySym.LEFT
        assert converter(None, glfw.KEY_RIGHT) == tcod.event.KeySym.RIGHT

    def test_special_key_conversion(self, glfw_key_converter) -> None:
        """Special keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_ESCAPE) == tcod.event.KeySym.ESCAPE
        assert converter(None, glfw.KEY_ENTER) == tcod.event.KeySym.RETURN
        assert converter(None, glfw.KEY_SPACE) == tcod.event.KeySym.SPACE
        assert converter(None, glfw.KEY_TAB) == tcod.event.KeySym.TAB
        assert converter(None, glfw.KEY_BACKSPACE) == tcod.event.KeySym.BACKSPACE

    def test_punctuation_key_conversion(self, glfw_key_converter) -> None:
        """Punctuation keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_COMMA) == tcod.event.KeySym.COMMA
        assert converter(None, glfw.KEY_PERIOD) == tcod.event.KeySym.PERIOD
        assert converter(None, glfw.KEY_SLASH) == tcod.event.KeySym.SLASH
        assert converter(None, glfw.KEY_SEMICOLON) == tcod.event.KeySym.SEMICOLON
        assert converter(None, glfw.KEY_GRAVE_ACCENT) == tcod.event.KeySym.GRAVE


class TestKeySymConsistency:
    """Test that KeySym values are consistent across the codebase."""

    def test_keys_class_uses_lowercase_keysyms(self) -> None:
        """Keys class should use lowercase KeySym values (SDL3/tcod 19 standard)."""
        # In SDL3/tcod 19, letter KeySyms are lowercase ASCII values (97-122)
        assert tcod.event.KeySym(ord("h")) == Keys.KEY_H, (
            "KEY_H should use lowercase 'h'"
        )
        assert tcod.event.KeySym(ord("i")) == Keys.KEY_I, (
            "KEY_I should use lowercase 'i'"
        )

    def test_keysym_constructor_with_invalid_value_returns_zero(self) -> None:
        """Verify that invalid KeySym values return 0 (our detection mechanism)."""
        # This documents the behavior we rely on to detect invalid KeySyms
        # Uppercase ASCII values (65-90) are NOT valid KeySym values in tcod 19
        invalid_keysym = tcod.event.KeySym(ord("I"))  # uppercase I = 73
        assert invalid_keysym == 0, (
            "tcod.event.KeySym(73) should return 0 (invalid). "
            "If this fails, the KeySym validation mechanism has changed."
        )

    def test_keysym_constructor_with_valid_value_returns_value(self) -> None:
        """Verify that valid KeySym values are preserved."""
        valid_keysym = tcod.event.KeySym(ord("i"))  # lowercase i = 105
        assert valid_keysym == ord("i"), (
            "tcod.event.KeySym(105) should return 105 (valid lowercase 'i')"
        )
        assert valid_keysym == tcod.event.KeySym.i


class TestKeyEventCreation:
    """Test that KeyDown/KeyUp events can be created with our KeySym values."""

    @pytest.mark.parametrize(
        "key_const",
        [
            Keys.KEY_H,
            Keys.KEY_J,
            Keys.KEY_K,
            Keys.KEY_L,
            Keys.KEY_Q,
            Keys.KEY_I,
            Keys.KEY_T,
            Keys.KEY_R,
        ],
    )
    def test_can_create_keydown_event_with_keys_constant(
        self, key_const: tcod.event.KeySym
    ) -> None:
        """KeyDown events should be creatable with Keys class constants."""
        event = tcod.event.KeyDown(
            sym=key_const,
            scancode=0,
            mod=tcod.event.Modifier.NONE,
        )
        assert event.sym == key_const
        assert event.sym != 0, "Event sym should not be 0 (invalid)"

    def test_keydown_event_sym_matches_keys_constant(self) -> None:
        """KeyDown event sym should match Keys constants for proper event handling."""
        event = tcod.event.KeyDown(
            sym=tcod.event.KeySym.i,
            scancode=0,
            mod=tcod.event.Modifier.NONE,
        )
        # This is how the input handler checks for the I key
        assert event.sym == Keys.KEY_I, (
            "KeyDown with KeySym.i should match Keys.KEY_I for input handling"
        )
