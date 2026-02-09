"""Tests for key handling and KeySym conversion.

These tests verify that:
1. The Keys class constants are valid KeySym values
2. GLFW key-to-keysym conversion produces valid KeySyms
3. Key constants match between different parts of the codebase
"""

import pytest

from brileta import input_events
from brileta.input_events import Keys


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
        self, key_name: str, key_value: input_events.KeySym
    ) -> None:
        """Each Keys constant should be a valid, non-zero KeySym."""
        # KeySym(invalid_value) returns 0, so check it's not 0
        assert key_value != 0, (
            f"Keys.{key_name} is 0 (invalid KeySym). "
            f"This likely means input_events.KeySym() was called with an invalid value."
        )
        # Verify it's actually a KeySym instance
        assert isinstance(key_value, input_events.KeySym), (
            f"Keys.{key_name} should be a KeySym, got {type(key_value)}"
        )

    @pytest.mark.parametrize(
        "key_name,expected_keysym",
        [
            ("KEY_H", input_events.KeySym(ord("h"))),
            ("KEY_J", input_events.KeySym(ord("j"))),
            ("KEY_K", input_events.KeySym(ord("k"))),
            ("KEY_L", input_events.KeySym(ord("l"))),
            ("KEY_Q", input_events.KeySym(ord("q"))),
            ("KEY_I", input_events.KeySym(ord("i"))),
            ("KEY_T", input_events.KeySym(ord("t"))),
            ("KEY_R", input_events.KeySym(ord("r"))),
        ],
    )
    def test_key_constant_matches_keysym(
        self, key_name: str, expected_keysym: input_events.KeySym
    ) -> None:
        """Keys constants should match the corresponding input_events.KeySym values."""
        actual = getattr(Keys, key_name)
        assert actual == expected_keysym, (
            f"Keys.{key_name} ({actual}) doesn't match "
            f"input_events.KeySym.{key_name[-1].lower()} ({expected_keysym})"
        )


class TestGlfwKeyConversion:
    """Test GLFW key-to-keysym conversion.

    Note: These tests import the GLFW app module which requires glfw to be available.
    """

    @pytest.fixture
    def glfw_key_converter(self):
        """Get the GLFW key conversion function and glfw module."""
        # Import here to avoid issues if glfw isn't available
        try:
            import glfw

            from brileta.backends.glfw.app import GlfwApp

            # Return the unbound method and glfw module
            return GlfwApp._glfw_key_to_keysym, glfw
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
        assert isinstance(result, input_events.KeySym), (
            f"Converter should return KeySym, got {type(result)}"
        )

    @pytest.mark.parametrize(
        "letter",
        list("abcdefghijklmnopqrstuvwxyz"),
    )
    def test_letter_key_conversion_matches_keysym(
        self, glfw_key_converter, letter: str
    ) -> None:
        """GLFW letter key conversion should match input_events.KeySym values."""
        converter, _glfw = glfw_key_converter
        glfw_key = ord(letter.upper())

        result = converter(None, glfw_key)
        expected = input_events.KeySym(ord(letter))

        assert result == expected, (
            f"GLFW key {letter.upper()} converted to {result}, "
            f"expected input_events.KeySym.{letter} ({expected})"
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
        assert isinstance(result, input_events.KeySym)

    def test_arrow_key_conversion(self, glfw_key_converter) -> None:
        """Arrow keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_UP) == input_events.KeySym.UP
        assert converter(None, glfw.KEY_DOWN) == input_events.KeySym.DOWN
        assert converter(None, glfw.KEY_LEFT) == input_events.KeySym.LEFT
        assert converter(None, glfw.KEY_RIGHT) == input_events.KeySym.RIGHT

    def test_special_key_conversion(self, glfw_key_converter) -> None:
        """Special keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_ESCAPE) == input_events.KeySym.ESCAPE
        assert converter(None, glfw.KEY_ENTER) == input_events.KeySym.RETURN
        assert converter(None, glfw.KEY_SPACE) == input_events.KeySym.SPACE
        assert converter(None, glfw.KEY_TAB) == input_events.KeySym.TAB
        assert converter(None, glfw.KEY_BACKSPACE) == input_events.KeySym.BACKSPACE

    def test_punctuation_key_conversion(self, glfw_key_converter) -> None:
        """Punctuation keys should convert to correct KeySym values."""
        converter, glfw = glfw_key_converter

        assert converter(None, glfw.KEY_COMMA) == input_events.KeySym.COMMA
        assert converter(None, glfw.KEY_PERIOD) == input_events.KeySym.PERIOD
        assert converter(None, glfw.KEY_SLASH) == input_events.KeySym.SLASH
        assert converter(None, glfw.KEY_SEMICOLON) == input_events.KeySym.SEMICOLON
        assert converter(None, glfw.KEY_GRAVE_ACCENT) == input_events.KeySym.GRAVE


class TestKeySymConsistency:
    """Test that KeySym values are consistent across the codebase."""

    def test_keys_class_uses_lowercase_keysyms(self) -> None:
        """Keys class should use lowercase KeySym values (SDL3 standard)."""
        # In SDL3, letter KeySyms are lowercase ASCII values (97-122)
        assert input_events.KeySym(ord("h")) == Keys.KEY_H, (
            "KEY_H should use lowercase 'h'"
        )
        assert input_events.KeySym(ord("i")) == Keys.KEY_I, (
            "KEY_I should use lowercase 'i'"
        )

    def test_keysym_constructor_with_unlisted_value_creates_adhoc_member(self) -> None:
        """Verify that unlisted KeySym values create ad-hoc members via _missing_."""
        # Uppercase ASCII values (65-90) are not named in our KeySym enum
        # but _missing_ creates ad-hoc members for them
        adhoc_keysym = input_events.KeySym(ord("I"))  # uppercase I = 73
        assert adhoc_keysym == 73, (
            "input_events.KeySym(73) should create an ad-hoc member with value 73."
        )
        assert isinstance(adhoc_keysym, input_events.KeySym)

    def test_keysym_constructor_with_valid_value_returns_value(self) -> None:
        """Verify that valid KeySym values are preserved."""
        valid_keysym = input_events.KeySym(ord("i"))  # lowercase i = 105
        assert valid_keysym == ord("i"), (
            "input_events.KeySym(105) should return 105 (valid lowercase 'i')"
        )
        assert valid_keysym == input_events.KeySym(ord("i"))


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
        self, key_const: input_events.KeySym
    ) -> None:
        """KeyDown events should be creatable with Keys class constants."""
        event = input_events.KeyDown(
            sym=key_const,
            scancode=0,
            mod=input_events.Modifier.NONE,
        )
        assert event.sym == key_const
        assert event.sym != 0, "Event sym should not be 0 (invalid)"

    def test_keydown_event_sym_matches_keys_constant(self) -> None:
        """KeyDown event sym should match Keys constants for proper event handling."""
        event = input_events.KeyDown(
            sym=input_events.KeySym(ord("i")),
            scancode=0,
            mod=input_events.Modifier.NONE,
        )
        # This is how the input handler checks for the I key
        assert event.sym == Keys.KEY_I, (
            "KeyDown with KeySym.i should match Keys.KEY_I for input handling"
        )
