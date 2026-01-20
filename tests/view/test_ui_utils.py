"""Tests for UI utility functions."""

from __future__ import annotations

from catley.view.ui.ui_utils import wrap_text_by_words


class TestWrapTextByWords:
    """Tests for the wrap_text_by_words function."""

    def test_short_text_no_wrap(self) -> None:
        """Text that fits should not be wrapped."""
        result = wrap_text_by_words("Hello", lambda s: len(s) <= 20)
        assert result == ["Hello"]

    def test_wraps_at_word_boundary(self) -> None:
        """Text should wrap at word boundaries, not mid-word."""
        result = wrap_text_by_words("Hello world", lambda s: len(s) <= 6)
        assert result == ["Hello", "world"]

    def test_preserves_word_integrity(self) -> None:
        """Should NOT break words in the middle (e.g., 'reliable' into 'reli')."""
        text = "Basic but reliable protection"
        result = wrap_text_by_words(text, lambda s: len(s) <= 15)
        # "Basic but" fits (9 chars), "reliable" starts new line
        # "reliable" fits (8 chars), "protection" starts new line
        for line in result:
            # No line should contain a partial word like "reli" without full "reliable"
            assert "reli" not in line or "reliable" in line

    def test_multiple_wraps(self) -> None:
        """Long text should wrap into multiple lines."""
        text = "This is a longer text that needs multiple wraps"
        result = wrap_text_by_words(text, lambda s: len(s) <= 15)
        assert len(result) > 1
        # Each line should fit within the constraint
        for line in result:
            assert len(line) <= 15 or len(line.split()) == 1

    def test_empty_text(self) -> None:
        """Empty text should return a list with an empty string."""
        result = wrap_text_by_words("", lambda s: True)
        assert result == [""]

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return appropriate result."""
        result = wrap_text_by_words("   ", lambda s: True)
        # split() on whitespace returns empty list, so we return the original
        assert result == ["   "]

    def test_single_long_word(self) -> None:
        """A single word that exceeds the limit should still be returned."""
        result = wrap_text_by_words(
            "Supercalifragilisticexpialidocious", lambda s: len(s) <= 10
        )
        # Word is kept intact even though it exceeds limit
        assert result == ["Supercalifragilisticexpialidocious"]

    def test_pixel_based_width_check(self) -> None:
        """Works with pixel-based width checking (simulated)."""
        # Simulate a font where each char is 8 pixels wide
        char_width = 8
        max_pixels = 80  # 10 chars

        def fits(s: str) -> bool:
            return len(s) * char_width <= max_pixels

        result = wrap_text_by_words("Hello world test", fits)
        # "Hello" = 5 chars, "world" = 5 chars, together = 11 chars > 10
        assert result == ["Hello", "world test"]

    def test_exact_fit(self) -> None:
        """Text that exactly fits should not be wrapped."""
        result = wrap_text_by_words("Hello", lambda s: len(s) <= 5)
        assert result == ["Hello"]

    def test_preserves_single_spaces(self) -> None:
        """Single spaces between words should be preserved in output."""
        text = "one two three"
        result = wrap_text_by_words(text, lambda s: len(s) <= 7)
        # "one two" = 7 chars, fits exactly
        assert result[0] == "one two"


class TestWrapTextByWordsEdgeCases:
    """Edge case tests for wrap_text_by_words."""

    def test_always_fits_returns_single_line(self) -> None:
        """If everything fits, return single line."""
        result = wrap_text_by_words("Any text here", lambda s: True)
        assert result == ["Any text here"]

    def test_nothing_fits_returns_words_on_separate_lines(self) -> None:
        """If nothing fits, each word goes on its own line."""
        result = wrap_text_by_words("a b c", lambda s: len(s) <= 1)
        assert result == ["a", "b", "c"]
