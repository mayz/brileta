"""Tests for the visual_scale rendering feature.

These tests verify that:
1. scale_for_size() correctly maps CreatureSize to scale factors
2. HUGE raises NotImplementedError (reserved for multi-tile footprint)
3. Actor defaults to visual_scale=1.0
4. Character derives visual_scale from creature_size
5. Explicit visual_scale overrides size-based defaults
"""

import pytest

from catley import colors
from catley.game.actors.core import Actor, Character
from catley.game.actors.idle_animation import scale_for_size
from catley.game.enums import CreatureSize


class TestScaleForSize:
    """Tests for the scale_for_size factory function."""

    def test_medium_is_baseline(self):
        """Medium creatures should have scale 1.0."""
        assert scale_for_size(CreatureSize.MEDIUM) == 1.0

    def test_larger_sizes_have_larger_scales(self):
        """Larger creatures should have larger visual_scale values."""
        tiny = scale_for_size(CreatureSize.TINY)
        small = scale_for_size(CreatureSize.SMALL)
        medium = scale_for_size(CreatureSize.MEDIUM)
        large = scale_for_size(CreatureSize.LARGE)

        assert tiny < small < medium < large

    def test_all_scales_are_positive(self):
        """All scale values should be positive."""
        for size in [
            CreatureSize.TINY,
            CreatureSize.SMALL,
            CreatureSize.MEDIUM,
            CreatureSize.LARGE,
        ]:
            assert scale_for_size(size) > 0

    def test_huge_raises_not_implemented(self):
        """HUGE should raise NotImplementedError (requires multi-tile footprint)."""
        with pytest.raises(NotImplementedError) as exc_info:
            scale_for_size(CreatureSize.HUGE)

        # Error message should explain the limitation
        assert "multi-tile" in str(exc_info.value).lower()
        assert "tile_footprint" in str(exc_info.value)

    def test_expected_scale_values(self):
        """Scale values should match the design specification."""
        assert scale_for_size(CreatureSize.TINY) == pytest.approx(0.6)
        assert scale_for_size(CreatureSize.SMALL) == pytest.approx(0.8)
        assert scale_for_size(CreatureSize.MEDIUM) == pytest.approx(1.0)
        assert scale_for_size(CreatureSize.LARGE) == pytest.approx(1.3)


class TestActorVisualScale:
    """Tests for visual_scale on Actor class."""

    def test_actor_default_visual_scale(self):
        """Actor should default to visual_scale=1.0."""
        actor = Actor(0, 0, "A", colors.WHITE, name="Test")
        assert actor.visual_scale == 1.0

    def test_actor_custom_visual_scale(self):
        """Actor should accept custom visual_scale."""
        actor = Actor(0, 0, "A", colors.WHITE, name="Test", visual_scale=1.5)
        assert actor.visual_scale == 1.5

    def test_actor_small_visual_scale(self):
        """Actor should accept visual_scale less than 1.0."""
        actor = Actor(0, 0, "A", colors.WHITE, name="Test", visual_scale=0.5)
        assert actor.visual_scale == 0.5


class TestCharacterVisualScale:
    """Tests for visual_scale on Character class."""

    def test_character_derives_scale_from_size(self):
        """Character should derive visual_scale from creature_size."""
        large_char = Character(
            0, 0, "L", colors.WHITE, "Large", creature_size=CreatureSize.LARGE
        )
        expected = scale_for_size(CreatureSize.LARGE)
        assert large_char.visual_scale == pytest.approx(expected)

    def test_character_explicit_scale_overrides_size(self):
        """Explicit visual_scale should override size-based default."""
        char = Character(
            0,
            0,
            "X",
            colors.WHITE,
            "Custom",
            creature_size=CreatureSize.TINY,
            visual_scale=2.0,  # Override the TINY default
        )
        assert char.visual_scale == 2.0

    def test_medium_character_has_unit_scale(self):
        """Medium-sized character should have visual_scale=1.0."""
        char = Character(0, 0, "M", colors.WHITE, "Medium")
        assert char.visual_scale == 1.0

    def test_small_character_has_small_scale(self):
        """Small-sized character should have visual_scale < 1.0."""
        char = Character(
            0, 0, "S", colors.WHITE, "Small", creature_size=CreatureSize.SMALL
        )
        assert char.visual_scale < 1.0
        assert char.visual_scale == pytest.approx(0.8)

    def test_large_character_has_large_scale(self):
        """Large-sized character should have visual_scale > 1.0."""
        char = Character(
            0, 0, "L", colors.WHITE, "Large", creature_size=CreatureSize.LARGE
        )
        assert char.visual_scale > 1.0
        assert char.visual_scale == pytest.approx(1.3)
