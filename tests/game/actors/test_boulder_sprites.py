"""Tests for the procedural boulder sprite generator."""

from __future__ import annotations

import numpy as np

from brileta.sprites.boulders import (
    BoulderArchetype,
    archetype_for_position,
    generate_boulder_sprite,
    generate_boulder_sprite_for_position,
    generate_preview_sheet,
    shadow_height_for_archetype,
)


class TestBoulderArchetype:
    """Tests for boulder archetype selection."""

    def test_deterministic_for_same_position(self) -> None:
        a = archetype_for_position(5, 3, 42)
        b = archetype_for_position(5, 3, 42)
        assert a == b

    def test_varies_across_positions(self) -> None:
        """Many positions should produce more than one archetype."""
        archetypes = {archetype_for_position(x, 0, 42) for x in range(40)}
        assert len(archetypes) >= 2

    def test_all_archetypes_appear(self) -> None:
        """Over enough samples, every archetype should appear at least once."""
        archetypes = {
            archetype_for_position(x, y, 42) for x in range(20) for y in range(20)
        }
        assert archetypes == set(BoulderArchetype)


class TestShadowHeight:
    """Tests for per-archetype shadow heights."""

    def test_all_archetypes_have_shadow_height(self) -> None:
        for archetype in BoulderArchetype:
            h = shadow_height_for_archetype(archetype)
            assert isinstance(h, int)
            assert h >= 1

    def test_tall_is_tallest(self) -> None:
        assert shadow_height_for_archetype(
            BoulderArchetype.TALL
        ) > shadow_height_for_archetype(BoulderArchetype.ROUNDED)

    def test_flat_is_shortest(self) -> None:
        assert shadow_height_for_archetype(
            BoulderArchetype.FLAT
        ) < shadow_height_for_archetype(BoulderArchetype.ROUNDED)


class TestGenerateBoulderSprite:
    """Core tests for boulder sprite generation."""

    def test_returns_rgba_uint8_array(self) -> None:
        sprite = generate_boulder_sprite(42, BoulderArchetype.ROUNDED)
        assert sprite.ndim == 3
        assert sprite.shape[2] == 4
        assert sprite.dtype == np.uint8

    def test_produces_visible_pixels(self) -> None:
        sprite = generate_boulder_sprite(42, BoulderArchetype.ROUNDED)
        assert sprite[:, :, 3].any()

    def test_deterministic_from_seed(self) -> None:
        a = generate_boulder_sprite(42, BoulderArchetype.ROUNDED)
        b = generate_boulder_sprite(42, BoulderArchetype.ROUNDED)
        np.testing.assert_array_equal(a, b)

    def test_each_archetype_produces_visible_pixels(self) -> None:
        for archetype in BoulderArchetype:
            sprite = generate_boulder_sprite(99, archetype)
            assert sprite[:, :, 3].any(), (
                f"{archetype.value} produced no visible pixels"
            )


class TestPositionSeededGeneration:
    """Tests for position/map-seeded boulder generation helpers."""

    def test_generate_for_position_deterministic(self) -> None:
        a = generate_boulder_sprite_for_position(7, 9, 123, BoulderArchetype.FLAT)
        b = generate_boulder_sprite_for_position(7, 9, 123, BoulderArchetype.FLAT)
        np.testing.assert_array_equal(a, b)

    def test_generate_for_position_varies_by_map_seed(self) -> None:
        a = generate_boulder_sprite_for_position(7, 9, 123, BoulderArchetype.ROUNDED)
        b = generate_boulder_sprite_for_position(7, 9, 124, BoulderArchetype.ROUNDED)
        assert not np.array_equal(a, b)

    def test_different_archetypes_differ(self) -> None:
        a = generate_boulder_sprite_for_position(3, 4, 42, BoulderArchetype.ROUNDED)
        b = generate_boulder_sprite_for_position(3, 4, 42, BoulderArchetype.TALL)
        assert not np.array_equal(a, b)


class TestPreviewSheet:
    """Tests for the boulder preview-sheet utility."""

    def test_returns_rgba_array(self) -> None:
        sheet = generate_preview_sheet(columns=4, rows=2, base_size=14)
        assert sheet.ndim == 3
        assert sheet.shape[2] == 4
        assert sheet.dtype == np.uint8

    def test_sheet_has_visible_content(self) -> None:
        sheet = generate_preview_sheet(columns=4, rows=2, base_size=14)
        non_bg = np.any(sheet[:, :, :3] != 30, axis=2)
        assert non_bg.sum() > 100
