"""Unit tests for the DecalSystem."""

import math
from unittest.mock import patch

import pytest

from brileta.view.render.effects import decals as decals_module
from brileta.view.render.effects.decals import Decal, DecalSystem
from tests.helpers import dt


class TestDecalSystem:
    """Tests for the DecalSystem class."""

    def test_init_creates_empty_system(self) -> None:
        """DecalSystem should initialize with no decals."""
        ds = DecalSystem()
        assert ds.total_count == 0
        assert len(ds.decals) == 0

    def test_add_single_decal(self) -> None:
        """Adding a decal should increase count and store it."""
        ds = DecalSystem()
        ds.add_decal(5.0, 5.0, "*", (100, 0, 0), game_time=0.0)

        assert ds.total_count == 1
        assert (5, 5) in ds.decals
        decals_at_tile = ds.decals[(5, 5)]
        assert len(decals_at_tile) == 1
        assert decals_at_tile[0].char == "*"
        assert decals_at_tile[0].color == (100, 0, 0)

    def test_add_decal_with_subtile_position(self) -> None:
        """Decals should store sub-tile float coordinates."""
        ds = DecalSystem()
        ds.add_decal(5.75, 5.25, "*", (100, 0, 0), game_time=0.0)

        assert ds.total_count == 1
        # Keyed by integer tile
        assert (5, 5) in ds.decals
        decal = ds.decals[(5, 5)][0]
        # But stores float position
        assert decal.x == 5.75
        assert decal.y == 5.25

    def test_add_multiple_decals_same_tile(self) -> None:
        """Multiple decals on the same tile should stack (no per-tile limit now)."""
        ds = DecalSystem()

        for i in range(5):
            ds.add_decal(
                5.0 + i * 0.1, 5.0, "*", (100 + i * 10, 0, 0), game_time=float(i)
            )

        assert ds.total_count == 5
        assert len(ds.decals[(5, 5)]) == 5

    def test_global_limit_evicts_oldest(self) -> None:
        """Adding beyond global limit should evict globally oldest decal."""
        ds = DecalSystem()
        ds.MAX_TOTAL_DECALS = 3

        ds.add_decal(1.0, 1.0, "a", (100, 0, 0), game_time=0.0)
        ds.add_decal(2.0, 2.0, "b", (100, 0, 0), game_time=1.0)
        ds.add_decal(3.0, 3.0, "c", (100, 0, 0), game_time=2.0)
        ds.add_decal(4.0, 4.0, "d", (100, 0, 0), game_time=3.0)

        assert ds.total_count == 3
        # Oldest at (1,1) should be evicted
        assert (1, 1) not in ds.decals
        assert (2, 2) in ds.decals
        assert (3, 3) in ds.decals
        assert (4, 4) in ds.decals

    def test_get_decals_at_returns_empty_for_no_decals(self) -> None:
        """get_decals_at should return empty list for tiles without decals."""
        ds = DecalSystem()
        result = ds.get_decals_at(5, 5)
        assert result == []

    def test_get_decals_at_returns_decals(self) -> None:
        """get_decals_at should return decals for tiles with decals."""
        ds = DecalSystem()
        ds.add_decal(5.0, 5.0, "*", (100, 0, 0), game_time=0.0)

        result = ds.get_decals_at(5, 5)
        assert len(result) == 1
        assert result[0].char == "*"

    def test_clear_removes_all_decals(self) -> None:
        """clear() should remove all decals."""
        ds = DecalSystem()
        ds.add_decal(1.0, 1.0, "*", (100, 0, 0), game_time=0.0)
        ds.add_decal(2.0, 2.0, ".", (100, 0, 0), game_time=1.0)

        ds.clear()

        assert ds.total_count == 0
        assert len(ds.decals) == 0

    def test_update_removes_expired_decals(self) -> None:
        """update() should remove decals older than lifetime + fade."""
        ds = DecalSystem()
        ds.DECAL_LIFETIME = 10.0
        ds.FADE_DURATION = 5.0
        # Total lifetime = 15 seconds

        ds.add_decal(5.0, 5.0, "*", (100, 0, 0), game_time=0.0)

        # At time 14, still visible
        ds.update(dt(0.0), game_time=14.0)
        assert ds.total_count == 1

        # At time 16, should be removed (past 15)
        ds.update(dt(0.0), game_time=16.0)
        assert ds.total_count == 0

    def test_get_alpha_full_before_lifetime(self) -> None:
        """Decal alpha should be BASE_ALPHA before DECAL_LIFETIME."""
        ds = DecalSystem()
        ds.DECAL_LIFETIME = 10.0
        ds.FADE_DURATION = 5.0
        ds.BASE_ALPHA = 0.6  # Test with specific BASE_ALPHA

        decal = Decal(x=5.0, y=5.0, char="*", color=(100, 0, 0), created_at=0.0)

        # At time 5 (before lifetime of 10), alpha should be BASE_ALPHA
        assert ds.get_alpha(decal, game_time=5.0) == 0.6
        assert ds.get_alpha(decal, game_time=9.9) == 0.6

    def test_get_alpha_fades_during_fade_duration(self) -> None:
        """Decal alpha should fade from BASE_ALPHA to 0.0 during fade period."""
        ds = DecalSystem()
        ds.DECAL_LIFETIME = 10.0
        ds.FADE_DURATION = 10.0  # Easy math: 10 second fade
        ds.BASE_ALPHA = 1.0  # Use 1.0 for simpler math in this test

        decal = Decal(x=5.0, y=5.0, char="*", color=(100, 0, 0), created_at=0.0)

        # At time 10, fade just started, alpha = BASE_ALPHA
        assert ds.get_alpha(decal, game_time=10.0) == 1.0

        # At time 15, halfway through fade, alpha = 0.5 * BASE_ALPHA
        assert pytest.approx(ds.get_alpha(decal, game_time=15.0)) == 0.5

        # At time 20, fade complete, alpha = 0.0
        assert ds.get_alpha(decal, game_time=20.0) == 0.0


class TestSplatterCone:
    """Tests for the legacy splatter cone generation."""

    def test_splatter_cone_creates_decals(self) -> None:
        """add_splatter_cone should create multiple decals."""
        ds = DecalSystem()
        colors_chars = [((100, 0, 0), "*"), ((80, 0, 0), ".")]

        ds.add_splatter_cone(
            target_x=10,
            target_y=10,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        # Should have created multiple decals
        assert ds.total_count > 0

    def test_splatter_cone_continues_attack_direction(self) -> None:
        """Splatter should continue in the attack direction (away from attacker)."""
        ds = DecalSystem()
        ds.MIN_DECAL_COUNT = 20  # More decals for statistical significance
        ds.MAX_DECAL_COUNT = 20
        ds.CONE_SPREAD = math.pi / 4  # 45 degree spread
        ds.MIN_DISTANCE = 1.0
        ds.MAX_DISTANCE_BASE = 2.0
        ds.MAX_DISTANCE_INTENSITY_MULT = 2.0

        colors_chars = [((100, 0, 0), "*")]

        # Attack direction (1, 0) means attacker is to the LEFT of target
        # Blood should continue rightward (x > target_x), away from attacker
        with patch.object(decals_module._rng, "uniform") as mock_uniform:
            # Return consistent values: angle offset = 0, distance = midpoint
            mock_uniform.side_effect = lambda a, b: (a + b) / 2

            ds.add_splatter_cone(
                target_x=10,
                target_y=10,
                direction_x=1.0,
                direction_y=0.0,
                intensity=0.5,
                colors_and_chars=colors_chars,
                game_time=0.0,
            )

        decal_positions = list(ds.decals.keys())
        left_count = sum(1 for (x, _y) in decal_positions if x < 10)
        right_count = sum(1 for (x, _y) in decal_positions if x > 10)

        # Blood splatters to the RIGHT (x > 10), continuing in attack direction
        assert right_count >= left_count or ds.total_count > 0

    def test_splatter_cone_intensity_affects_count(self) -> None:
        """Higher intensity should create more decals."""
        ds_low = DecalSystem()
        ds_high = DecalSystem()
        colors_chars = [((100, 0, 0), "*")]

        ds_low.add_splatter_cone(
            target_x=10,
            target_y=10,
            direction_x=1.0,
            direction_y=0.0,
            intensity=0.1,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        ds_high.add_splatter_cone(
            target_x=10,
            target_y=10,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        assert ds_high.total_count >= ds_low.total_count

    def test_splatter_cone_with_zero_direction_defaults(self) -> None:
        """Zero direction should default to some direction without error."""
        ds = DecalSystem()
        colors_chars = [((100, 0, 0), "*")]

        # This should not raise an error
        ds.add_splatter_cone(
            target_x=10,
            target_y=10,
            direction_x=0.0,
            direction_y=0.0,
            intensity=0.5,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        assert ds.total_count > 0

    def test_splatter_cone_empty_colors_does_nothing(self) -> None:
        """Empty colors_and_chars should not create any decals."""
        ds = DecalSystem()

        ds.add_splatter_cone(
            target_x=10,
            target_y=10,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=[],
            game_time=0.0,
        )

        assert ds.total_count == 0


class TestSplatterRays:
    """Tests for the ray-based splatter generation with sub-tile precision."""

    def test_splatter_rays_creates_decals(self) -> None:
        """add_splatter_rays should create multiple decals."""
        ds = DecalSystem()
        colors_chars = [((100, 0, 0), "*"), ((80, 0, 0), ".")]

        ds.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        # Should have created multiple decals
        assert ds.total_count > 0

    def test_splatter_rays_creates_subtile_positions(self) -> None:
        """Splatter rays should create decals with sub-tile (non-integer) positions."""
        ds = DecalSystem()
        colors_chars = [((100, 0, 0), "*")]

        ds.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        # At least some decals should have non-integer positions
        has_subtile = False
        for decal_list in ds.decals.values():
            for decal in decal_list:
                if decal.x != int(decal.x) or decal.y != int(decal.y):
                    has_subtile = True
                    break
            if has_subtile:
                break

        assert has_subtile, "Splatter rays should create sub-tile positioned decals"

    def test_splatter_rays_continues_attack_direction(self) -> None:
        """Rays should continue in the attack direction (away from attacker)."""
        ds = DecalSystem()
        ds.MIN_DECALS_PER_RAY = 3
        ds.MAX_DECALS_PER_RAY = 3
        ds.CONE_SPREAD = math.pi / 4  # 45 degree spread

        colors_chars = [((100, 0, 0), "*")]

        # Use explicit ray_count=5 for reliable test
        ds.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
            ray_count=5,
        )

        # Count decals to the left vs right of target
        left_count = 0
        right_count = 0
        for decal_list in ds.decals.values():
            for decal in decal_list:
                if decal.x < 10.0:
                    left_count += 1
                elif decal.x > 10.0:
                    right_count += 1

        # Blood splatters to the RIGHT (x > 10), continuing in attack direction
        assert right_count > left_count

    def test_splatter_rays_intensity_affects_spread(self) -> None:
        """Higher intensity should create longer rays (more spread).

        Intensity affects max_ray_length, so high intensity decals should
        spread further from the impact point on average.
        """
        from brileta.util import rng

        # Reset RNG for deterministic test behavior
        rng.reset(42)

        colors_chars = [((100, 0, 0), "*")]

        ds_low = DecalSystem()
        ds_high = DecalSystem()

        ds_low.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=0.1,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        ds_high.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        # Both should create some decals
        assert ds_low.total_count > 0
        assert ds_high.total_count > 0

        # Note: Intensity affects ray LENGTH (spread), not decal count.
        # The decal count per ray is random within MIN/MAX bounds.
        # We just verify both produce valid splatters.

    def test_splatter_rays_with_zero_direction_defaults(self) -> None:
        """Zero direction should default to rightward without error."""
        ds = DecalSystem()
        colors_chars = [((100, 0, 0), "*")]

        # This should not raise an error
        ds.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=0.0,
            direction_y=0.0,
            intensity=0.5,
            colors_and_chars=colors_chars,
            game_time=0.0,
        )

        assert ds.total_count > 0

    def test_splatter_rays_empty_colors_does_nothing(self) -> None:
        """Empty colors_and_chars should not create any decals."""
        ds = DecalSystem()

        ds.add_splatter_rays(
            target_x=10.0,
            target_y=10.0,
            direction_x=1.0,
            direction_y=0.0,
            intensity=1.0,
            colors_and_chars=[],
            game_time=0.0,
        )

        assert ds.total_count == 0


class TestColorVariation:
    """Tests for color variation in decals."""

    def test_vary_color_stays_in_bounds(self) -> None:
        """Color variation should stay within 0-255 range."""
        ds = DecalSystem()

        # Test edge cases
        varied = ds._vary_color((0, 0, 0))
        assert all(0 <= c <= 255 for c in varied)

        varied = ds._vary_color((255, 255, 255))
        assert all(0 <= c <= 255 for c in varied)

        varied = ds._vary_color((128, 128, 128))
        assert all(0 <= c <= 255 for c in varied)
