"""Tests for the idle animation system.

These tests verify that:
1. IdleAnimationProfile correctly configures animation parameters
2. CreatureSize-based factory creates appropriate profiles
3. VisualEffectsComponent correctly applies drift
4. Characters are created with proper idle animation profiles
"""

import pytest

from catley.game.actors.components import VisualEffectsComponent
from catley.game.actors.idle_animation import (
    PROFILE_MECHANICAL,
    PROFILE_STATIC,
    IdleAnimationProfile,
    create_profile_for_size,
)
from catley.game.enums import CreatureSize


class TestIdleAnimationProfile:
    """Tests for IdleAnimationProfile configuration."""

    def test_default_profile_has_sensible_values(self):
        """Default profile should have human-baseline values."""
        profile = IdleAnimationProfile()

        # Weight-shifting drift is enabled for subtle movement
        assert profile.drift_enabled is True
        assert profile.drift_amplitude == pytest.approx(0.025)
        assert profile.drift_speed == pytest.approx(0.3)

    def test_profile_can_disable_drift(self):
        """Profile should allow disabling drift."""
        profile = IdleAnimationProfile(drift_enabled=False)

        assert profile.drift_enabled is False


class TestCreatureSizeProfiles:
    """Tests for size-based profile creation."""

    def test_medium_creatures_use_defaults(self):
        """Medium creatures should use default human values."""
        profile = create_profile_for_size(CreatureSize.MEDIUM)

        # Medium creatures use the base defaults
        assert profile.drift_amplitude == pytest.approx(0.025)
        assert profile.drift_speed == pytest.approx(0.3)

    def test_size_affects_drift_speed(self):
        """Larger creatures should have slower drift than medium-sized ones."""
        medium = create_profile_for_size(CreatureSize.MEDIUM)
        large = create_profile_for_size(CreatureSize.LARGE)
        huge = create_profile_for_size(CreatureSize.HUGE)

        # Medium is baseline, larger creatures move more slowly
        assert medium.drift_speed > large.drift_speed > huge.drift_speed

    def test_size_affects_drift_amplitude(self):
        """Larger creatures should have larger drift amplitude (wider stance)."""
        tiny = create_profile_for_size(CreatureSize.TINY)
        medium = create_profile_for_size(CreatureSize.MEDIUM)
        huge = create_profile_for_size(CreatureSize.HUGE)

        # Larger creatures cover more ground with their weight shifts
        assert tiny.drift_amplitude < medium.drift_amplitude < huge.drift_amplitude


class TestPresetProfiles:
    """Tests for preset animation profiles."""

    def test_mechanical_profile_has_minimal_drift(self):
        """Mechanical entities should have minimal, slow drift."""
        assert PROFILE_MECHANICAL.drift_enabled is True
        assert PROFILE_MECHANICAL.drift_amplitude < 0.01  # Very small
        assert PROFILE_MECHANICAL.drift_speed < 0.1  # Very slow

    def test_static_profile_disables_drift(self):
        """Static profile should disable all animations."""
        assert PROFILE_STATIC.drift_enabled is False


class TestVisualEffectsIdleAnimation:
    """Tests for idle animation in VisualEffectsComponent."""

    def test_default_profile_created_lazily(self):
        """Component should create default profile on first access."""
        component = VisualEffectsComponent()

        # Access triggers lazy creation
        profile = component.idle_profile

        assert profile is not None
        assert profile.drift_enabled is True

    def test_custom_profile_used(self):
        """Custom profile should be used when provided."""
        custom_profile = IdleAnimationProfile(drift_amplitude=0.1)
        component = VisualEffectsComponent(idle_profile=custom_profile)

        assert component.idle_profile.drift_amplitude == pytest.approx(0.1)

    def test_drift_offset_returns_zero_when_disabled(self):
        """Drift should return (0, 0) when disabled."""
        profile = IdleAnimationProfile(drift_enabled=False)
        component = VisualEffectsComponent(idle_profile=profile)

        offset = component.get_idle_drift_offset()

        assert offset == (0.0, 0.0)

    def test_drift_offset_varies_over_time(self):
        """Drift should change as time passes."""
        component = VisualEffectsComponent()

        offset1 = component.get_idle_drift_offset()
        component.update(0.5)  # Advance 500ms
        offset2 = component.get_idle_drift_offset()

        assert offset1 != offset2

    def test_drift_offset_bounded_by_amplitude(self):
        """Drift should never exceed the configured amplitude."""
        profile = IdleAnimationProfile(drift_amplitude=0.05)
        component = VisualEffectsComponent(idle_profile=profile)

        # Sample many time points
        for _ in range(100):
            component.update(0.1)
            x, y = component.get_idle_drift_offset()
            assert abs(x) <= 0.05
            assert abs(y) <= 0.05

    def test_actors_desynchronized(self):
        """Different components should have different phases."""
        comp1 = VisualEffectsComponent()
        comp2 = VisualEffectsComponent()

        # Random offsets mean they shouldn't be in sync
        # (This test may occasionally fail due to randomness, but very unlikely)
        assert comp1._idle_timer_offset != comp2._idle_timer_offset

    def test_set_idle_profile_changes_behavior(self):
        """Changing profile should affect animation behavior."""
        component = VisualEffectsComponent()

        # Start with default (drift enabled)
        profile = component.idle_profile
        assert profile.drift_enabled is True

        # Change to static profile
        component.set_idle_profile(PROFILE_STATIC)

        assert component.idle_profile.drift_enabled is False

    def test_idle_animation_uses_real_time(self):
        """Idle animation should use real clock time for consistent speed.

        This ensures animation speed doesn't depend on how often get_idle_drift_offset()
        is called or the delta_time passed to update().
        """
        import time

        component = VisualEffectsComponent()

        # Get initial offset
        offset1 = component.get_idle_drift_offset()

        # Wait a bit and sample again
        time.sleep(0.05)
        offset2 = component.get_idle_drift_offset()

        # Should have moved due to real time passing
        assert offset1 != offset2
