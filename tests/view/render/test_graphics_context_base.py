"""Tests for default no-op helpers on GraphicsContext."""

from __future__ import annotations

from typing import cast

from brileta.view.render.graphics import GraphicsContext


def test_set_atmospheric_layer_accepts_affects_foreground_kwarg() -> None:
    """Base method should accept new keyword args without requiring overrides."""
    result = GraphicsContext.set_atmospheric_layer(
        cast(GraphicsContext, object()),
        viewport_offset=(0, 0),
        viewport_size=(1, 1),
        map_size=(1, 1),
        sky_exposure_threshold=0.5,
        sky_exposure_texture=None,
        explored_texture=None,
        visible_texture=None,
        roof_surface_mask_buffer=None,
        noise_scale=0.1,
        noise_threshold_low=0.2,
        noise_threshold_high=0.8,
        strength=0.2,
        tint_color=(10, 20, 30),
        drift_offset=(0.0, 0.0),
        turbulence_offset=0.0,
        turbulence_strength=0.0,
        turbulence_scale=0.1,
        blend_mode="darken",
        pixel_bounds=(0, 0, 1, 1),
        affects_foreground=False,
    )

    assert result is None
