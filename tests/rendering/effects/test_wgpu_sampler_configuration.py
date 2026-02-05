"""Tests for WGPU sampler configuration.

These tests verify that texture samplers are configured correctly for the
lighting system, particularly that sky exposure uses nearest-neighbor filtering
to prevent bleeding at tile boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestWGPUSamplerConfiguration:
    """Test suite for WGPU sampler configuration."""

    def test_sky_exposure_sampler_uses_nearest_filtering(self) -> None:
        """Test that sky exposure sampler uses nearest filtering.

        Nearest filtering is required to prevent interpolation bleeding at
        tile boundaries - walls should have a sharp sky exposure cutoff.
        """
        import wgpu

        # Mock the device to capture sampler creation args
        mock_device = MagicMock()
        mock_sampler = MagicMock()
        mock_device.create_sampler.return_value = mock_sampler

        # Import and create a partial system to test _create_samplers
        from catley.backends.wgpu.gpu_lighting import GPULightingSystem

        system = object.__new__(GPULightingSystem)
        system.device = mock_device

        # Call the method under test
        system._create_samplers()

        # Verify create_sampler was called with nearest filtering
        mock_device.create_sampler.assert_called_once()
        call_kwargs = mock_device.create_sampler.call_args.kwargs

        assert call_kwargs["mag_filter"] == wgpu.FilterMode.nearest, (
            f"mag_filter should be nearest, got {call_kwargs['mag_filter']}"
        )
        assert call_kwargs["min_filter"] == wgpu.FilterMode.nearest, (
            f"min_filter should be nearest, got {call_kwargs['min_filter']}"
        )

    def test_sky_exposure_sampler_uses_clamp_to_edge(self) -> None:
        """Test that sky exposure sampler clamps to edge.

        Clamping prevents texture coordinates outside [0,1] from wrapping
        around to the other side of the texture.
        """
        import wgpu

        mock_device = MagicMock()
        mock_device.create_sampler.return_value = MagicMock()

        from catley.backends.wgpu.gpu_lighting import GPULightingSystem

        system = object.__new__(GPULightingSystem)
        system.device = mock_device

        system._create_samplers()

        call_kwargs = mock_device.create_sampler.call_args.kwargs

        assert call_kwargs["address_mode_u"] == wgpu.AddressMode.clamp_to_edge
        assert call_kwargs["address_mode_v"] == wgpu.AddressMode.clamp_to_edge
        assert call_kwargs["address_mode_w"] == wgpu.AddressMode.clamp_to_edge
