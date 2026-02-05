"""Test WGSL shader compilation for the lighting system."""

from pathlib import Path

import pytest

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False


class TestWGSLLightingShaders:
    """Test WGSL lighting shader compilation and validation."""

    @pytest.mark.skipif(not WGPU_AVAILABLE, reason="wgpu-py not available")
    def test_point_light_shader_compilation(self):
        """Test that the point light WGSL shader compiles successfully."""

        # Path to the WGSL shader
        shader_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "assets/shaders/wgsl/lighting/point_light.wgsl"
        )

        assert shader_path.exists(), f"Shader not found: {shader_path}"

        # Read shader source
        with shader_path.open() as f:
            shader_source = f.read()

        # Verify basic shader structure
        assert "@vertex" in shader_source, "Missing vertex shader entry point"
        assert "@fragment" in shader_source, "Missing fragment shader entry point"
        assert "vs_main" in shader_source, "Missing vertex shader main function"
        assert "fs_main" in shader_source, "Missing fragment shader main function"

        # Create a device for testing compilation
        adapter = wgpu.gpu.request_adapter()  # type: ignore[possibly-unbound]
        device = adapter.request_device()

        # Try to create shader module - this will raise if compilation fails
        shader_module = device.create_shader_module(code=shader_source)

        # If we get here, compilation succeeded
        assert shader_module is not None

    @pytest.mark.skipif(not WGPU_AVAILABLE, reason="wgpu-py not available")
    def test_critical_shadow_algorithms_preserved(self):
        """Verify that critical shadow algorithms are preserved in WGSL translation."""

        shader_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "assets/shaders/wgsl/lighting/point_light.wgsl"
        )

        with shader_path.open() as f:
            shader_source = f.read()

        # Check for sign-based stepping in directional shadow
        # Both terrain and actor shadows use sign-based direction vectors
        assert "step_dir.x = select" in shader_source, (
            "Sign-based shadow direction calculation not preserved"
        )

        # Check for discrete tile-based stepping preservation
        assert "discrete" in shader_source.lower(), (
            "Comment about discrete tile-based stepping missing"
        )

        # Check for basic lighting calculation preservation
        assert "distance = length(world_pos - light_pos)" in shader_source, (
            "Basic lighting distance calculation not preserved"
        )

        # Verify texture-based terrain shadow functions exist
        assert "computePointLightShadow" in shader_source, (
            "Texture-based point light shadow calculation missing"
        )
        assert "computeDirectionalShadow" in shader_source, (
            "Texture-based directional shadow calculation missing"
        )

        # Verify actor shadow functions exist
        assert "computeActorShadow" in shader_source, "Actor shadow calculation missing"
        assert "computeActorDirectionalShadow" in shader_source, (
            "Actor directional shadow calculation missing"
        )

        # Verify shadow grid texture is used
        assert "shadow_grid" in shader_source, "Shadow grid texture binding missing"

    def test_shader_has_key_algorithms(self):
        """Test that WGSL shader has the key lighting algorithms."""

        # Read WGSL shader
        wgsl_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "assets/shaders/wgsl/lighting/point_light.wgsl"
        )
        with wgsl_path.open() as f:
            wgsl_source = f.read()

        # Check that key algorithms are present
        # Noise function
        assert "noise2d" in wgsl_source, "noise2d function missing in WGSL"

        # Shadow calculations
        assert "computePointLightShadow" in wgsl_source, (
            "WGSL should have texture-based point light shadows"
        )

        # Directional shadows
        assert "computeDirectionalShadow" in wgsl_source, (
            "WGSL should have texture-based directional shadows"
        )

        # Actor shadows
        assert "computeActorShadow" in wgsl_source, (
            "WGSL should have actor shadow function"
        )

        # Flicker effects
        assert "flicker" in wgsl_source.lower(), "Flicker effects missing in WGSL"
