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

        # Check for critical shadow direction calculation preservation
        # The WGPU migration plan specifically calls out sign-based direction vectors
        expected_select = (
            "select(select(0.0, 1.0, uniforms.sun_direction.x < 0.0), "
            "-1.0, uniforms.sun_direction.x > 0.0)"
        )
        assert expected_select in shader_source, (
            "Sign-based shadow direction calculation not preserved"
        )

        # Check for discrete tile-based stepping preservation
        assert "discrete tile-based stepping" in shader_source, (
            "Comment about discrete tile-based stepping missing"
        )

        # Check for basic lighting calculation preservation
        assert "distance = length(world_pos - light_pos)" in shader_source, (
            "Basic lighting distance calculation not preserved"
        )

        # Check for critical algorithm preservation comments
        assert "CRITICAL:" in shader_source, (
            "Critical algorithm preservation markers missing"
        )

        # Verify shadow attenuation functions exist
        assert "calculateShadowAttenuation" in shader_source, (
            "Shadow attenuation calculation missing"
        )
        assert "calculateDirectionalShadowAttenuation" in shader_source, (
            "Directional shadow attenuation calculation missing"
        )

    def test_shader_structure_matches_glsl_original(self):
        """Test that WGSL shader preserves the structure of the original GLSL."""

        # Read WGSL shader
        wgsl_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "assets/shaders/wgsl/lighting/point_light.wgsl"
        )
        with wgsl_path.open() as f:
            wgsl_source = f.read()

        # Read original GLSL fragment shader
        glsl_frag_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "assets/shaders/glsl/lighting/point_light.frag"
        )
        with glsl_frag_path.open() as f:
            glsl_frag_source = f.read()

        # Check that key algorithms are preserved
        # Noise function
        assert "noise2d" in wgsl_source, "noise2d function missing in WGSL"
        assert "noise2d" in glsl_frag_source, "Original GLSL should have noise2d"

        # Shadow calculations
        assert "calculateShadowAttenuation" in wgsl_source, (
            "Shadow attenuation missing in WGSL"
        )
        assert "calculateShadowAttenuation" in glsl_frag_source, (
            "Original GLSL should have shadow attenuation"
        )

        # Directional shadows
        assert "calculateDirectionalShadowAttenuation" in wgsl_source, (
            "Directional shadows missing in WGSL"
        )
        assert "calculateDirectionalShadowAttenuation" in glsl_frag_source, (
            "Original GLSL should have directional shadows"
        )

        # Flicker effects
        assert "flicker" in wgsl_source.lower(), "Flicker effects missing in WGSL"
        assert "flicker" in glsl_frag_source.lower(), (
            "Original GLSL should have flicker effects"
        )
