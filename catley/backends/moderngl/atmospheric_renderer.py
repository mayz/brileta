"""ModernGL renderer for atmospheric effects (cloud shadows, ground mist)."""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image as PILImage

from catley import config
from catley.backends.moderngl import get_uniform
from catley.backends.moderngl.screen_renderer import VERTEX_DTYPE
from catley.backends.moderngl.shader_manager import ShaderManager


class ModernGLAtmosphericRenderer:
    """ModernGL renderer for atmospheric effects (cloud shadows, ground mist)."""

    def __init__(self, mgl_context: moderngl.Context) -> None:
        self._get_uniform = get_uniform
        self.mgl_context = mgl_context
        self.shader_manager = ShaderManager(mgl_context)
        self.program = self.shader_manager.create_program(
            "glsl/effects/cloud_layer.vert",
            "glsl/effects/cloud_layer.frag",
            "atmospheric",
        )

        self._vertex_dtype = VERTEX_DTYPE
        self._cpu_vertices = np.zeros(6, dtype=self._vertex_dtype)
        self._vbo = self.mgl_context.buffer(
            reserve=6 * self._vertex_dtype.itemsize, dynamic=True
        )
        self._vao = self.mgl_context.vertex_array(
            self.program,
            [(self._vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )
        self._noise_texture = self._load_noise_texture()
        self._default_explored_texture: moderngl.Texture | None = None
        self._default_visible_texture: moderngl.Texture | None = None

    def render(
        self,
        viewport_offset: tuple[int, int],
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: moderngl.Texture | None,
        explored_texture: moderngl.Texture | None,
        visible_texture: moderngl.Texture | None,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        strength: float,
        tint_color: tuple[int, int, int],
        drift_offset: tuple[float, float],
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        blend_mode: str,
        pixel_bounds: tuple[int, int, int, int],
        letterbox_geometry: tuple[int, int, int, int],
    ) -> None:
        """Render the atmospheric overlay."""
        if sky_exposure_texture is None or self._noise_texture is None:
            return
        if explored_texture is None:
            explored_texture = self._get_default_explored_texture()
        if explored_texture is None:
            return
        if visible_texture is None:
            visible_texture = self._get_default_visible_texture()
        if visible_texture is None:
            return
        left, top, right, bottom = pixel_bounds
        width_px = right - left
        height_px = bottom - top
        if width_px <= 0 or height_px <= 0:
            return

        vertices = self._cpu_vertices
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        vertices[0] = ((left, top), (u1, v1), color_rgba)
        vertices[1] = ((right, top), (u2, v1), color_rgba)
        vertices[2] = ((left, bottom), (u1, v2), color_rgba)
        vertices[3] = ((right, top), (u2, v1), color_rgba)
        vertices[4] = ((left, bottom), (u1, v2), color_rgba)
        vertices[5] = ((right, bottom), (u2, v2), color_rgba)

        self._vbo.write(vertices.tobytes())

        self._get_uniform(self.program, "u_letterbox").value = letterbox_geometry
        self._get_uniform(self.program, "u_viewport_offset").value = viewport_offset
        self._get_uniform(self.program, "u_viewport_size").value = viewport_size
        self._get_uniform(self.program, "u_map_size").value = map_size
        self._get_uniform(
            self.program, "u_sky_exposure_threshold"
        ).value = sky_exposure_threshold
        self._get_uniform(self.program, "u_noise_scale").value = noise_scale
        self._get_uniform(
            self.program, "u_noise_threshold_low"
        ).value = noise_threshold_low
        self._get_uniform(
            self.program, "u_noise_threshold_high"
        ).value = noise_threshold_high
        self._get_uniform(self.program, "u_strength").value = strength
        self._get_uniform(self.program, "u_tint_color").value = tint_color
        self._get_uniform(self.program, "u_drift_offset").value = drift_offset
        self._get_uniform(self.program, "u_turbulence_offset").value = turbulence_offset
        self._get_uniform(
            self.program, "u_turbulence_strength"
        ).value = turbulence_strength
        self._get_uniform(self.program, "u_turbulence_scale").value = turbulence_scale
        self._get_uniform(self.program, "u_blend_mode").value = (
            0 if blend_mode == "darken" else 1
        )
        sky_exposure_texture.use(location=0)
        self._get_uniform(self.program, "u_sky_exposure_map").value = 0
        self._noise_texture.use(location=1)
        self._get_uniform(self.program, "u_noise_texture").value = 1
        explored_texture.use(location=2)
        self._get_uniform(self.program, "u_explored_map").value = 2
        visible_texture.use(location=3)
        self._get_uniform(self.program, "u_visible_map").value = 3
        self.mgl_context.enable(moderngl.BLEND)
        if blend_mode == "lighten":
            self.mgl_context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )
        else:
            self.mgl_context.blend_func = moderngl.DST_COLOR, moderngl.ZERO
        try:
            self._vao.render(moderngl.TRIANGLES, vertices=6)
        finally:
            self.mgl_context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

    def _load_noise_texture(self) -> moderngl.Texture | None:
        """Load the static cloud noise texture for atmospheric overlays."""
        if config.IS_TEST_ENVIRONMENT:
            return None

        texture_path = (
            Path(config.PROJECT_ROOT_PATH) / "assets" / "textures" / "cloud_noise.png"
        )
        if not texture_path.exists():
            return None

        img = PILImage.open(str(texture_path)).convert("L")
        pixels = np.array(img, dtype="u1")
        texture = self.mgl_context.texture(img.size, 1, pixels.tobytes())
        texture.repeat_x = True
        texture.repeat_y = True
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _get_default_explored_texture(self) -> moderngl.Texture | None:
        """Return a 1x1 white texture to treat all tiles as explored."""
        if self._default_explored_texture is not None:
            return self._default_explored_texture
        if config.IS_TEST_ENVIRONMENT:
            return None
        pixels = np.array([[1.0]], dtype="f4")
        texture = self.mgl_context.texture((1, 1), components=1, dtype="f4")
        texture.write(pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._default_explored_texture = texture
        return texture

    def _get_default_visible_texture(self) -> moderngl.Texture | None:
        """Return a 1x1 white texture to treat all tiles as visible."""
        if self._default_visible_texture is not None:
            return self._default_visible_texture
        if config.IS_TEST_ENVIRONMENT:
            return None
        pixels = np.array([[1.0]], dtype="f4")
        texture = self.mgl_context.texture((1, 1), components=1, dtype="f4")
        texture.write(pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._default_visible_texture = texture
        return texture
