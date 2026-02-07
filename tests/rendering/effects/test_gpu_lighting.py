from unittest.mock import Mock

from catley.backends.wgpu.gpu_lighting import GPULightingSystem
from catley.util.coordinates import Rect


def test_compute_lightmap_texture_path_skips_full_readback_copy() -> None:
    """GPU texture path should not issue a full texture->buffer copy.

    Full readback is only needed for the CPU lighting path, which performs
    that copy in _readback_lightmap_from_output_texture().
    """
    system = GPULightingSystem.__new__(GPULightingSystem)

    # Required state for _compute_lightmap_gpu_to_texture.
    system.revision = 0
    system._render_pipeline = Mock()
    system._vertex_buffer = Mock()
    system._output_texture = Mock()
    system._bind_group = Mock()
    system._last_light_data_hash = None
    system._cached_light_revision = -1
    system._cached_light_data = None
    system._first_frame = True

    # Bypass resource/light-data side effects.
    system._ensure_resources_for_viewport = Mock(return_value=True)
    system._collect_light_data = Mock(return_value=[])
    system._update_sky_exposure_texture = Mock()
    system._update_explored_texture = Mock()
    system._update_visible_texture = Mock()
    system._update_shadow_grid_texture = Mock()
    system._update_emission_texture = Mock()
    system._update_uniform_buffer = Mock()
    system._create_bind_group = Mock()

    # Mock command encoding.
    render_pass = Mock()
    command_encoder = Mock()
    command_encoder.begin_render_pass.return_value = render_pass
    command_encoder.finish.return_value = "command_buffer"

    system.device = Mock()
    system.device.create_command_encoder.return_value = command_encoder
    system.queue = Mock()

    viewport = Rect(0, 0, 10, 8)
    assert system._compute_lightmap_gpu_to_texture(viewport) is True

    # The texture path should only render + submit; no full readback copy.
    command_encoder.copy_texture_to_buffer.assert_not_called()
    system.queue.submit.assert_called_once_with(["command_buffer"])
