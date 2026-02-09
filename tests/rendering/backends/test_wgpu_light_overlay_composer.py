"""Unit tests for WGPULightOverlayComposer."""

from unittest.mock import Mock

import numpy as np

from brileta.backends.wgpu.light_overlay_composer import WGPULightOverlayComposer
from brileta.util.coordinates import Rect


def _mock_texture(width: int, height: int) -> Mock:
    texture = Mock()
    texture.width = width
    texture.height = height
    texture.create_view.return_value = Mock()
    return texture


def _build_composer() -> tuple[WGPULightOverlayComposer, Mock, Mock]:
    queue = Mock()
    device = Mock()
    device.create_buffer.side_effect = [Mock(), Mock()]
    device.create_bind_group_layout.return_value = Mock()
    device.create_texture.return_value = _mock_texture(4, 4)

    resource_manager = Mock()
    resource_manager.device = device
    resource_manager.queue = queue
    resource_manager.get_or_create_render_texture.return_value = _mock_texture(4, 4)

    shader_manager = Mock()
    shader_manager.create_render_pipeline.return_value = Mock()

    composer = WGPULightOverlayComposer(resource_manager, shader_manager)
    return composer, resource_manager, device


def test_uniform_buffer_allocates_two_vec4s() -> None:
    """Compose uniforms are 8 floats (2 vec4<f32>) => 32 bytes."""
    _, resource_manager, device = _build_composer()

    first_call = device.create_buffer.call_args_list[0]
    assert first_call.kwargs["size"] == 32
    assert first_call.kwargs["label"] == "light_overlay_compose_uniforms"
    assert resource_manager.queue.write_buffer.called


def test_compose_writes_32_byte_uniforms_and_draws() -> None:
    """Compose should push 8 floats of uniforms and submit one fullscreen draw."""
    composer, resource_manager, device = _build_composer()

    command_encoder = Mock()
    render_pass = Mock()
    command_encoder.begin_render_pass.return_value = render_pass
    command_encoder.finish.return_value = Mock()
    device.create_command_encoder.return_value = command_encoder
    device.create_bind_group.return_value = Mock()

    dark_texture = _mock_texture(4, 4)
    light_texture = _mock_texture(4, 4)
    lightmap_texture = _mock_texture(4, 4)

    resource_manager.queue.write_buffer.reset_mock()
    result = composer.compose(
        dark_texture=dark_texture,
        light_texture=light_texture,
        lightmap_texture=lightmap_texture,
        visible_mask_buffer=np.zeros((4, 4), dtype=np.bool_),
        viewport_bounds=Rect(0, 0, 4, 4),
        viewport_offset=(0, 0),
        pad_tiles=1,
        tile_dimensions=(20, 20),
    )

    assert result is resource_manager.get_or_create_render_texture.return_value
    uniform_write = resource_manager.queue.write_buffer.call_args_list[-1]
    assert uniform_write.args[2].nbytes == 32
    render_pass.draw.assert_called_once_with(6, 1, 0, 0)
    resource_manager.queue.submit.assert_called_once()
