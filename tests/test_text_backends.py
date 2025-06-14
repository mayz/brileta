from unittest.mock import MagicMock

from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import PillowTextBackend, TCODTextBackend


def _make_renderer(tile_height: int = 16) -> Renderer:
    renderer = MagicMock(spec=Renderer)
    renderer.tile_dimensions = (8, tile_height)
    renderer.sdl_renderer = MagicMock()
    renderer.root_console = MagicMock()
    return renderer


def test_backend_interchangeability() -> None:
    renderer = _make_renderer()
    tcod_backend = TCODTextBackend(renderer)
    pillow_backend = PillowTextBackend(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)

    assert (
        tcod_backend.get_effective_line_height()
        == pillow_backend.get_effective_line_height()
    )

    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()
    assert asc_t + desc_t == asc_p + desc_p

    metrics_t = tcod_backend.get_text_metrics("Hi")
    metrics_p = pillow_backend.get_text_metrics("Hi")
    assert metrics_t[2] == metrics_p[2]


def test_no_unnecessary_scaling() -> None:
    renderer = _make_renderer()
    backend = PillowTextBackend(renderer)
    update_mock = MagicMock()
    backend._update_scaling_internal = update_mock  # type: ignore[assignment]
    backend.configure_scaling(16)
    backend.configure_scaling(16)
    backend.configure_scaling(20)
    assert update_mock.call_count == 1


def test_pillow_font_sizing() -> None:
    renderer = _make_renderer()
    backend = PillowTextBackend(renderer)
    for height in [12, 16, 20, 24, 32]:
        backend.configure_scaling(height)
        ascent, descent = backend.get_font_metrics()
        assert abs((ascent + descent) - height) <= 1
        assert backend.get_effective_line_height() == height


def test_layout_consistency() -> None:
    renderer = _make_renderer()
    tcod_backend = TCODTextBackend(renderer)
    pillow_backend = PillowTextBackend(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)
    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()
    baseline_t = 16 - desc_t
    baseline_p = 16 - desc_p
    assert abs(baseline_t - baseline_p) <= 1
