from unittest.mock import MagicMock

import pytest

from catley.game.enums import BlendMode
from catley.util.coordinates import Rect
from catley.view.render.effects.environmental import (
    EnvironmentalEffect,
    EnvironmentalEffectSystem,
)


def test_environmental_effect_initialization_sets_max_lifetime() -> None:
    eff = EnvironmentalEffect(
        position=(1.0, 2.0),
        radius=2.5,
        effect_type="smoke",
        intensity=0.7,
        lifetime=4.0,
        tint_color=(10, 20, 30),
        blend_mode=BlendMode.TINT,
    )
    assert eff.max_lifetime == 4.0
    assert eff.position == (1.0, 2.0)
    assert eff.radius == 2.5


def test_add_effect_and_update_removal() -> None:
    system = EnvironmentalEffectSystem()
    e1 = EnvironmentalEffect((0.0, 0.0), 1.0, "a", 1.0, 1.0, (0, 0, 0), BlendMode.TINT)
    e2 = EnvironmentalEffect((1.0, 1.0), 1.0, "b", 1.0, 0.1, (0, 0, 0), BlendMode.TINT)
    system.add_effect(e1)
    system.add_effect(e2)
    assert len(system.effects) == 2
    system.update(0.05)
    assert pytest.approx(0.95) == e1.lifetime
    assert pytest.approx(0.05) == e2.lifetime
    system.update(0.1)
    assert len(system.effects) == 1
    assert system.effects[0] is e1


def test_render_effects_culling_and_intensity() -> None:
    system = EnvironmentalEffectSystem()
    e = EnvironmentalEffect(
        (1.0, 1.0), 1.0, "flash", 1.0, 2.0, (100, 100, 100), BlendMode.REPLACE
    )
    system.add_effect(e)
    system.update(1.0)
    renderer = MagicMock()
    viewport = Rect.from_bounds(0, 0, 2, 2)
    system.render_effects(renderer, viewport, (0, 0))
    renderer.apply_environmental_effect.assert_called_once()
    args = renderer.apply_environmental_effect.call_args[0]
    assert args[0] == (1.0, 1.0)
    assert args[1] == 1.0
    assert args[2] == (100, 100, 100)
    assert args[3] == pytest.approx(0.5)
    assert args[4] is BlendMode.REPLACE

    renderer.reset_mock()
    off_view = Rect.from_bounds(5, 5, 6, 6)
    system.render_effects(renderer, off_view, (0, 0))
    renderer.apply_environmental_effect.assert_not_called()
