import pytest

from brileta.view.render.effects.atmospheric import (
    AtmosphericConfig,
    AtmosphericLayerConfig,
    AtmosphericLayerSystem,
)
from tests.helpers import dt


def test_default_config_values() -> None:
    config = AtmosphericConfig.create_default()

    assert config.cloud_coverage == 0.8
    assert len(config.layers) == 3

    cloud = config.layers[0]
    mist = config.layers[1]
    wet = config.layers[2]

    assert cloud.name == "cloud_shadows"
    assert cloud.blend_mode == "darken"
    assert cloud.strength == 0.8
    assert cloud.tint_color == (70, 75, 80)
    assert cloud.noise_scale == 0.03
    assert cloud.noise_threshold_low == 0.46
    assert cloud.noise_threshold_high == 0.62
    assert cloud.drift_direction == (1.0, 0.2)
    assert cloud.drift_speed == 0.01
    assert cloud.turbulence_strength == 0.0
    assert cloud.affects_foreground is True
    assert cloud.enable_when_raining is False
    assert cloud.disable_when_overcast is True
    assert cloud.sky_exposure_threshold == 0.85

    assert mist.name == "ground_mist"
    assert mist.blend_mode == "lighten"
    assert mist.strength == 0.085
    assert mist.tint_color == (238, 242, 246)
    assert mist.noise_scale == 0.13
    assert mist.noise_threshold_low == 0.15
    assert mist.noise_threshold_high == 0.85
    assert mist.drift_direction == (0.2, 1.0)
    assert mist.drift_speed == 0.04
    assert mist.turbulence_strength == 0.7
    assert mist.turbulence_scale == 0.16
    assert mist.turbulence_speed == 0.03
    assert mist.affects_foreground is True
    assert mist.enable_when_raining is False
    assert mist.disable_when_overcast is False
    assert mist.sky_exposure_threshold == 0.85

    assert wet.name == "rain_ground_wet"
    assert wet.blend_mode == "darken"
    assert wet.strength == 0.08
    assert wet.tint_color == (122, 130, 142)
    assert wet.noise_scale == 0.02
    assert wet.noise_threshold_low == 0.04
    assert wet.noise_threshold_high == 0.96
    assert wet.drift_direction == (0.8, 0.3)
    assert wet.drift_speed == 0.01
    assert wet.turbulence_strength == 0.2
    assert wet.turbulence_scale == 0.09
    assert wet.turbulence_speed == 0.012
    assert wet.affects_foreground is False
    assert wet.enable_when_raining is True
    assert wet.disable_when_overcast is False
    assert wet.sky_exposure_threshold == 0.85


def test_update_advances_drift_offsets() -> None:
    layer = AtmosphericLayerConfig(
        name="drift",
        drift_direction=(3.0, 4.0),
        drift_speed=1.0,
    )
    system = AtmosphericLayerSystem(AtmosphericConfig(layers=[layer]))

    system.update(dt(1.0))
    (_, state) = system.get_active_layers()[0]

    assert state.drift_offset_x == pytest.approx(0.6)
    assert state.drift_offset_y == pytest.approx(0.8)


def test_offsets_wrap_at_100() -> None:
    layer = AtmosphericLayerConfig(
        name="wrap",
        drift_direction=(1.0, 0.0),
        drift_speed=200.0,
        turbulence_strength=1.0,
        turbulence_speed=150.0,
    )
    system = AtmosphericLayerSystem(AtmosphericConfig(layers=[layer]))

    system.update(dt(1.0))
    (_, state) = system.get_active_layers()[0]

    assert state.drift_offset_x == pytest.approx(0.0)
    assert state.drift_offset_y == pytest.approx(0.0)
    assert state.turbulence_offset == pytest.approx(50.0)


def test_get_active_layers_respects_overcast_disable() -> None:
    cloud = AtmosphericLayerConfig(
        name="clouds",
        disable_when_overcast=True,
    )
    mist = AtmosphericLayerConfig(
        name="mist",
        disable_when_overcast=False,
    )
    system = AtmosphericLayerSystem(
        AtmosphericConfig(cloud_coverage=1.0, layers=[cloud, mist])
    )

    active = system.get_active_layers()
    assert [layer.name for layer, _ in active] == ["mist"]


def test_set_cloud_coverage_clamps_values() -> None:
    system = AtmosphericLayerSystem(AtmosphericConfig())

    system.set_cloud_coverage(1.5)
    assert system.config.cloud_coverage == 1.0

    system.set_cloud_coverage(-0.5)
    assert system.config.cloud_coverage == 0.0

    system.set_cloud_coverage(0.7)
    assert system.config.cloud_coverage == 0.7


def test_get_active_layers_skips_disabled_layers() -> None:
    enabled_layer = AtmosphericLayerConfig(name="enabled", enabled=True)
    disabled_layer = AtmosphericLayerConfig(name="disabled", enabled=False)
    system = AtmosphericLayerSystem(
        AtmosphericConfig(layers=[enabled_layer, disabled_layer])
    )

    active = system.get_active_layers()
    assert [layer.name for layer, _ in active] == ["enabled"]


def test_get_active_layers_respects_rain_requirement() -> None:
    wet_layer = AtmosphericLayerConfig(name="wet", enable_when_raining=True)
    always_layer = AtmosphericLayerConfig(name="always")
    system = AtmosphericLayerSystem(AtmosphericConfig(layers=[wet_layer, always_layer]))

    dry_active = system.get_active_layers(is_raining=False)
    assert [layer.name for layer, _ in dry_active] == ["always"]

    rainy_active = system.get_active_layers(is_raining=True)
    assert [layer.name for layer, _ in rainy_active] == ["wet", "always"]
