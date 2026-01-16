import math

import numpy as np

from catley.environment.map import MapRegion
from catley.game.game_world import GameWorld
from catley.game.lights import DirectionalLight, DynamicLight, StaticLight, Vec2
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.cpu import CPULightingSystem


def test_cpu_lighting_system_basic() -> None:
    """Test that CPULightingSystem can be created and compute a basic lightmap."""
    import random

    # Seed RNG for deterministic map generation - avoids intermittent failures
    # when the light happens to land in an outdoor room (20% chance), which
    # reduces light intensity via sky_exposure calculations
    random.seed(42)

    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add a static light
    static_light = StaticLight(position=(25, 25), radius=5, color=(255, 255, 255))
    gw.add_light(static_light)

    # Compute lightmap for a small area around the light
    viewport_bounds = Rect(20, 20, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    # Should return a numpy array
    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (10, 10, 3)

    # Light should be brighter in the center
    center_intensity = lightmap[5, 5]
    corner_intensity = lightmap[0, 0]
    assert np.any(center_intensity > corner_intensity)


def test_cpu_lighting_system_dynamic_light() -> None:
    """Test that dynamic lights can be added and affect the lightmap."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add a dynamic light
    dynamic_light = DynamicLight(
        position=(25, 25), radius=3, color=(255, 128, 64), flicker_enabled=True
    )
    gw.add_light(dynamic_light)

    # Compute lightmap
    viewport_bounds = Rect(22, 22, 6, 6)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    # Should return a numpy array with the right shape
    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (6, 6, 3)


def test_cpu_lighting_system_update() -> None:
    """Test that lighting system can be updated without errors."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Should not raise any exceptions
    lighting_system.update(FixedTimestep(0.016))  # 60 FPS delta time


# Vec2 Tests
def test_vec2_initialization() -> None:
    """Test Vec2 basic initialization and properties."""
    vec = Vec2(3.0, 4.0)
    assert vec.x == 3.0
    assert vec.y == 4.0


def test_vec2_normalization() -> None:
    """Test Vec2 normalization functionality."""
    # Test standard vector normalization
    vec = Vec2(3.0, 4.0)
    normalized = vec.normalized()

    # Should have magnitude of 1 (within floating point precision)
    magnitude = math.sqrt(normalized.x**2 + normalized.y**2)
    assert abs(magnitude - 1.0) < 1e-6

    # Should maintain direction proportions
    assert abs(normalized.x / normalized.y - 3.0 / 4.0) < 1e-6


def test_vec2_normalization_zero_vector() -> None:
    """Test Vec2 normalization of zero vector edge case."""
    vec = Vec2(0.0, 0.0)
    normalized = vec.normalized()

    # Zero vector should remain zero after normalization
    assert normalized.x == 0.0
    assert normalized.y == 0.0


def test_vec2_normalization_unit_vectors() -> None:
    """Test Vec2 normalization of already-normalized vectors."""
    # Test cardinal directions
    vec_north = Vec2(0.0, -1.0).normalized()
    assert abs(vec_north.x) < 1e-6
    assert abs(vec_north.y - (-1.0)) < 1e-6

    vec_east = Vec2(1.0, 0.0).normalized()
    assert abs(vec_east.x - 1.0) < 1e-6
    assert abs(vec_east.y) < 1e-6


# DirectionalLight Tests
def test_directional_light_creation() -> None:
    """Test DirectionalLight basic creation and properties."""
    direction = Vec2(0.6, -0.8)
    color = (255, 243, 204)
    intensity = 0.8

    light = DirectionalLight(direction=direction, color=color, intensity=intensity)

    assert light.direction.x == 0.6
    assert light.direction.y == -0.8
    assert light.color == color
    assert light.intensity == intensity
    assert light.is_static() is True


def test_directional_light_direction_normalization() -> None:
    """Test that DirectionalLight normalizes direction vectors."""
    # Create with non-normalized direction
    direction = Vec2(3.0, 4.0)  # Magnitude = 5
    light = DirectionalLight(direction=direction, color=(255, 255, 255))

    # Direction should be normalized
    magnitude = math.sqrt(light.direction.x**2 + light.direction.y**2)
    assert abs(magnitude - 1.0) < 1e-6


def test_directional_light_create_sun_default() -> None:
    """Test DirectionalLight.create_sun() with default parameters."""
    sun = DirectionalLight.create_sun()

    # Check default values
    assert sun.intensity == 0.8
    assert sun.color == (255, 243, 204)  # Warm sunlight
    assert sun.is_static() is True

    # Direction should be normalized
    magnitude = math.sqrt(sun.direction.x**2 + sun.direction.y**2)
    assert abs(magnitude - 1.0) < 1e-6


def test_directional_light_create_sun_custom_angles() -> None:
    """Test DirectionalLight.create_sun() with custom elevation and azimuth."""
    # Test noon sun (high elevation, south-facing)
    noon_sun = DirectionalLight.create_sun(
        elevation_degrees=90.0,  # Directly overhead
        azimuth_degrees=180.0,  # South
        intensity=1.0,
        color=(255, 255, 255),
    )

    # At 90 degree elevation, y should be -1 (downward)
    assert abs(noon_sun.direction.y - (-1.0)) < 1e-6
    # x should be close to 0 (no horizontal component)
    assert abs(noon_sun.direction.x) < 1e-6

    # Test sunrise (low elevation, east-facing)
    sunrise_sun = DirectionalLight.create_sun(
        elevation_degrees=0.0,  # On horizon
        azimuth_degrees=90.0,  # East
    )

    # At 0 degree elevation, y should be close to 0
    assert abs(sunrise_sun.direction.y) < 1e-6
    # x should be positive (eastward)
    assert sunrise_sun.direction.x > 0.5


def test_directional_light_southeast_default() -> None:
    """Test that default sun direction matches expected southeast orientation."""
    sun = DirectionalLight.create_sun()

    # Default azimuth is 135 degrees (southeast)
    # Direction should have positive x (east component) and negative y (down component)
    assert sun.direction.x > 0
    assert sun.direction.y < 0


# Sky Exposure and Region Tests
def test_map_region_outdoor_factory() -> None:
    """Test MapRegion.create_outdoor_region() factory method."""
    outdoor_region = MapRegion.create_outdoor_region(
        map_region_id=1, region_type="clearing"
    )

    assert outdoor_region.id == 1
    assert outdoor_region.region_type == "clearing"
    assert outdoor_region.sky_exposure == 1.0  # Full sky exposure


def test_map_region_indoor_factory() -> None:
    """Test MapRegion.create_indoor_region() factory method."""
    indoor_region = MapRegion.create_indoor_region(
        map_region_id=2, region_type="dungeon"
    )

    assert indoor_region.id == 2
    assert indoor_region.region_type == "dungeon"
    assert indoor_region.sky_exposure == 0.0  # No sky exposure


def test_map_region_custom_sky_exposure() -> None:
    """Test MapRegion creation with custom sky exposure values."""
    # Test partial sky exposure (e.g., covered outdoor area)
    covered_area = MapRegion.create_outdoor_region(map_region_id=3, sky_exposure=0.5)
    assert covered_area.sky_exposure == 0.5

    # Test underground with minimal light shaft
    underground = MapRegion.create_indoor_region(map_region_id=4, sky_exposure=0.1)
    assert underground.sky_exposure == 0.1


def test_map_region_default_values() -> None:
    """Test MapRegion default values for bounds and connections."""
    region = MapRegion.create_outdoor_region(map_region_id=5)

    assert region.bounds == []
    assert region.connections == {}
    assert region.sky_exposure == 1.0


# Directional Lighting Integration Tests
def test_cpu_lighting_system_with_directional_light() -> None:
    """Test that CPULightingSystem can handle DirectionalLight."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add a directional light (sun)
    sun = DirectionalLight.create_sun(intensity=0.5)
    gw.add_light(sun)

    # Should be able to compute lightmap without errors
    viewport_bounds = Rect(20, 20, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (10, 10, 3)


def test_global_lights_filtering() -> None:
    """Test GameWorld.get_global_lights() correctly filters DirectionalLight."""
    gw = GameWorld(30, 30)

    # Add different types of lights
    static_light = StaticLight(position=(10, 10), radius=3, color=(255, 255, 255))
    dynamic_light = DynamicLight(position=(15, 15), radius=2, color=(255, 128, 0))
    sun = DirectionalLight.create_sun()

    gw.add_light(static_light)
    gw.add_light(dynamic_light)
    gw.add_light(sun)

    # Should only return global lights
    global_lights = gw.get_global_lights()
    assert len(global_lights) == 1
    assert global_lights[0] is sun
    assert isinstance(global_lights[0], DirectionalLight)


def test_static_lights_filtering() -> None:
    """Test GameWorld.get_static_lights() excludes global lights."""
    gw = GameWorld(30, 30)

    # Add different types of lights
    static_light = StaticLight(position=(10, 10), radius=3, color=(255, 255, 255))
    dynamic_light = DynamicLight(position=(15, 15), radius=2, color=(255, 128, 0))
    sun = DirectionalLight.create_sun()

    gw.add_light(static_light)
    gw.add_light(dynamic_light)
    gw.add_light(sun)

    # Should only return static point lights (not global lights)
    static_lights = gw.get_static_lights()
    assert len(static_lights) == 1
    assert static_lights[0] is static_light


def test_time_of_day_sun_position() -> None:
    """Test GameWorld.set_time_of_day() updates sun elevation."""
    gw = GameWorld(30, 30)
    lighting_system = CPULightingSystem(gw)
    gw.lighting_system = lighting_system

    # Add sun
    sun = DirectionalLight.create_sun(elevation_degrees=45.0)
    gw.add_light(sun)

    # Just test that set_time_of_day doesn't crash and updates direction
    gw.set_time_of_day(12.0)  # Noon
    gw.set_time_of_day(6.0)  # Sunrise

    # Just verify the method works and direction is normalized
    magnitude = math.sqrt(sun.direction.x**2 + sun.direction.y**2)
    assert abs(magnitude - 1.0) < 1e-6


def test_directional_light_with_sky_exposure() -> None:
    """Test that directional lighting respects sky exposure in regions."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Create outdoor and indoor regions
    outdoor_region = MapRegion.create_outdoor_region(map_region_id=1, sky_exposure=1.0)
    indoor_region = MapRegion.create_indoor_region(map_region_id=2, sky_exposure=0.0)

    # Add regions to map (basic setup)
    gw.game_map.regions = {1: outdoor_region, 2: indoor_region}

    # Add sun
    sun = DirectionalLight.create_sun(intensity=0.8)
    gw.add_light(sun)

    # Should be able to compute lightmap
    viewport_bounds = Rect(20, 20, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (10, 10, 3)


# Indoor/Outdoor Behavior Tests
def test_torch_effectiveness_with_sunlight_config() -> None:
    """Test that torch effectiveness is configured correctly with sun enabled."""
    from catley.view.render.lighting.cpu import LightingConfig

    # Test basic configuration creation and properties
    config_with_sun = LightingConfig(sun_enabled=True, sun_intensity=0.8)
    assert config_with_sun.sun_enabled is True
    assert config_with_sun.sun_intensity == 0.8

    config_no_sun = LightingConfig(sun_enabled=False)
    assert config_no_sun.sun_enabled is False

    # Test one lighting system instance to avoid cache conflicts
    gw = GameWorld(30, 30)
    lighting_system = CPULightingSystem(gw, config=config_with_sun)
    assert lighting_system.config.sun_enabled is True
    assert lighting_system.config.sun_intensity == 0.8


def test_lighting_config_defaults() -> None:
    """Test LightingConfig default values match expected configuration."""
    from catley.view.render.lighting.cpu import LightingConfig

    config = LightingConfig()

    # Check default sun configuration
    assert config.sun_enabled is True  # Assuming sun is enabled by default
    assert config.sun_intensity > 0.0
    assert config.sun_elevation_degrees > 0.0
    assert config.sun_azimuth_degrees > 0.0
    assert config.sky_exposure_power > 0.0

    # Check color is a valid RGB tuple
    assert len(config.sun_color) == 3
    assert all(0 <= c <= 255 for c in config.sun_color)


def test_multiple_light_types_integration() -> None:
    """Test that static, dynamic, and directional lights work together."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add all types of lights
    static_light = StaticLight(position=(20, 20), radius=3, color=(255, 255, 255))
    dynamic_light = DynamicLight(position=(25, 25), radius=2, color=(255, 128, 0))
    sun = DirectionalLight.create_sun(intensity=0.3)

    gw.add_light(static_light)
    gw.add_light(dynamic_light)
    gw.add_light(sun)

    # Should be able to compute combined lighting
    viewport_bounds = Rect(18, 18, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (10, 10, 3)

    # Should have some illumination from multiple sources
    max_intensity = np.max(lightmap)
    assert max_intensity > 0.0


def test_lighting_system_light_management() -> None:
    """Test adding and removing lights properly updates the lighting system."""
    gw = GameWorld(30, 30)
    lighting_system = CPULightingSystem(gw)
    gw.lighting_system = lighting_system

    # Initially no lights
    assert len(gw.lights) == 0

    # Add a light
    light = StaticLight(position=(15, 15), radius=3, color=(255, 255, 255))
    gw.add_light(light)
    assert len(gw.lights) == 1
    assert light in gw.lights

    # Remove the light
    gw.remove_light(light)
    assert len(gw.lights) == 0
    assert light not in gw.lights

    # Removing non-existent light should not crash
    gw.remove_light(light)  # Should not raise exception


def test_sun_configuration_edge_cases() -> None:
    """Test DirectionalLight.create_sun() with edge case angles."""
    # Test extreme elevations
    low_sun = DirectionalLight.create_sun(elevation_degrees=0.1)  # Almost on horizon
    high_sun = DirectionalLight.create_sun(elevation_degrees=89.9)  # Almost overhead

    # Both should have normalized directions
    assert abs(math.sqrt(low_sun.direction.x**2 + low_sun.direction.y**2) - 1.0) < 1e-6
    assert (
        abs(math.sqrt(high_sun.direction.x**2 + high_sun.direction.y**2) - 1.0) < 1e-6
    )

    # Low sun should have small downward component
    assert low_sun.direction.y > -0.1

    # High sun should have large downward component
    assert high_sun.direction.y < -0.9

    # Test all cardinal directions
    north_sun = DirectionalLight.create_sun(azimuth_degrees=0.0)  # North
    east_sun = DirectionalLight.create_sun(azimuth_degrees=90.0)  # East
    south_sun = DirectionalLight.create_sun(azimuth_degrees=180.0)  # South
    west_sun = DirectionalLight.create_sun(azimuth_degrees=270.0)  # West

    # Each should have appropriate x components
    assert north_sun.direction.x < 0.1  # Little/no eastward component
    assert east_sun.direction.x > 0.5  # Strong eastward component
    assert south_sun.direction.x < 0.1  # Little/no eastward component
    assert west_sun.direction.x < -0.5  # Strong westward component
