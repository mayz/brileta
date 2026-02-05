import math
from unittest.mock import Mock

from catley import config
from catley.environment.map import MapRegion
from catley.game.game_world import GameWorld
from catley.game.lights import (
    DirectionalLight,
    DynamicLight,
    StaticLight,
    Vec2,
    _angles_to_direction,
)
from catley.view.render.lighting.base import LightingConfig


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

    # Check default values match config
    assert sun.intensity == config.SUN_INTENSITY
    assert sun.color == config.SUN_COLOR
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

    # At 90° elevation, the raw direction vector is nearly (0, 0) before
    # normalization. After normalization, it becomes a unit vector in some
    # direction determined by floating-point rounding. The key is that the
    # direction is still normalized (magnitude 1).
    magnitude = math.sqrt(noon_sun.direction.x**2 + noon_sun.direction.y**2)
    assert abs(magnitude - 1.0) < 1e-6, "Direction should be normalized"

    # Test sunrise (low elevation, east-facing)
    sunrise_sun = DirectionalLight.create_sun(
        elevation_degrees=0.0,  # On horizon
        azimuth_degrees=90.0,  # East
    )

    # Sun at east: direction points toward east (+x), no north/south component
    # At 0 degree elevation, full horizontal strength
    assert sunrise_sun.direction.x > 0.9  # Strong eastward
    assert abs(sunrise_sun.direction.y) < 1e-6  # No north/south


def test_directional_light_southeast_default() -> None:
    """Test that default sun direction matches expected southeast orientation."""
    sun = DirectionalLight.create_sun()

    # Default azimuth is 135 degrees (southeast)
    # Direction points toward sun: positive x (east) and positive y (south in screen)
    # Screen coords: +X=right(east), +Y=down(south)
    assert sun.direction.x > 0  # East component
    assert sun.direction.y > 0  # South component (SE is south of E-W line)


def test_create_sun_stores_angles() -> None:
    """Test that create_sun() stores the source azimuth and elevation."""
    sun = DirectionalLight.create_sun(azimuth_degrees=200.0, elevation_degrees=30.0)

    assert sun.azimuth_degrees == 200.0
    assert sun.elevation_degrees == 30.0


def test_set_angles_updates_direction() -> None:
    """Test that set_angles() recomputes direction from the new angles."""
    sun = DirectionalLight.create_sun(azimuth_degrees=135.0, elevation_degrees=45.0)
    old_x, old_y = sun.direction.x, sun.direction.y

    sun.set_angles(azimuth=225.0)

    # Direction should have changed
    assert sun.direction.x != old_x or sun.direction.y != old_y
    # Stored angle should be updated
    assert sun.azimuth_degrees == 225.0
    # Elevation should be unchanged
    assert sun.elevation_degrees == 45.0


def test_set_angles_partial_update() -> None:
    """Test that set_angles() only updates the provided angle."""
    sun = DirectionalLight.create_sun(azimuth_degrees=135.0, elevation_degrees=45.0)

    sun.set_angles(elevation=20.0)

    assert sun.azimuth_degrees == 135.0  # Unchanged
    assert sun.elevation_degrees == 20.0


def test_set_angles_direction_is_normalized() -> None:
    """Test that direction stays normalized after set_angles()."""
    sun = DirectionalLight.create_sun()

    sun.set_angles(azimuth=73.0, elevation=15.0)

    magnitude = math.sqrt(sun.direction.x**2 + sun.direction.y**2)
    assert abs(magnitude - 1.0) < 1e-6


def test_angles_to_direction_cardinal_directions() -> None:
    """Test _angles_to_direction for the four cardinal directions at 0 elevation."""
    # East (azimuth 90): +x, no y
    east = _angles_to_direction(90.0, 0.0)
    assert east.x > 0.99
    assert abs(east.y) < 1e-6

    # South (azimuth 180): no x, +y (screen coords)
    south = _angles_to_direction(180.0, 0.0)
    assert abs(south.x) < 1e-6
    assert south.y > 0.99

    # West (azimuth 270): -x, no y
    west = _angles_to_direction(270.0, 0.0)
    assert west.x < -0.99
    assert abs(west.y) < 1e-6

    # North (azimuth 0): no x, -y (screen coords)
    north = _angles_to_direction(0.0, 0.0)
    assert abs(north.x) < 1e-6
    assert north.y < -0.99


def test_angles_to_direction_high_elevation_shrinks_vector() -> None:
    """Test that high elevation reduces the horizontal direction magnitude."""
    low = _angles_to_direction(90.0, 10.0)
    high = _angles_to_direction(90.0, 80.0)

    low_mag = math.sqrt(low.x**2 + low.y**2)
    high_mag = math.sqrt(high.x**2 + high.y**2)

    # Higher elevation means shorter horizontal projection (sun more overhead)
    assert high_mag < low_mag


def test_set_angles_both_arguments() -> None:
    """Test that set_angles() updates both angles simultaneously."""
    sun = DirectionalLight.create_sun(azimuth_degrees=0.0, elevation_degrees=45.0)

    sun.set_angles(azimuth=180.0, elevation=30.0)

    assert sun.azimuth_degrees == 180.0
    assert sun.elevation_degrees == 30.0
    # Direction should point south (azimuth 180) in screen coords (+y)
    assert sun.direction.y > 0


def test_set_angles_clamps_elevation_below_zero() -> None:
    """Test that set_angles() clamps negative elevation to 0."""
    sun = DirectionalLight.create_sun(elevation_degrees=45.0)

    sun.set_angles(elevation=-10.0)

    assert sun.elevation_degrees == 0.0


def test_set_angles_clamps_elevation_above_90() -> None:
    """Test that set_angles() clamps elevation above 90 to 90."""
    sun = DirectionalLight.create_sun(elevation_degrees=45.0)

    sun.set_angles(elevation=120.0)

    assert sun.elevation_degrees == 90.0


def test_directional_light_init_stores_explicit_angles() -> None:
    """Test that DirectionalLight.__init__ stores angles when provided."""
    sun = DirectionalLight(
        direction=Vec2(0.5, 0.5),
        color=(255, 255, 255),
        intensity=0.8,
        azimuth_degrees=135.0,
        elevation_degrees=30.0,
    )

    assert sun.azimuth_degrees == 135.0
    assert sun.elevation_degrees == 30.0
    assert sun.intensity == 0.8


def test_directional_light_init_default_angles() -> None:
    """Test that DirectionalLight.__init__ defaults angles to 0."""
    sun = DirectionalLight(
        direction=Vec2(1.0, 0.0),
        color=(255, 255, 255),
    )

    assert sun.azimuth_degrees == 0.0
    assert sun.elevation_degrees == 0.0


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


# Lighting configuration tests
def test_lighting_config_defaults() -> None:
    """Test LightingConfig default values match expected configuration."""
    config = LightingConfig()

    # Check default sun configuration
    assert config.sun_enabled is True
    assert config.sun_intensity > 0.0
    assert config.sun_elevation_degrees > 0.0
    assert config.sun_azimuth_degrees > 0.0
    assert config.sky_exposure_power > 0.0

    # Check color is a valid RGB tuple
    assert len(config.sun_color) == 3
    assert all(0 <= c <= 255 for c in config.sun_color)


def test_lighting_config_overrides() -> None:
    """Test LightingConfig supports explicit overrides."""
    config = LightingConfig(sun_enabled=False, sun_intensity=0.5)

    assert config.sun_enabled is False
    assert config.sun_intensity == 0.5


# GameWorld lighting integration tests
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


def test_lighting_system_light_management() -> None:
    """Test adding and removing lights properly notifies the lighting system."""
    gw = GameWorld(30, 30)
    lighting_system = Mock()
    gw.lighting_system = lighting_system

    # Initially no lights
    assert len(gw.lights) == 0

    # Add a light
    light = StaticLight(position=(15, 15), radius=3, color=(255, 255, 255))
    gw.add_light(light)
    assert len(gw.lights) == 1
    assert light in gw.lights
    lighting_system.on_light_added.assert_called_once_with(light)

    # Remove the light
    gw.remove_light(light)
    assert len(gw.lights) == 0
    assert light not in gw.lights
    lighting_system.on_light_removed.assert_called_once_with(light)

    # Removing non-existent light should not crash
    gw.remove_light(light)  # Should not raise exception


def test_sun_configuration_edge_cases() -> None:
    """Test DirectionalLight.create_sun() with edge case angles."""
    # Test extreme elevations (default azimuth is 135° SE)
    low_sun = DirectionalLight.create_sun(elevation_degrees=0.1)  # Almost on horizon
    high_sun = DirectionalLight.create_sun(elevation_degrees=89.9)  # Almost overhead

    # Both should have normalized directions (magnitude 1)
    assert abs(math.sqrt(low_sun.direction.x**2 + low_sun.direction.y**2) - 1.0) < 1e-6
    assert (
        abs(math.sqrt(high_sun.direction.x**2 + high_sun.direction.y**2) - 1.0) < 1e-6
    )

    # Low sun (near horizon): full horizontal strength toward SE
    # At azimuth 135°, direction.y should be positive (south component)
    assert low_sun.direction.y > 0.5  # Strong south component

    # High sun (nearly overhead): direction is still toward SE because
    # normalization preserves direction. The raw components are tiny but
    # normalizing produces a unit vector still pointing toward SE.
    # (In practice, near-overhead sun means very short shadows, handled elsewhere)
    assert high_sun.direction.x > 0  # Still points toward east (SE)
    assert high_sun.direction.y > 0  # Still points toward south (SE)

    # Test all cardinal directions (at default 45° elevation)
    north_sun = DirectionalLight.create_sun(azimuth_degrees=0.0)  # North
    east_sun = DirectionalLight.create_sun(azimuth_degrees=90.0)  # East
    south_sun = DirectionalLight.create_sun(azimuth_degrees=180.0)  # South
    west_sun = DirectionalLight.create_sun(azimuth_degrees=270.0)  # West

    # X component: sin(azimuth) - positive for E, negative for W
    assert abs(north_sun.direction.x) < 0.1  # sin(0°) = 0
    assert east_sun.direction.x > 0.5  # sin(90°) = 1
    assert abs(south_sun.direction.x) < 0.1  # sin(180°) = 0
    assert west_sun.direction.x < -0.5  # sin(270°) = -1

    # Y component: -cos(azimuth) - positive for S, negative for N (screen coords)
    assert north_sun.direction.y < -0.5  # -cos(0°) = -1 (north in screen)
    assert abs(east_sun.direction.y) < 0.1  # -cos(90°) = 0
    assert south_sun.direction.y > 0.5  # -cos(180°) = 1 (south in screen)
    assert abs(west_sun.direction.y) < 0.1  # -cos(270°) = 0


# Controller Sun Angle Tests


def test_controller_get_sun_returns_directional_light() -> None:
    """Test that Controller._get_sun() finds the DirectionalLight."""
    from catley.controller import Controller

    gw = GameWorld(30, 30)
    sun = DirectionalLight.create_sun(azimuth_degrees=135.0)
    gw.add_light(sun)

    # Call the unbound method with a mock self that has the right .gw
    mock_self = Mock()
    mock_self.gw = gw
    result = Controller._get_sun(mock_self)

    assert result is sun


def test_controller_get_sun_returns_none_without_directional() -> None:
    """Test that _get_sun() returns None when no DirectionalLight exists."""
    from catley.controller import Controller

    gw = GameWorld(30, 30)
    gw.add_light(StaticLight(position=(5, 5), radius=3, color=(255, 255, 255)))

    mock_self = Mock()
    mock_self.gw = gw
    result = Controller._get_sun(mock_self)

    assert result is None


def test_controller_set_sun_angle_updates_light_and_invalidates() -> None:
    """Test that _set_sun_angle() updates angles and invalidates lighting cache."""
    from catley.controller import Controller

    gw = GameWorld(30, 30)
    sun = DirectionalLight.create_sun(azimuth_degrees=135.0, elevation_degrees=45.0)
    gw.add_light(sun)

    mock_lighting = Mock()
    gw.lighting_system = mock_lighting

    mock_self = Mock()
    mock_self.gw = gw
    mock_self._get_sun = lambda: Controller._get_sun(mock_self)

    Controller._set_sun_angle(mock_self, azimuth=200.0)

    assert sun.azimuth_degrees == 200.0
    assert sun.elevation_degrees == 45.0  # Unchanged
    mock_lighting.on_global_light_changed.assert_called_once()


def test_controller_set_sun_angle_elevation() -> None:
    """Test that _set_sun_angle() can update elevation."""
    from catley.controller import Controller

    gw = GameWorld(30, 30)
    sun = DirectionalLight.create_sun(azimuth_degrees=135.0, elevation_degrees=45.0)
    gw.add_light(sun)

    mock_lighting = Mock()
    gw.lighting_system = mock_lighting

    mock_self = Mock()
    mock_self.gw = gw
    mock_self._get_sun = lambda: Controller._get_sun(mock_self)

    Controller._set_sun_angle(mock_self, elevation=20.0)

    assert sun.azimuth_degrees == 135.0  # Unchanged
    assert sun.elevation_degrees == 20.0
    mock_lighting.on_global_light_changed.assert_called_once()


# Player Torch Auto-Toggle Tests


class TestPlayerTorchAutoToggle:
    """Tests for automatic player torch enable/disable based on sky exposure.

    The torch auto-toggle feature disables the player's torch in well-lit outdoor
    areas (high sky exposure) and re-enables it indoors. Uses hysteresis to prevent
    flickering at doorways: torch turns OFF at sky_exposure >= 0.7 but only turns
    back ON at sky_exposure <= 0.3.
    """

    def setup_method(self):
        """Set up test fixtures with minimal mocking."""
        # Test _update_player_torch in isolation by creating a mock controller
        # with the necessary attributes rather than instantiating the full Controller
        self.mock_controller = Mock()
        self.mock_torch = Mock(spec=DynamicLight)
        self.mock_controller._player_torch = self.mock_torch
        self.mock_controller._player_torch_active = True

        # Mock game world with player and game_map
        self.mock_gw = Mock()
        self.mock_player = Mock()
        self.mock_player.x = 5
        self.mock_player.y = 5
        self.mock_gw.player = self.mock_player
        self.mock_controller.gw = self.mock_gw

        # Import the actual method and bind it to our mock
        from catley.controller import Controller

        self._update_player_torch = Controller._update_player_torch.__get__(
            self.mock_controller, Controller
        )

    def test_torch_turns_off_at_high_sky_exposure(self):
        """Torch should disable when sky_exposure >= 0.7 (clearly outdoors)."""
        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=0.7
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=outdoor_region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.remove_light.assert_called_once_with(self.mock_torch)
        assert self.mock_controller._player_torch_active is False

    def test_torch_turns_off_at_very_high_sky_exposure(self):
        """Torch should disable when sky_exposure is 1.0 (full outdoor)."""
        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=1.0
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=outdoor_region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.remove_light.assert_called_once_with(self.mock_torch)
        assert self.mock_controller._player_torch_active is False

    def test_torch_turns_on_at_low_sky_exposure(self):
        """Torch should enable when sky_exposure <= 0.3 (clearly indoors)."""
        indoor_region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.3
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=indoor_region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_called_once_with(self.mock_torch)
        assert self.mock_controller._player_torch_active is True

    def test_torch_turns_on_at_zero_sky_exposure(self):
        """Torch should enable when sky_exposure is 0.0 (fully indoors)."""
        indoor_region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.0
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=indoor_region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_called_once_with(self.mock_torch)
        assert self.mock_controller._player_torch_active is True

    def test_hysteresis_torch_stays_off_in_dead_zone(self):
        """Torch should stay OFF when 0.3 < sky_exposure < 0.7 (hysteresis)."""
        partial_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="partial", sky_exposure=0.5
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=partial_region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()
        assert self.mock_controller._player_torch_active is False

    def test_hysteresis_torch_stays_on_in_dead_zone(self):
        """Torch should stay ON when 0.3 < sky_exposure < 0.7 (hysteresis)."""
        partial_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="partial", sky_exposure=0.5
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=partial_region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()
        assert self.mock_controller._player_torch_active is True

    def test_hysteresis_prevents_flickering_at_boundary(self):
        """Simulates walking through a doorway - torch shouldn't flicker."""
        # Sequence: indoors (0.2) -> doorway (0.5) -> outdoors (0.8)
        #        -> doorway (0.5) -> indoors (0.2)

        indoor_region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.2
        )
        doorway_region = MapRegion.create_outdoor_region(
            map_region_id=2, region_type="partial", sky_exposure=0.5
        )
        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=3, region_type="outdoor", sky_exposure=0.8
        )

        # Start indoors with torch on
        self.mock_controller._player_torch_active = True

        # Step 1: Stay indoors - torch stays on
        self.mock_gw.game_map.get_region_at = Mock(return_value=indoor_region)
        self._update_player_torch()
        assert self.mock_controller._player_torch_active is True

        # Step 2: Enter doorway (dead zone) - torch stays on
        self.mock_gw.game_map.get_region_at = Mock(return_value=doorway_region)
        self.mock_gw.reset_mock()
        self._update_player_torch()
        assert self.mock_controller._player_torch_active is True
        self.mock_gw.remove_light.assert_not_called()

        # Step 3: Go fully outdoors - torch turns off
        self.mock_gw.game_map.get_region_at = Mock(return_value=outdoor_region)
        self.mock_gw.reset_mock()
        self._update_player_torch()
        assert self.mock_controller._player_torch_active is False
        self.mock_gw.remove_light.assert_called_once()

        # Step 4: Back to doorway (dead zone) - torch stays off
        self.mock_gw.game_map.get_region_at = Mock(return_value=doorway_region)
        self.mock_gw.reset_mock()
        self._update_player_torch()
        assert self.mock_controller._player_torch_active is False
        self.mock_gw.add_light.assert_not_called()

        # Step 5: Back indoors - torch turns on
        self.mock_gw.game_map.get_region_at = Mock(return_value=indoor_region)
        self.mock_gw.reset_mock()
        self._update_player_torch()
        assert self.mock_controller._player_torch_active is True
        self.mock_gw.add_light.assert_called_once()

    def test_no_change_when_already_correct_state_indoors(self):
        """Torch should not be re-added if already active indoors."""
        indoor_region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.1
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=indoor_region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()

    def test_no_change_when_already_correct_state_outdoors(self):
        """Torch should not be re-removed if already inactive outdoors."""
        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=0.9
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=outdoor_region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()

    def test_graceful_handling_missing_get_region_at(self):
        """Should handle game_map without get_region_at method gracefully."""
        mock_game_map = Mock(spec=[])  # Empty spec means no attributes
        self.mock_gw.game_map = mock_game_map
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        assert self.mock_controller._player_torch_active is True
        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()

    def test_graceful_handling_none_region(self):
        """Should handle None region (e.g., unexplored area) gracefully."""
        self.mock_gw.game_map.get_region_at = Mock(return_value=None)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        assert self.mock_controller._player_torch_active is True
        self.mock_gw.add_light.assert_not_called()
        self.mock_gw.remove_light.assert_not_called()

    def test_threshold_boundary_exactly_at_turn_off(self):
        """Test exact boundary: sky_exposure = 0.7 should turn torch off."""
        region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=0.7
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.remove_light.assert_called_once()
        assert self.mock_controller._player_torch_active is False

    def test_threshold_boundary_exactly_at_turn_on(self):
        """Test exact boundary: sky_exposure = 0.3 should turn torch on."""
        region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.3
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_called_once()
        assert self.mock_controller._player_torch_active is True

    def test_threshold_boundary_just_below_turn_off(self):
        """Test just below turn-off threshold: 0.69 should NOT turn off."""
        region = MapRegion.create_outdoor_region(
            map_region_id=1, region_type="outdoor", sky_exposure=0.69
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=region)
        self.mock_controller._player_torch_active = True

        self._update_player_torch()

        self.mock_gw.remove_light.assert_not_called()
        assert self.mock_controller._player_torch_active is True

    def test_threshold_boundary_just_above_turn_on(self):
        """Test just above turn-on threshold: 0.31 should NOT turn on."""
        region = MapRegion.create_indoor_region(
            map_region_id=1, region_type="indoor", sky_exposure=0.31
        )
        self.mock_gw.game_map.get_region_at = Mock(return_value=region)
        self.mock_controller._player_torch_active = False

        self._update_player_torch()

        self.mock_gw.add_light.assert_not_called()
        assert self.mock_controller._player_torch_active is False
