"""Tests for the player torch hysteresis logic.

The torch auto-toggles based on sky exposure at the player's position to avoid
washing out sun shadows in outdoor areas. It uses hysteresis thresholds to
prevent flickering at doorways:
- OFF when sky_exposure >= 0.7 (clearly outdoors)
- ON  when sky_exposure <= 0.3 (clearly indoors)
- No change between 0.3 and 0.7 (dead zone)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from brileta.controller import Controller
from brileta.environment.map import MapRegion
from tests.helpers import get_controller_with_dummy_world


def _place_player_in_region(
    controller: Controller,
    sky_exposure: float,
    region_id: int = 0,
) -> MapRegion:
    """Assign a MapRegion with the given sky_exposure at the player's position."""
    player = controller.gw.player
    gm = controller.gw.game_map

    region = MapRegion(id=region_id, region_type="room", sky_exposure=sky_exposure)
    gm.regions[region_id] = region
    gm.tile_to_region_id[player.x, player.y] = region_id
    return region


def _make_controller() -> Controller:
    """Create a controller with torch tracking initialized."""
    controller = get_controller_with_dummy_world()
    # Simulate the torch state that new_world() would set up
    controller._player_torch = MagicMock(name="player_torch")
    controller._player_torch_active = True
    controller.gw.add_light(controller._player_torch)
    return controller


class TestPlayerTorchHysteresis:
    """Torch auto-toggle uses hysteresis to prevent flickering at doorways."""

    def test_torch_turns_off_at_high_sky_exposure(self) -> None:
        """Torch disables when sky_exposure >= 0.7 (clearly outdoors)."""
        controller = _make_controller()
        _place_player_in_region(controller, sky_exposure=0.7)

        controller._update_player_torch()

        assert not controller._player_torch_active
        assert controller._player_torch not in controller.gw.lights

    def test_torch_stays_on_in_dead_zone(self) -> None:
        """Torch stays ON when sky_exposure is between thresholds (0.3 < x < 0.7)."""
        controller = _make_controller()
        _place_player_in_region(controller, sky_exposure=0.5)

        controller._update_player_torch()

        assert controller._player_torch_active

    def test_torch_turns_on_at_low_sky_exposure(self) -> None:
        """Torch re-enables when sky_exposure <= 0.3 (clearly indoors)."""
        controller = _make_controller()
        controller._player_torch_active = False
        _place_player_in_region(controller, sky_exposure=0.3)

        controller._update_player_torch()

        assert controller._player_torch_active
        assert controller._player_torch in controller.gw.lights

    def test_torch_stays_off_in_dead_zone(self) -> None:
        """Torch stays OFF when sky_exposure is between thresholds."""
        controller = _make_controller()
        controller._player_torch_active = False
        _place_player_in_region(controller, sky_exposure=0.5)

        controller._update_player_torch()

        assert not controller._player_torch_active

    def test_full_hysteresis_cycle(self) -> None:
        """Walk through a complete indoor -> outdoor -> dead zone -> indoor cycle."""
        controller = _make_controller()
        player = controller.gw.player
        gm = controller.gw.game_map

        # Create regions at different positions along a hallway
        indoor_region = MapRegion(id=0, region_type="room", sky_exposure=0.1)
        outdoor_region = MapRegion(id=1, region_type="outdoor", sky_exposure=0.9)
        doorway_region = MapRegion(id=2, region_type="doorway", sky_exposure=0.5)
        gm.regions = {0: indoor_region, 1: outdoor_region, 2: doorway_region}

        # Start indoors - torch is on
        assert controller._player_torch_active
        gm.tile_to_region_id[player.x, player.y] = 0
        controller._update_player_torch()
        assert controller._player_torch_active, "Should stay on indoors"

        # Move outdoors - torch turns off
        gm.tile_to_region_id[player.x, player.y] = 1
        controller._update_player_torch()
        assert not controller._player_torch_active, "Should turn off outdoors"

        # Move to doorway (dead zone) - torch stays off
        gm.tile_to_region_id[player.x, player.y] = 2
        controller._update_player_torch()
        assert not controller._player_torch_active, "Should stay off in dead zone"

        # Move back indoors - torch turns on again
        gm.tile_to_region_id[player.x, player.y] = 0
        controller._update_player_torch()
        assert controller._player_torch_active, "Should turn back on indoors"

    def test_no_region_is_a_noop(self) -> None:
        """If the player isn't in any region, torch state doesn't change."""
        controller = _make_controller()
        # tile_to_region_id defaults to -1 (no region) in DummyGameWorld
        original_state = controller._player_torch_active

        controller._update_player_torch()

        assert controller._player_torch_active == original_state
