"""Tests for optional EnergyComponent on actors.

This module tests that:
- Static objects (Containers) don't have energy budgets
- Active actors (Characters, ContainedFire) do have energy budgets
- The turn manager correctly filters actors based on energy presence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actors.components import EnergyComponent
from catley.game.actors.container import Container, create_bookcase
from catley.game.actors.core import Actor, Character
from catley.game.actors.environmental import ContainedFire
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld

# =============================================================================
# CONTAINER ENERGY TESTS
# =============================================================================


class TestContainerEnergy:
    """Tests verifying containers don't have energy budgets."""

    def test_container_has_no_energy(self) -> None:
        """Container instances should have energy = None."""
        container = Container(x=0, y=0)
        assert container.energy is None

    def test_bookcase_has_no_energy(self) -> None:
        """Bookcases (created via factory) should have energy = None."""
        bookcase = create_bookcase(x=0, y=0)
        assert bookcase.energy is None

    def test_container_has_no_energy_attribute_for_hasattr(self) -> None:
        """Containers should still have the energy attribute (just set to None).

        This is important because hasattr checks are used to filter actors,
        but the attribute exists - it's just None.
        """
        container = Container(x=0, y=0)
        # The attribute exists but is None
        assert hasattr(container, "energy")
        assert container.energy is None


# =============================================================================
# CHARACTER ENERGY TESTS
# =============================================================================


class TestCharacterEnergy:
    """Tests verifying characters have energy budgets."""

    def test_character_has_energy(self) -> None:
        """Character instances should have a valid EnergyComponent."""
        character = Character(
            x=0,
            y=0,
            ch="@",
            color=(255, 255, 255),
            name="Test Character",
        )
        assert character.energy is not None
        assert isinstance(character.energy, EnergyComponent)

    def test_character_energy_has_actor_reference(self) -> None:
        """Character's EnergyComponent should have back-reference to the character."""
        character = Character(
            x=0,
            y=0,
            ch="@",
            color=(255, 255, 255),
            name="Test Character",
        )
        assert character.energy is not None
        assert character.energy.actor is character

    def test_character_energy_respects_speed(self) -> None:
        """Character's EnergyComponent should use the provided speed."""
        character = Character(
            x=0,
            y=0,
            ch="@",
            color=(255, 255, 255),
            name="Fast Character",
            speed=150,
        )
        assert character.energy is not None
        assert character.energy.speed == 150


# =============================================================================
# CONTAINED FIRE ENERGY TESTS
# =============================================================================


class TestContainedFireEnergy:
    """Tests verifying ContainedFire has energy budget."""

    def test_contained_fire_has_energy(self) -> None:
        """ContainedFire instances should have a valid EnergyComponent."""
        fire = ContainedFire(
            x=0,
            y=0,
            ch=".",
            color=(255, 100, 0),
            name="Test Fire",
        )
        assert fire.energy is not None
        assert isinstance(fire.energy, EnergyComponent)

    def test_contained_fire_energy_has_actor_reference(self) -> None:
        """ContainedFire's EnergyComponent should have back-reference."""
        fire = ContainedFire(
            x=0,
            y=0,
            ch=".",
            color=(255, 100, 0),
            name="Test Fire",
        )
        assert fire.energy is not None
        assert fire.energy.actor is fire


# =============================================================================
# BASE ACTOR ENERGY TESTS
# =============================================================================


class TestBaseActorEnergy:
    """Tests for base Actor energy behavior."""

    def test_actor_without_energy_param_has_none(self) -> None:
        """Base Actor without energy param should have energy = None."""
        actor = Actor(
            x=0,
            y=0,
            ch="?",
            color=(128, 128, 128),
            name="Plain Actor",
        )
        assert actor.energy is None

    def test_actor_with_energy_param_has_energy(self) -> None:
        """Base Actor with energy param should have that EnergyComponent."""
        energy = EnergyComponent(speed=120)
        actor = Actor(
            x=0,
            y=0,
            ch="?",
            color=(128, 128, 128),
            name="Active Actor",
            energy=energy,
        )
        assert actor.energy is energy
        assert actor.energy is not None
        assert actor.energy.speed == 120

    def test_actor_sets_energy_back_reference(self) -> None:
        """Actor should set the back-reference on the EnergyComponent."""
        energy = EnergyComponent(speed=100)
        assert energy.actor is None  # Not set yet

        actor = Actor(
            x=0,
            y=0,
            ch="?",
            color=(128, 128, 128),
            name="Test Actor",
            energy=energy,
        )

        assert energy.actor is actor


# =============================================================================
# ENERGY COMPONENT TESTS
# =============================================================================


class TestEnergyComponentLateBinding:
    """Tests for EnergyComponent late binding of actor reference."""

    def test_energy_component_created_without_actor(self) -> None:
        """EnergyComponent can be created without actor reference."""
        energy = EnergyComponent(speed=100)
        assert energy.actor is None
        assert energy.speed == 100

    def test_energy_component_speed_based_amount_requires_actor(self) -> None:
        """get_speed_based_energy_amount should assert actor is set."""
        import pytest

        energy = EnergyComponent(speed=100)

        # Should raise AssertionError because actor is None
        with pytest.raises(AssertionError, match="actor must be set"):
            energy.get_speed_based_energy_amount()

    def test_energy_component_basic_methods_work_without_actor(self) -> None:
        """Basic energy methods should work without actor reference."""
        energy = EnergyComponent(speed=100)

        # These should all work without actor
        assert energy.can_afford(50)
        energy.accumulate_energy(50)
        assert energy.accumulated_energy == 150  # 100 initial + 50
        energy.spend(30)
        assert energy.accumulated_energy == 120


# =============================================================================
# TURN MANAGER INTEGRATION TESTS
# =============================================================================


@dataclass
class DummyController:
    """Minimal controller for testing TurnManager integration."""

    gw: DummyGameWorld
    frame_manager: object | None = None

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()

    def update_fov(self) -> None:
        pass


class TestTurnManagerEnergyCache:
    """Tests verifying turn manager correctly filters actors by energy."""

    def test_containers_excluded_from_energy_cache(self) -> None:
        """Containers should not appear in _energy_actors_cache."""
        gw = DummyGameWorld()
        player = Character(
            0,
            0,
            "@",
            colors.WHITE,
            "Player",
            game_world=cast(GameWorld, gw),
        )
        container = Container(x=5, y=5)
        container.gw = cast(GameWorld, gw)

        gw.player = player
        gw.add_actor(player)
        gw.add_actor(container)

        controller = DummyController(gw=gw)

        # Force cache update
        controller.turn_manager._update_energy_actors_cache()

        # Container should not be in the cache
        assert container not in controller.turn_manager._energy_actors_cache
        # Player should be in the cache
        assert player in controller.turn_manager._energy_actors_cache

    def test_containers_dont_accumulate_energy_on_player_action(self) -> None:
        """When player acts, containers should not accumulate energy."""
        gw = DummyGameWorld()
        player = Character(
            0,
            0,
            "@",
            colors.WHITE,
            "Player",
            game_world=cast(GameWorld, gw),
        )
        container = Container(x=5, y=5)
        container.gw = cast(GameWorld, gw)

        gw.player = player
        gw.add_actor(player)
        gw.add_actor(container)

        controller = DummyController(gw=gw)

        # Record initial energy states
        assert player.energy is not None
        initial_player_energy = player.energy.accumulated_energy

        # Simulate player action
        controller.turn_manager.on_player_action()

        # Player should have accumulated energy
        assert player.energy.accumulated_energy > initial_player_energy
        # Container should still have no energy component
        assert container.energy is None
