"""Tests for Actor.heal() method and healing floating text.

This module tests the Actor.heal() method which provides:
- Healing HP (specific amount or full restore)
- Visual feedback (green flash)
- Floating text display showing amount healed
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

from brileta import colors
from brileta.events import FloatingTextEvent, FloatingTextValence
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


def make_character() -> Character:
    """Create a test character with health component.

    Creates a character with toughness=10 so max_hp=15, giving room to test healing.
    """
    gw = DummyGameWorld()
    actor = Character(
        0,
        0,
        "A",
        colors.WHITE,
        "Test",
        game_world=cast(GameWorld, gw),
        toughness=10,  # max_hp = 10 + 5 = 15
    )
    gw.add_actor(actor)
    return actor


class TestActorHeal:
    """Tests for Actor.heal() method."""

    def test_heal_specific_amount(self) -> None:
        """heal(amount) should heal the specified amount."""
        actor = make_character()
        actor.health._hp = 5

        healed = actor.heal(3)

        assert healed == 3
        assert actor.health.hp == 8

    def test_heal_returns_actual_amount_healed(self) -> None:
        """heal() should return actual amount healed when capped at max."""
        actor = make_character()
        actor.health._hp = actor.health.max_hp - 2  # 2 HP below max

        healed = actor.heal(10)  # Try to heal 10

        assert healed == 2  # Only healed 2
        assert actor.health.hp == actor.health.max_hp

    def test_heal_none_restores_to_full(self) -> None:
        """heal(None) should restore to full HP."""
        actor = make_character()
        actor.health._hp = 1

        healed = actor.heal(None)

        assert actor.health.hp == actor.health.max_hp
        assert healed == actor.health.max_hp - 1

    def test_heal_no_argument_restores_to_full(self) -> None:
        """heal() with no argument should restore to full HP."""
        actor = make_character()
        actor.health._hp = 1

        actor.heal()

        assert actor.health.hp == actor.health.max_hp

    def test_heal_at_full_hp_returns_zero(self) -> None:
        """heal() at full HP should return 0 and not emit floating text."""
        actor = make_character()
        assert actor.health.hp == actor.health.max_hp

        with patch("brileta.game.actors.core.publish_event") as mock_publish:
            healed = actor.heal(10)

        assert healed == 0
        mock_publish.assert_not_called()

    def test_heal_without_health_component_returns_zero(self) -> None:
        """heal() on actor without health component should return 0."""
        gw = DummyGameWorld()
        from brileta.game.actors import Actor

        actor = Actor(
            0, 0, "A", colors.WHITE, "NoHealth", game_world=cast(GameWorld, gw)
        )
        actor.health = None

        healed = actor.heal(10)

        assert healed == 0

    def test_heal_emits_floating_text(self) -> None:
        """heal() should emit a FloatingTextEvent with positive valence."""
        actor = make_character()
        actor.health._hp = 5

        with patch("brileta.game.actors.core.publish_event") as mock_publish:
            actor.heal(3)

        mock_publish.assert_called_once()
        event = mock_publish.call_args[0][0]
        assert isinstance(event, FloatingTextEvent)
        assert event.text == "+3"
        assert event.valence == FloatingTextValence.POSITIVE
        assert event.target_actor_id == id(actor)

    def test_heal_triggers_green_flash(self) -> None:
        """heal() should trigger a green flash on visual effects."""
        actor = make_character()
        actor.health._hp = 5

        with patch("brileta.game.actors.core.publish_event"):
            actor.heal(3)

        # Check that flash was called with green
        flash_color = actor.visual_effects.get_flash_color()
        assert flash_color == colors.GREEN

    def test_heal_full_restore_emits_correct_amount(self) -> None:
        """Full restore heal should emit floating text with actual amount healed."""
        actor = make_character()
        actor.health._hp = 3
        expected_heal = actor.health.max_hp - 3

        with patch("brileta.game.actors.core.publish_event") as mock_publish:
            actor.heal()  # Full restore

        event = mock_publish.call_args[0][0]
        assert event.text == f"+{expected_heal}"
