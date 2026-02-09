"""Tests for ConsumableEffectType.HEAL with the unified API.

The HEAL effect type supports two modes:
- effect_value=N: Heal N hit points (partial heal)
- effect_value=None: Restore to full HP (full restore)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from brileta import colors
from brileta.controller import Controller
from brileta.game.actors import Character
from brileta.game.enums import ConsumableEffectType
from brileta.game.game_world import GameWorld
from brileta.game.items.capabilities import ConsumableEffect, ConsumableEffectSpec
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character]:
    """Create test world with character that has room to heal.

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
    gw.player = actor
    gw.add_actor(actor)
    controller = DummyController(gw=gw)
    return controller, actor


class TestConsumableHealPartial:
    """Tests for HEAL with specific effect_value (partial healing)."""

    def test_heal_specific_amount(self) -> None:
        """HEAL with effect_value heals that specific amount."""
        controller, actor = make_world()
        actor.health._hp = 5

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=3,
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event"):
            result = effect.consume(actor, controller)

        assert result is True
        assert actor.health.hp == 8

    def test_heal_capped_at_max_hp(self) -> None:
        """HEAL with effect_value doesn't exceed max HP."""
        controller, actor = make_world()
        actor.health._hp = actor.health.max_hp - 2

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=10,
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event"):
            effect.consume(actor, controller)

        assert actor.health.hp == actor.health.max_hp


class TestConsumableHealFull:
    """Tests for HEAL with effect_value=None (full restore)."""

    def test_heal_full_restore_with_none(self) -> None:
        """HEAL with effect_value=None restores to full HP."""
        controller, actor = make_world()
        actor.health._hp = 1

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=None,
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event"):
            effect.consume(actor, controller)

        assert actor.health.hp == actor.health.max_hp

    def test_heal_full_restore_without_effect_value(self) -> None:
        """HEAL without effect_value argument restores to full HP."""
        controller, actor = make_world()
        actor.health._hp = 1

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            # No effect_value - defaults to None
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event"):
            effect.consume(actor, controller)

        assert actor.health.hp == actor.health.max_hp


class TestConsumableHealAlreadyFull:
    """Tests for HEAL when already at full HP."""

    def test_heal_at_full_hp_shows_message(self) -> None:
        """HEAL at full HP shows 'Already at full HP' message."""
        controller, actor = make_world()
        assert actor.health.hp == actor.health.max_hp

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=10,
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event") as mock_publish:
            effect.consume(actor, controller)

        # Should show "Already at full HP" message
        call_args = mock_publish.call_args[0][0]
        assert "Already at full HP" in call_args.text

    def test_heal_at_full_hp_still_consumes_use(self) -> None:
        """HEAL at full HP still consumes a use."""
        controller, actor = make_world()
        assert actor.health.hp == actor.health.max_hp

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=10,
            max_uses=2,
        )
        effect = ConsumableEffect(spec)

        with patch("brileta.events.publish_event"):
            effect.consume(actor, controller)

        assert effect.uses_remaining == 1


class TestConsumableHealTriggersFloatingText:
    """Tests that HEAL triggers floating text via Actor.heal()."""

    def test_heal_triggers_actor_heal_floating_text(self) -> None:
        """HEAL should trigger floating text via Actor.heal()."""
        controller, actor = make_world()
        actor.health._hp = 5

        spec = ConsumableEffectSpec(
            effect_type=ConsumableEffectType.HEAL,
            effect_value=3,
        )
        effect = ConsumableEffect(spec)

        # Don't mock the actor.heal call - let it run
        with patch("brileta.events.publish_event"):
            effect.consume(actor, controller)

        # The actor's visual effects should have been flashed green
        flash_color = actor.visual_effects.get_flash_color()
        assert flash_color == colors.GREEN
