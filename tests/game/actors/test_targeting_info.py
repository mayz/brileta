"""Tests for actor-provided targeting info."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actors import Actor, Character
from catley.game.actors.container import Container, ItemPile
from catley.game.enums import ItemSize
from catley.game.items.item_core import ItemType


@dataclass
class DummyController:
    """Minimal controller placeholder for contextual action generation."""

    def start_plan(self, *args: object, **kwargs: object) -> bool:
        return True


class TestActorTargetingInfo:
    """Validate the actor-provided targeting information helpers."""

    def test_actor_defaults(self) -> None:
        """Base Actor should provide empty targeting info."""
        actor = Actor(0, 0, "A", colors.WHITE, name="Test Actor")
        player = Character(1, 1, "@", colors.WHITE, "Player")

        controller = cast(Controller, DummyController())

        assert actor.get_target_description() is None
        assert actor.get_contextual_actions(controller, player) == []

    def test_container_description(self) -> None:
        """Containers should describe themselves as containers."""
        container = Container(0, 0)

        assert container.get_target_description() == "A container"

    def test_item_pile_description_single_item(self) -> None:
        """Item piles should describe a single item on the ground."""
        item_type = ItemType(
            name="Widget",
            description="A test widget.",
            size=ItemSize.TINY,
        )
        item = item_type.create()
        pile = ItemPile(0, 0, items=[item])

        assert pile.get_target_description() == "An item on the ground"

    def test_item_pile_actions_when_away(self) -> None:
        """Item piles should offer pickup actions when the player is away."""
        item_type = ItemType(
            name="Widget",
            description="A test widget.",
            size=ItemSize.TINY,
        )
        item = item_type.create()
        pile = ItemPile(0, 0, items=[item])
        player = Character(1, 1, "@", colors.WHITE, "Player")

        controller = cast(Controller, DummyController())

        actions = pile.get_contextual_actions(controller, player)

        assert len(actions) == 1
        assert actions[0].id == "pickup-walk"
