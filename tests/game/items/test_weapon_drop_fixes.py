from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.events import MessageEvent, reset_event_bus_for_testing, subscribe_to_event
from catley.game.actors import Character
from catley.game.consequences import ConsequenceHandler
from catley.game.game_world import GameWorld
from catley.game.items.item_types import FISTS_TYPE, PISTOL_TYPE
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.dual_pane_menu import DualPaneMenu, ExternalInventory
from tests.helpers import DummyGameWorld, _make_renderer


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []
        subscribe_to_event(MessageEvent, lambda e: self.add_message(e.text))

    def add_message(self, text: str, *_args, **_kwargs) -> None:
        self.messages.append(text)


@dataclass
class DummyFrameManager:
    def get_visible_bounds(self) -> None:
        return None

    def create_effect(self, *args, **kwargs) -> None:
        pass

    def trigger_screen_shake(self, *_args, **_kwargs) -> None:
        pass


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager
    graphics: GraphicsContext


def make_world() -> tuple[DummyController, Character]:
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    actor = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    gw.player = actor
    controller = DummyController(
        gw=gw,
        message_log=DummyMessageLog(),
        frame_manager=DummyFrameManager(),
        graphics=_make_renderer(),
    )
    return controller, actor


def test_unarmed_weapon_drop_creates_no_actor() -> None:
    controller, actor = make_world()
    handler = ConsequenceHandler()
    handler._apply_weapon_drop(actor, FISTS_TYPE.create())
    # Only the actor should exist; no dropped item actors added
    assert gw_actor_count(controller.gw) == 1


def test_dual_pane_menu_removes_empty_container() -> None:
    """Test that DualPaneMenu removes empty ground containers after pickup."""
    controller, actor = make_world()
    # Equip and drop a pistol
    pistol = PISTOL_TYPE.create()
    actor.inventory.equip_to_slot(pistol, 0)
    handler = ConsequenceHandler()
    handler._apply_weapon_drop(actor, pistol)

    # There should now be a dropped item actor on the same tile
    ground_actor = next(
        a
        for a in controller.gw.actors
        if a is not actor and a.x == actor.x and a.y == actor.y
    )
    assert ground_actor.inventory is not None
    assert pistol in ground_actor.inventory

    source = ExternalInventory((actor.x, actor.y), "On the ground")
    menu = DualPaneMenu(cast(Controller, controller), source=source)
    menu._transfer_to_inventory(pistol)

    # Dropped container should be removed from world after pickup
    assert gw_actor_count(controller.gw) == 1


def test_dual_pane_pickup_from_dead_actor_clears_slot() -> None:
    """Test that picking up from dead actors clears their equipment slots."""
    controller, _player = make_world()
    npc = Character(
        2,
        1,
        "B",
        colors.WHITE,
        "NPC",
        game_world=cast(GameWorld, controller.gw),
        starting_weapon=PISTOL_TYPE.create(),
    )
    controller.gw.add_actor(npc)
    npc.health._hp = 0

    pistol = npc.inventory.ready_slots[0]
    assert pistol is not None

    source = ExternalInventory((npc.x, npc.y), "Dead NPC")
    menu = DualPaneMenu(cast(Controller, controller), source=source)
    menu._transfer_to_inventory(pistol)

    assert controller.gw.get_pickable_items_at_location(npc.x, npc.y) == []
    assert all(slot is None for slot in npc.inventory.ready_slots)


def gw_actor_count(gw: DummyGameWorld) -> int:
    return len(gw.actors)
