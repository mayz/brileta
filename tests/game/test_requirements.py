from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.base import GameIntent
from catley.game.actions.combat import AttackIntent
from catley.game.actions.discovery import ActionDiscovery
from catley.game.actions.discovery.types import ActionRequirement, CombatIntentCache
from catley.game.actions.environment import OpenDoorIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.view.ui.action_browser_state import ActionBrowserStateMachine
from tests.game.actions.test_action_discovery import _make_combat_world
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None
    combat_intent_cache: CombatIntentCache | None = None
    queued_action: GameIntent | None = None

    def create_resolver(self, **kwargs: object) -> object:
        from catley.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]

    def queue_action(self, action: GameIntent) -> None:
        self.queued_action = action


def test_attack_requirement_handoff() -> None:
    base_controller, player, melee_target, _, _ = _make_combat_world()
    controller = DummyController(gw=base_controller.gw)

    discovery = ActionDiscovery()
    sm = ActionBrowserStateMachine(discovery)
    context = discovery.context_builder.build_context(
        cast(Controller, controller), player
    )
    combat_opts = discovery.combat_discovery.discover_combat_actions(
        cast(Controller, controller), player, context
    )
    melee_option = next(
        o for o in combat_opts if o.static_params.get("attack_mode") == "melee"
    )

    sm.set_current_action(melee_option)
    target_opts = sm.get_options_for_current_state(cast(Controller, controller), player)
    select_opt = next(o for o in target_opts if o.id.endswith(melee_target.name))
    assert select_opt.execute is not None
    select_opt.execute()
    sm.get_options_for_current_state(cast(Controller, controller), player)

    assert isinstance(controller.queued_action, AttackIntent)
    assert controller.queued_action.defender == melee_target


def test_environment_target_tile_requirement() -> None:
    gw = DummyGameWorld()
    # Place two doors to test the tile selection requirement
    gw.game_map.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    gw.game_map.tiles[0, 1] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    controller = DummyController(gw=gw)
    discovery = ActionDiscovery()
    sm = ActionBrowserStateMachine(discovery)
    context = discovery.context_builder.build_context(
        cast(Controller, controller), player
    )
    env_opts = discovery.environment_discovery.discover_environment_actions(
        cast(Controller, controller), player, context
    )
    open_option = next(o for o in env_opts if o.id == "open-door")
    assert open_option.requirements == [ActionRequirement.TARGET_TILE]

    sm.set_current_action(open_option)
    tile_opts = sm.get_options_for_current_state(cast(Controller, controller), player)
    select_tile = next(o for o in tile_opts if o.id == "select-tile-1-0")
    assert select_tile.execute is not None
    select_tile.execute()
    sm.get_options_for_current_state(cast(Controller, controller), player)

    action = controller.queued_action
    assert isinstance(action, OpenDoorIntent)
    assert (action.x, action.y) == (1, 0)
