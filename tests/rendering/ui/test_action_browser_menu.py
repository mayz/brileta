from typing import cast

from catley.controller import Controller
from catley.game.actions.discovery import CombatIntentCache
from catley.util.message_log import MessageLog
from catley.view.ui.action_browser_menu import ActionBrowserMenu
from tests.game.actions.test_action_discovery import _make_combat_world
from tests.rendering.backends.test_canvases import _make_renderer


def test_continue_option_shown_when_target_dead() -> None:
    controller, _player, melee_target, _ranged_target, pistol = _make_combat_world()

    # Provide required dependencies for the menu.
    controller.graphics = _make_renderer()
    controller.message_log = MessageLog()

    # Simulate a cached attack that killed the target.
    melee_target.health._hp = 0
    controller.combat_intent_cache = CombatIntentCache(
        weapon=pistol,
        attack_mode="ranged",
        target=melee_target,
    )

    menu = ActionBrowserMenu(cast(Controller, controller))
    menu.populate_options()

    assert menu.options
    assert menu.options[0].text.startswith("[Enter] Continue")
