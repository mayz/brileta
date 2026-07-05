"""Conversation scenarios driven through the headless SimHarness.

These cover what the unit tests can't: the full player-driven pipeline through
the real loop - Talk plan -> approach -> TalkExecutor -> ConversationMenu overlay
-> verb -> disposition/goal outcome - plus two behaviors only observable end to
end: that opening a conversation freezes the world, and that the combat "Accept
Surrender" path pacifies a yielding NPC.

Unit-level scoring/goal logic lives in tests/game/actors/**; this file proves the
pieces are wired together and driven the way a player drives them.
"""

from __future__ import annotations

from brileta import input_events
from brileta.game.actors.ai.behaviors.request_help import RequestHelpGoal
from brileta.game.actors.ai.behaviors.surrender import SurrenderGoal
from brileta.game.actors.indicators import IndicatorKind
from brileta.game.actors.needs import Need, NeedType
from brileta.game.actors.npc_types import BRIGAND_TYPE, RESIDENT_TYPE
from brileta.testing import SimHarness
from brileta.util.pathfinding import probe_step
from tests.helpers import find_tile_near


def _press(convo, key: str) -> None:
    """Send a single verb keypress to the conversation menu."""
    convo.handle_input(input_events.KeyDown(sym=input_events.KeySym(ord(key))))


def _spawn_adjacent(sim: SimHarness, npc_type, name: str):
    """Spawn an NPC on an open tile next to the player and return it."""
    gw = sim.controller.gw
    spot = find_tile_near(
        gw.game_map,
        sim.player_pos,
        lambda x, y: probe_step(gw.game_map, gw, x, y) is None,
        min_radius=1,
    )
    return sim.spawn(npc_type, *spot, name=name)


def _needy_resident(sim: SimHarness):
    """Spawn an adjacent resident that is mid-request toward the player."""
    resident = _spawn_adjacent(sim, RESIDENT_TYPE, "Mara")
    resident.needs.append(Need(type=NeedType.REPAIR, urgency=0.9))
    resident.current_goal = RequestHelpGoal(help_target_id=sim.player.actor_id)
    return resident


def test_accept_request_completes_and_resumes() -> None:
    """Talk to a needy NPC, Accept, and the request resolves end to end.

    Drives the headline Phase 7 flow through the real pipeline: the menu opens on
    the NPC's request, Accept completes the goal, clears the need, raises
    disposition, and un-freezes the world.
    """
    sim = SimHarness(seed="talk-accept")
    resident = _needy_resident(sim)
    before = sim.disposition(resident)

    convo = sim.talk_to(resident)
    assert convo is not None, "conversation never opened"
    assert sim.controller.paused, "conversation should freeze the world while open"
    # The menu opened on the request: Accept is offered.
    assert any(opt.key == "a" for opt in convo.options)

    _press(convo, "a")  # Accept

    assert resident.current_goal is None
    assert resident.needs == []
    assert sim.disposition(resident) > before
    assert not convo.is_active
    assert not sim.controller.paused, "closing the conversation should resume the world"


def test_decline_records_failed_attempt() -> None:
    """Declining records a per-target failed attempt and lowers disposition.

    A spoken "no" must count like a timeout so the NPC stops re-asking - the
    same per-target decay the ignore path uses.
    """
    sim = SimHarness(seed="talk-decline")
    resident = _needy_resident(sim)
    before = sim.disposition(resident)

    convo = sim.talk_to(resident)
    assert convo is not None

    _press(convo, "d")  # Decline

    assert resident.current_goal is None
    assert resident.ai.failed_help_attempts_toward(sim.player) == 1
    assert sim.disposition(resident) < before
    assert not sim.controller.paused


def test_conversation_freezes_the_world() -> None:
    """While a conversation is open the world holds still; closing it resumes.

    Regression for the walk-away bug: the sim ticks NPC turns independently of
    open menus, so without the freeze the NPC being addressed (and everyone else)
    would keep moving mid-conversation.
    """
    sim = SimHarness(seed="talk-freeze")
    resident = _spawn_adjacent(sim, RESIDENT_TYPE, "Sal")

    convo = sim.talk_to(resident)
    assert convo is not None
    assert sim.controller.paused

    # Snapshot every nearby NPC's position, then let real time pass.
    positions = {npc.actor_id: (npc.x, npc.y) for npc in sim.npcs()}
    assert positions, "expected NPCs near the player to observe as frozen"
    sim.tick(60)

    frozen = {npc.actor_id: (npc.x, npc.y) for npc in sim.npcs()}
    assert frozen == positions, "NPCs moved while a conversation was open"

    _press(convo, "l")  # Leave
    assert not convo.is_active
    assert not sim.controller.paused


def test_accept_surrender_pacifies() -> None:
    """The combat 'Accept Surrender' action opens the menu and pacifies the NPC.

    A brigand that has adopted a SurrenderGoal surfaces an Accept-Surrender option
    in combat targeting; taking it opens the surrender conversation, and Accept
    swings disposition non-hostile and clears the goal.
    """
    sim = SimHarness(seed="surrender-accept")
    controller = sim.controller
    brigand = _spawn_adjacent(sim, BRIGAND_TYPE, "Yielder")
    assert brigand.ai.is_hostile_toward(sim.player)
    before = sim.disposition(brigand)

    # Stage the brigand as already surrendering, then interact via the real
    # combat-discovery path. (That surrender *scores* under fire is unit-tested;
    # this proves the player-facing resolution is wired.)
    controller.enter_combat_mode()
    brigand.current_goal = SurrenderGoal()
    brigand.indicator = IndicatorKind.SURRENDER

    options = controller.action_discovery.get_options_for_target(
        controller, sim.player, brigand
    )
    accept = next((o for o in options if o.id == "accept-surrender"), None)
    assert accept is not None, "surrendering NPC did not offer Accept Surrender"
    assert accept.execute is not None
    accept.execute()  # Opens the surrender conversation.

    convo = sim.active_conversation()
    assert convo is not None
    _press(convo, "a")  # Accept

    assert brigand.current_goal is None
    assert not brigand.ai.is_hostile_toward(sim.player)
    assert sim.disposition(brigand) > before


def test_refuse_surrender_records_refusal_and_keeps_hostility() -> None:
    """Refusing a surrender via the combat menu records it and keeps hostility.

    The recorded refusal is what stops the NPC re-adopting Surrender next tick
    (the suppression is unit-tested in tests/game/actors/ai/test_surrender.py).
    Here we prove the player-facing Refuse path records it and leaves the brigand
    hostile and goal-free to fight on.
    """
    sim = SimHarness(seed="surrender-refuse")
    controller = sim.controller
    brigand = _spawn_adjacent(sim, BRIGAND_TYPE, "Defiant")
    assert brigand.ai.is_hostile_toward(sim.player)

    controller.enter_combat_mode()
    brigand.current_goal = SurrenderGoal()
    brigand.indicator = IndicatorKind.SURRENDER

    options = controller.action_discovery.get_options_for_target(
        controller, sim.player, brigand
    )
    accept = next((o for o in options if o.id == "accept-surrender"), None)
    assert accept is not None and accept.execute is not None
    accept.execute()

    convo = sim.active_conversation()
    assert convo is not None
    _press(convo, "r")  # Refuse

    assert brigand.current_goal is None
    assert brigand.ai.was_surrender_refused_by(sim.player)
    assert brigand.ai.is_hostile_toward(sim.player)
