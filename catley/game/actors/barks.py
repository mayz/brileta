"""NPC bump barks: short phrases for speech bubbles when bumped by the player."""

from __future__ import annotations

from catley.game.actors.conditions import Injury
from catley.game.enums import Disposition
from catley.util import rng

from .core import NPC

_rng = rng.get("npc.barks")

BARKS_BY_DISPOSITION: dict[Disposition, tuple[str, ...]] = {
    Disposition.HOSTILE: (
        "Back off.",
        "Move.",
        "Outta my way.",
        "Touch me and die.",
    ),
    Disposition.UNFRIENDLY: (
        "Don't touch me.",
        "Watch it.",
        "Keep your distance.",
        "You lost?",
    ),
    Disposition.WARY: (
        "Easy.",
        "Careful.",
        "Whoa.",
        "Mind yourself.",
    ),
    Disposition.APPROACHABLE: (
        "Hey.",
        "Oh—sorry.",
        "Need something?",
        "Yeah?",
    ),
    Disposition.FRIENDLY: (
        "Hi there.",
        "You okay?",
        "All good.",
        "Hey, friend.",
    ),
    Disposition.ALLY: (
        "We're good.",
        "You alright?",
        "Got your back.",
        "All set.",
    ),
}

BARKS_BY_STATE: dict[str, tuple[str, ...]] = {
    "wounded": (
        "I'm hurt...",
        "Easy—I'm wounded.",
        "Not now.",
        "Careful—I'm bleeding.",
    ),
    "exhausted": (
        "Need a breather.",
        "I'm beat.",
        "Give me a sec.",
        "Can't keep this up.",
    ),
    "injured": (
        "That stings.",
        "Mind the limp.",
        "Not my best day.",
        "Ow—watch it.",
    ),
}


def _get_bark_state(npc: NPC) -> str | None:
    health = npc.health
    if health.hp <= max(1, health.max_hp // 3):
        return "wounded"

    if npc.modifiers.get_exhaustion_count() >= 2:
        return "exhausted"

    if npc.conditions.get_conditions_by_type(Injury):
        return "injured"

    return None


def pick_bump_bark(npc: NPC) -> str | None:
    """Pick a short bark when the player bumps this NPC."""
    state = _get_bark_state(npc)
    if state is not None:
        candidates = BARKS_BY_STATE.get(state)
        if candidates:
            return _rng.choice(candidates)

    disposition = npc.ai.disposition
    candidates = BARKS_BY_DISPOSITION.get(disposition)
    if not candidates:
        return None

    return _rng.choice(candidates)
