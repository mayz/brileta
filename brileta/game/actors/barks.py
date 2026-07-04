"""NPC barks: short phrases shown in speech bubbles for moment-to-moment reactions.

Barks cover the player bumping an NPC (pick_bump_bark) and an NPC being shoved
(pick_shove_bark). emit_bark() is the shared, throttled entry point that any
system can call to raise a bubble without spamming the same NPC.
"""

from __future__ import annotations

import time

from brileta import colors
from brileta.events import (
    FloatingTextEvent,
    FloatingTextSize,
    FloatingTextValence,
    publish_event,
)
from brileta.game.actors.ai import disposition_label
from brileta.game.actors.conditions import Injury
from brileta.util import rng

from .core import NPC, Character

_rng = rng.get("npc.barks")

# A bark bubble lingers this long, and a given NPC stays quiet for a short beat
# after so back-to-back triggers (repeated bumps/shoves) don't stack bubbles.
BARK_DURATION = 1.1
BARK_COOLDOWN_SECONDS = 0.25

# Bark dialogue is drawn as dark ink on the parchment bubble (see
# floating_text.BUBBLE_FILL); a light color would wash out on the cream.
BARK_INK: colors.Color = (56, 42, 28)

BARKS_BY_DISPOSITION: dict[str, tuple[str, ...]] = {
    "Hostile": (
        "Back off.",
        "Move.",
        "Outta my way.",
        "Touch me and die.",
    ),
    "Unfriendly": (
        "Don't touch me.",
        "Watch it.",
        "Keep your distance.",
        "You lost?",
    ),
    "Wary": (
        "Easy.",
        "Careful.",
        "Whoa.",
        "Mind yourself.",
    ),
    "Approachable": (
        "Hey.",
        "Oh - sorry.",
        "Need something?",
        "Yeah?",
    ),
    "Friendly": (
        "Hi there.",
        "You okay?",
        "All good.",
        "Hey, friend.",
    ),
    "Ally": (
        "We're good.",
        "You alright?",
        "Got your back.",
        "All set.",
    ),
}

# Reactions to being shoved. Punchier than a bump: the NPC was physically
# displaced, so these read as annoyed/rattled regardless of disposition. A
# wounded/injured NPC still prefers its state bark (see pick_shove_bark).
SHOVE_BARKS: tuple[str, ...] = (
    "Hey!",
    "Get off!",
    "Hands off!",
    "Watch it!",
    "Don't shove me!",
    "Back off!",
)

BARKS_BY_STATE: dict[str, tuple[str, ...]] = {
    "wounded": (
        "I'm hurt...",
        "Easy - I'm wounded.",
        "Not now.",
        "Careful - I'm bleeding.",
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
        "Ow - watch it.",
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


def pick_bump_bark(npc: NPC, player: Character) -> str | None:
    """Pick a short bark when the player bumps this NPC."""
    state = _get_bark_state(npc)
    if state is not None:
        candidates = BARKS_BY_STATE.get(state)
        if candidates:
            return _rng.choice(candidates)

    label = disposition_label(npc.ai.disposition_toward(player))
    candidates = BARKS_BY_DISPOSITION.get(label)
    if not candidates:
        return None

    return _rng.choice(candidates)


def pick_shove_bark(npc: NPC) -> str | None:
    """Pick a short bark when this NPC is shoved (Push stunt)."""
    state = _get_bark_state(npc)
    if state is not None:
        candidates = BARKS_BY_STATE.get(state)
        if candidates:
            return _rng.choice(candidates)

    return _rng.choice(SHOVE_BARKS)


def emit_bark(npc: NPC, text: str) -> bool:
    """Raise a throttled speech-bubble bark above an NPC.

    Returns False (and does nothing) if the NPC barked too recently, so callers
    can fire on every trigger without stacking bubbles. The throttle window is
    the bubble's own lifetime plus a short cooldown.
    """
    now = time.perf_counter()
    if now < npc.bark_block_until:
        return False

    npc.bark_block_until = now + BARK_DURATION + BARK_COOLDOWN_SECONDS
    publish_event(
        FloatingTextEvent(
            text=text,
            target_actor_id=npc.actor_id,
            valence=FloatingTextValence.NEUTRAL,
            size=FloatingTextSize.NORMAL,
            duration=BARK_DURATION,
            color=BARK_INK,
            world_x=npc.x,
            world_y=npc.y,
            bubble=True,
        )
    )
    return True
