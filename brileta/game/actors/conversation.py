"""Conversation stance: how an NPC is disposed to talk, derived from utility state.

A Stance is the NPC's conversational posture at the moment a conversation opens.
It is not stored - it is derived each time from disposition plus what the NPC
currently wants (needs) and can give (offers). The Phase 7 ConversationMenu reads
the stance to label the header and decide which verbs to show.

This lives in the game layer (not the UI) so the derivation logic stays testable
without a rendering backend.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brileta.game.actors.core import NPC, Actor

# Disposition thresholds for stance bands. These mirror the design doc's
# derive_stance and are deliberately coarse - stance is a UI posture, not a
# behavioral gate (that is what utility scoring is for).
_HOSTILE_BELOW = -50
_GUARDED_BELOW = -10
_OPEN_ABOVE = 30


class Stance(Enum):
    """An NPC's conversational posture toward the player."""

    HOSTILE = "Hostile"  # Wants the player gone; talk is barely possible.
    GUARDED = "Guarded"  # Suspicious, gives little.
    DESPERATE = "Desperate"  # Has an urgent need and leads with it.
    OPEN = "Open"  # Friendly and forthcoming, will trade/share.
    TRANSACTIONAL = "Transactional"  # Neutral, all business.


def has_urgent_need(npc: NPC) -> bool:
    """Whether the NPC carries a need urgent enough to lead a conversation with."""
    return any(need.urgency >= 0.5 for need in npc.needs)


def derive_stance(npc: NPC, player: Actor) -> Stance:
    """Derive the NPC's conversational stance toward the player.

    Follows the design doc's ordering: hostility and guardedness first (they
    override everything), then an urgent need (the NPC leads with its ask),
    then openness for well-liked NPCs with something to offer, else a neutral
    transactional stance.
    """
    disposition = npc.ai.disposition_toward(player)

    if disposition < _HOSTILE_BELOW:
        return Stance.HOSTILE
    if disposition < _GUARDED_BELOW:
        return Stance.GUARDED
    if has_urgent_need(npc):
        return Stance.DESPERATE
    if disposition > _OPEN_ABOVE and npc.offers:
        return Stance.OPEN
    return Stance.TRANSACTIONAL
