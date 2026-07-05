"""Persistent presence indicators shown hovering over NPCs.

An indicator is a single-glyph speech bubble that persists while an NPC is in a
notable state. Unlike floating text, which rises and fades, an indicator is
polled from NPC state each frame (see IndicatorRenderer) and drawn only while
that state holds. The AI sets NPC.indicator each tick; the renderer maps the
kind to a glyph and color via INDICATOR_STYLES.

The vocabulary is deliberately small so a bubble always means "this NPC is
exceptional right now." ATTACK and FLEE are wired today. REQUEST and INVESTIGATE
are defined but not yet raised - they wire in when NUBS Phase 6 (RequestHelpGoal)
and NUBS Phase 8 (InvestigateNoiseGoal) land, at which point those goals just set
NPC.indicator to the matching kind.
"""

from __future__ import annotations

from enum import Enum

from brileta import colors


class IndicatorKind(Enum):
    """A notable NPC state worth flagging with a hovering bubble."""

    ATTACK = "attack"  # Live: hostile and attacking.
    FLEE = "flee"  # Live: fleeing a threat.
    REQUEST = "request"  # Deferred to NUBS 6: NPC wants the player's help.
    INVESTIGATE = "investigate"  # Deferred to NUBS 8: NPC is checking a noise.
    SURRENDER = "surrender"  # Live (NUBS 7): NPC has yielded and is cowering.


# Glyph and color drawn for each kind. "!" reads as urgent/notable; "?" reads as
# uncertain/searching. Colors are darkened/saturated so they stay legible on the
# parchment bubble (see floating_text.BUBBLE_FILL) - bright red/yellow wash out
# on the cream fill.
INDICATOR_STYLES: dict[IndicatorKind, tuple[str, colors.Color]] = {
    IndicatorKind.ATTACK: ("!", (200, 30, 30)),  # deep red
    IndicatorKind.FLEE: ("!", (190, 120, 20)),  # amber
    IndicatorKind.REQUEST: ("!", (40, 110, 120)),  # teal
    IndicatorKind.INVESTIGATE: ("?", (150, 110, 20)),  # olive
    IndicatorKind.SURRENDER: ("⚑", (210, 210, 210)),  # white flag
}
