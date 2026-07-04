"""NPC social needs: unmet wants that drive help-seeking behavior.

A Need is a small piece of social state an NPC carries: something it wants
resolved (a broken tool repaired, protection from a threat, an item it lacks)
and how badly it wants it. The NUBS 6 RequestHelp behavior reads the most
urgent need to decide whether to approach a capable helper and ask.

Need *generation* (populating this from real broken items, threats, etc.) is a
later roadmap task ("Generated Need Sources"). For now needs are placed by the
dev console for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NeedType(Enum):
    """What kind of help an NPC is after.

    Only the kinds something currently consumes are defined. The design doc
    also lists INFORMATION and HEALTH; those wire in when a behavior uses them.
    """

    REPAIR = "repair"  # A broken possession the NPC wants fixed.
    PROTECTION = "protection"  # The NPC wants an escort/guard against danger.
    ITEM = "item"  # The NPC lacks an item it wants.


@dataclass(slots=True)
class Need:
    """A single unmet want carried by an NPC.

    Attributes:
        type: The category of help wanted.
        urgency: How badly the NPC wants it, in [0, 1]. Feeds RequestHelp
            scoring directly, so a mild need barely competes with routine
            behavior while an urgent one wins.
        target: What the need is about (e.g. the broken item). Loosely typed
            for now; the need-generation task fills it with real objects.
    """

    type: NeedType
    urgency: float
    target: object | None = None

    def describe(self) -> str:
        """Return a short phrase voicing this need, for the Talk hook message."""
        thing = getattr(self.target, "name", None) or self.target
        match self.type:
            case NeedType.REPAIR:
                return f"their {thing} is broken" if thing else "something is broken"
            case NeedType.PROTECTION:
                return "they need protection"
            case NeedType.ITEM:
                return f"they need {thing}" if thing else "they need something"
        return "they need help"
