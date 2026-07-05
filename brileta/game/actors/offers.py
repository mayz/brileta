"""NPC offers: things an NPC can provide through conversation.

An Offer is the mirror of a Need (see needs.py): where a Need is something the
NPC wants from someone else, an Offer is something the NPC can give - a repair
service, a spare item, a piece of information, access to a locked place. The
Phase 7 conversation UI surfaces offers under the "Request" and "Trade" verbs so
the player can ask for them.

Offer *generation* (populating this from settlement roles and situations) is a
later roadmap task (Shopkeeper NPCs / Settlement Population). For now offers are
placed by the dev console for testing, exactly like needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OfferType(Enum):
    """What kind of thing an NPC can provide."""

    SERVICE = "service"  # A service the NPC performs (repair, healing, escort).
    ITEM = "item"  # A physical item the NPC will part with.
    INFORMATION = "information"  # Something the NPC knows and can tell.
    ACCESS = "access"  # Access to a place or thing (a locked door, a room).


@dataclass(slots=True)
class Offer:
    """A single thing an NPC can provide, surfaced in conversation.

    Attributes:
        type: The category of thing offered.
        content: What is offered (a service name, an item, a fact). Loosely
            typed for now; later economy tasks fill it with real objects.
        price: Coin the NPC asks in return, or 0 for free. A plain integer
            placeholder until the Coin & Payment task wires a real economy.
    """

    type: OfferType
    content: object | None = None
    price: int = 0
    conditions: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Return a short label naming this offer, for the conversation menu."""
        thing = getattr(self.content, "name", None) or self.content
        match self.type:
            case OfferType.SERVICE:
                label = f"{thing} service" if thing else "a service"
            case OfferType.ITEM:
                label = f"{thing}" if thing else "an item"
            case OfferType.INFORMATION:
                label = f"what they know about {thing}" if thing else "information"
            case OfferType.ACCESS:
                label = f"access to {thing}" if thing else "access"
            case _:
                label = "something"
        if self.price > 0:
            label += f" ({self.price} coin)"
        return label
