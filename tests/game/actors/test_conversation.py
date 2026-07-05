"""Tests for NUBS 7 conversation stance derivation and the Offer data model."""

from __future__ import annotations

from brileta.game.actors.conversation import Stance, derive_stance
from brileta.game.actors.needs import Need, NeedType
from brileta.game.actors.offers import Offer, OfferType
from tests.helpers import make_ai_world


def _npc_and_player(disposition: int):
    """A player and an NPC with the given disposition toward that player."""
    _, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=disposition)
    return npc, player


def test_stance_hostile_below_threshold() -> None:
    npc, player = _npc_and_player(-75)
    assert derive_stance(npc, player) is Stance.HOSTILE


def test_stance_guarded_when_mildly_disliked() -> None:
    npc, player = _npc_and_player(-30)
    assert derive_stance(npc, player) is Stance.GUARDED


def test_stance_desperate_when_urgent_need() -> None:
    npc, player = _npc_and_player(0)
    npc.needs.append(Need(type=NeedType.REPAIR, urgency=0.9))
    assert derive_stance(npc, player) is Stance.DESPERATE


def test_stance_transactional_when_neutral() -> None:
    npc, player = _npc_and_player(5)
    assert derive_stance(npc, player) is Stance.TRANSACTIONAL


def test_stance_open_when_liked_with_offers() -> None:
    npc, player = _npc_and_player(50)
    npc.offers.append(Offer(type=OfferType.SERVICE, content="repair"))
    assert derive_stance(npc, player) is Stance.OPEN


def test_liked_without_offers_is_not_open() -> None:
    # High disposition but nothing to offer -> transactional, not open.
    npc, player = _npc_and_player(50)
    assert derive_stance(npc, player) is Stance.TRANSACTIONAL


# ---------------------------------------------------------------------------
# Offer data model
# ---------------------------------------------------------------------------


def test_offer_describe_service_with_price() -> None:
    offer = Offer(type=OfferType.SERVICE, content="repair", price=8)
    assert offer.describe() == "repair service (8 coin)"


def test_offer_describe_item_free() -> None:
    offer = Offer(type=OfferType.ITEM, content="a canteen")
    assert offer.describe() == "a canteen"
