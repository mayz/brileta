"""ConversationMenu - the Phase 7 conversation interface.

Conversation is a UI layer on top of the utility system, not a separate system.
The menu presents the NPC's current *stance* (derived from disposition and what
they want/offer) and offers social *verbs* rather than pre-written dialogue.

Two special openings are auto-detected from the NPC's active goal:
  - A RequestHelpGoal opens the conversation on the NPC's ask (Accept/Decline).
  - A SurrenderGoal opens the surrender resolution (Accept/Refuse/Ignore).

Otherwise the menu opens on a neutral greeting with the general verbs (Greet,
Request, Trade, Leave). Verb outcomes feed back into disposition and goal state,
which is the whole point: the conversation moves the same numbers the utility
brain reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors, input_events
from brileta.events import MessageEvent, publish_event
from brileta.game.actors.ai.behaviors.request_help import RequestHelpGoal
from brileta.game.actors.ai.behaviors.surrender import SurrenderGoal
from brileta.game.actors.barks import emit_bark
from brileta.game.actors.conversation import Stance, derive_stance
from brileta.game.actors.offers import OfferType
from brileta.view.ui.overlays import Menu, MenuOption

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actors.core import NPC, Actor
    from brileta.game.actors.offers import Offer

# Verb actions return this to keep the menu open (re-render the current page);
# returning None/True closes it. Mirrors the base Menu's mouse convention where a
# False result leaves the overlay up.
_STAY = False

# Stance -> the NPC's opening line. Kept as flat templates (no narrative
# generation); Phase 9 (Knowledge & Topics) adds real content behind "Ask About".
_STANCE_GREETING: dict[Stance, str] = {
    Stance.HOSTILE: "Get away from me.",
    Stance.GUARDED: "What do you want?",
    Stance.DESPERATE: "Please - I could really use a hand.",
    Stance.OPEN: "Good to see you. What can I do for you?",
    Stance.TRANSACTIONAL: "What do you need?",
}


def open_conversation(controller: Controller, npc: NPC) -> None:
    """Build and show a conversation with ``npc``, auto-picking the opening.

    Shared entry point for the Talk executor (player-initiated) and the combat
    "Accept Surrender" action, so both route through the same menu.
    """
    assert controller.overlay_system is not None
    menu = ConversationMenu(controller, npc)
    controller.overlay_system.show_menu(menu)


class ConversationMenu(Menu):
    """A stance-driven conversation with a single NPC."""

    def __init__(self, controller: Controller, npc: NPC) -> None:
        self.npc = npc
        self.player: Actor = controller.gw.player
        self.stance = derive_stance(npc, self.player)
        # Sub-page state: "root" is the verb list, "offers" lists requestable
        # offers of a given kind (goods for Trade, favors for Request).
        self._page = "root"
        self._offer_kinds: tuple[OfferType, ...] = ()
        # Whether we paused the sim on open, so hide() only unpauses what we
        # paused (a conversation opened while the player had already hit space
        # must stay paused afterward).
        self._paused_by_us = False
        title = f"{npc.name} - {self.stance.value}"
        super().__init__(title=title, controller=controller, width=44)
        # This overlay pauses the sim itself (see show/hide), so it is its own
        # "world frozen" signal - the PAUSED banner stays hidden while it is up.
        self.freezes_sim = True

    def show(self) -> None:
        """Open the conversation and freeze the world while it is up.

        Talking is a face-to-face moment: the sim keeps ticking NPC turns
        independently of overlays, so without this the NPC you addressed would
        keep walking its routine off-screen. Pausing holds everyone in place
        until the conversation closes.
        """
        super().show()
        if not self.controller.paused:
            self.controller.paused = True
            self._paused_by_us = True
        # Stop the NPC mid-route so it is not frozen a step into walking away and
        # does not immediately resume that walk the instant the menu closes.
        self.controller.stop_plan(self.npc)

    def hide(self) -> None:
        """Close the conversation and resume the world if we paused it."""
        super().hide()
        if self._paused_by_us:
            self.controller.paused = False
            self._paused_by_us = False

    # -- page building --------------------------------------------------------

    def populate_options(self) -> None:
        """Dispatch option building by the active page and the NPC's goal."""
        goal = self.npc.current_goal
        if isinstance(goal, SurrenderGoal) and not goal.is_complete:
            self._populate_surrender(goal)
        elif self._page == "offers":
            self._populate_offers()
        elif isinstance(goal, RequestHelpGoal) and not goal.is_complete:
            self._populate_request(goal)
        else:
            self._populate_root()

    def _add_speech(self, line: str) -> None:
        """Add the NPC's spoken line as a non-selectable header row."""
        self.add_option(MenuOption(key=None, text=f'"{line}"', enabled=False))

    def _populate_root(self) -> None:
        """The general verb list, gated by stance and what the NPC offers."""
        self._add_speech(_STANCE_GREETING[self.stance])

        self.add_option(MenuOption(key="g", text="Greet", action=self._greet))

        # Request covers favors/services/info/access; Trade covers goods. Both
        # are only shown when the NPC actually has offers of that kind, so the
        # verb list never dangles a dead end.
        favor_kinds = (OfferType.SERVICE, OfferType.INFORMATION, OfferType.ACCESS)
        if any(o.type in favor_kinds for o in self.npc.offers):
            self.add_option(
                MenuOption(
                    key="r",
                    text="Request...",
                    action=lambda: self._open_offers(favor_kinds),
                )
            )
        if any(o.type is OfferType.ITEM for o in self.npc.offers):
            self.add_option(
                MenuOption(
                    key="t",
                    text="Trade...",
                    action=lambda: self._open_offers((OfferType.ITEM,)),
                )
            )

        self.add_option(MenuOption(key="l", text="Leave", action=self._leave))

    def _populate_request(self, goal: RequestHelpGoal) -> None:
        """The NPC leads with its help request: Accept / Decline / Leave."""
        need = self._most_urgent_need()
        ask = need.describe() if need is not None else "they need a hand"
        self._add_speech(f"Hey - {ask}. Can you help?")
        self.add_option(
            MenuOption(
                key="a",
                text="Accept - I'll help.",
                color=colors.GREEN,
                action=self._accept_request,
            )
        )
        self.add_option(
            MenuOption(
                key="d",
                text="Decline - Not now.",
                action=self._decline_request,
            )
        )
        self.add_option(MenuOption(key="l", text="Leave", action=self._leave))

    def _populate_surrender(self, goal: SurrenderGoal) -> None:
        """A surrendering NPC: Accept / Refuse / Ignore."""
        self._add_speech("I yield! Don't hurt me - please!")
        self.add_option(
            MenuOption(
                key="a",
                text="Accept - Let them go.",
                color=colors.GREEN,
                action=self._accept_surrender,
            )
        )
        self.add_option(
            MenuOption(
                key="r",
                text="Refuse - No mercy.",
                color=colors.RED,
                action=self._refuse_surrender,
            )
        )
        self.add_option(
            MenuOption(key="i", text="Ignore - Walk away.", action=self._leave)
        )

    def _populate_offers(self) -> None:
        """List the NPC's offers of the selected kinds, plus a Back row."""
        offers = [o for o in self.npc.offers if o.type in self._offer_kinds]
        self._add_speech("Here's what I've got.")
        # Number the offers 1..9 so their keys never collide with the "b"ack row.
        for index, offer in enumerate(offers[:9]):
            self.add_option(
                MenuOption(
                    key=str(index + 1),
                    text=offer.describe().capitalize(),
                    action=lambda o=offer: self._request_offer(o),
                )
            )
        self.add_option(MenuOption(key="b", text="Back", action=self._back))

    # -- verb outcomes --------------------------------------------------------

    def _greet(self) -> bool:
        """Voice the NPC's stance-appropriate line, stay in the conversation."""
        emit_bark(self.npc, _STANCE_GREETING[self.stance])
        return _STAY

    def _open_offers(self, kinds: tuple[OfferType, ...]) -> bool:
        """Drill into the offer list for the given kinds."""
        self._page = "offers"
        self._offer_kinds = kinds
        return _STAY

    def _back(self) -> bool:
        """Return from the offer list to the verb list."""
        self._page = "root"
        return _STAY

    def _request_offer(self, offer: Offer) -> bool:
        """Ask for one offer. Voices it and nudges disposition; stays open.

        Actually fulfilling the offer (handing over goods, performing a service,
        taking payment) waits on the Coin & Payment and Shopkeeper tasks; for now
        requesting is a social move that acknowledges the offer and warms the NPC
        slightly.
        """
        publish_event(
            MessageEvent(
                f"You ask {self.npc.name} for {offer.describe()}.", colors.CYAN
            )
        )
        self.npc.ai.modify_disposition(self.player, +2)
        return _STAY

    def _accept_request(self) -> None:
        """Agree to help: complete the goal, warm the NPC, clear the need."""
        goal = self.npc.current_goal
        if isinstance(goal, RequestHelpGoal):
            goal.fulfill(self.npc)
            self.npc.current_goal = None
        need = self._most_urgent_need()
        if need is not None:
            # Resolving the need stands in for real fulfillment (Repair /
            # Generated Needs land later); without it the NPC re-adopts the
            # request and re-approaches next tick.
            self.npc.needs.remove(need)
        self.npc.ai.modify_disposition(self.player, +10)
        publish_event(
            MessageEvent(
                f"You agree to help {self.npc.name}. They're grateful.", colors.GREEN
            )
        )

    def _decline_request(self) -> None:
        """Turn the NPC down: record the miss so they stop re-asking."""
        goal = self.npc.current_goal
        if isinstance(goal, RequestHelpGoal):
            goal.decline(self.npc, self.player)
            self.npc.current_goal = None
        self.npc.ai.modify_disposition(self.player, -5)
        publish_event(MessageEvent(f"You turn {self.npc.name} down.", colors.GREY))

    def _accept_surrender(self) -> None:
        """Accept the surrender: the NPC disengages and is pacified."""
        goal = self.npc.current_goal
        if isinstance(goal, SurrenderGoal):
            goal.accept(self.npc, self.player)
            self.npc.current_goal = None
        publish_event(
            MessageEvent(
                f"{self.npc.name} lowers their weapon and backs away.", colors.GREEN
            )
        )

    def _refuse_surrender(self) -> None:
        """Refuse the surrender: the NPC drops back into a desperate fight."""
        goal = self.npc.current_goal
        if isinstance(goal, SurrenderGoal):
            goal.refuse(self.npc, self.player)
            self.npc.current_goal = None
        publish_event(
            MessageEvent(f"{self.npc.name} snarls and fights on!", colors.RED)
        )

    def _leave(self) -> None:
        """End the conversation with no state change."""
        return

    # -- helpers --------------------------------------------------------------

    def _most_urgent_need(self):
        """Return the NPC's most urgent need, or None."""
        if not self.npc.needs:
            return None
        return max(self.npc.needs, key=lambda n: n.urgency)

    def _rebuild(self) -> None:
        """Re-render the current page after a stay-open verb."""
        self.options.clear()
        self.hovered_option_index = None
        self.populate_options()
        self.invalidate()

    # -- input ----------------------------------------------------------------

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Delegate to the base Menu, keeping a conversation from being dismissed
        by SPACE or RETURN.

        The base Menu treats SPACE and RETURN as "close"; in a conversation those
        should do nothing, so a stray keypress can't end the exchange. Everything
        else - verb keys, ESC, clicks outside, hover - is the base behavior, with
        per-verb stay-open handled by the overridden :meth:`_activate_option`.
        Delegating (rather than reimplementing) means base-menu input fixes flow
        through to conversations too.
        """
        if isinstance(event, input_events.KeyDown) and event.sym in (
            input_events.KeySym.SPACE,
            input_events.KeySym.RETURN,
        ):
            return True  # Swallow: a conversation is not dismissed by these.
        return super().handle_input(event)

    def _activate_option(self, option: MenuOption) -> None:
        """Run a verb; on a stay-open verb re-render the page instead of closing.

        Overrides the base activation seam: a verb returning ``_STAY`` keeps the
        conversation open and rebuilds the current page (e.g. drilling into the
        offer list); any other result closes it.
        """
        if not option.enabled or option.action is None:
            return
        result = option.action()
        if result is _STAY:
            self._rebuild()
        else:
            self.hide()
