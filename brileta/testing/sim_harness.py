"""Headless simulation harness for driving the Brileta game without a window.

The engine already runs headless: ``tests.helpers`` boots a full, real-world
``Controller`` backed by a no-op graphics layer (no GPU, no window). This module
wraps that boot in an agent-friendly API so a scenario test (or an AI agent) can
act as the player, tick the world, and observe state - all deterministic from a
seed.

It tests **behavior, not pixels**: graphics, lighting, and sound are stubbed, so
nothing here asserts anything about rendering. What it drives is the semantic
game state - player position, NPC health and disposition, mode transitions, the
captured message log.

Driving layer
-------------
The harness calls the same controller/mode methods the input handler ultimately
routes to (``start_plan`` with ``WalkToPlan``/``MeleeAttackPlan``,
``enter_combat_mode``). It does **not** replay pixel-level ``input_events``
through ``InputHandler.dispatch``; that pixel-to-intent path is a possible future
harness mode for input-layer tests, but everything below the input handler is
exercised as-is here.

The game loop is two controller methods pumped by hand:
``process_player_input()`` (player action/autopilot, once per frame) and
``update_logic_step()`` (NPCs + world at 60Hz). ``_pump`` runs them in lockstep
and clears presentation timing each step so autopilot and NPC reactions - which
in the real app are gated on wall-clock ``duration_ms`` pacing - advance one
action per pumped step instead of stalling on real time.

Extending it
------------
- **Add a verb**: wrap the controller/mode method the UI would call. A targeted
  action is usually ``self.controller.start_plan(player, SomePlan, ...)`` (see
  ``walk_to``/``attack``); a mode toggle is a direct controller call. Keep verbs
  on this semantic layer - don't reach for ``input_events``. Grow verbs per
  scenario; don't speculatively add them.
- **Add a scenario**: construct ``SimHarness(seed=...)``, ``spawn`` any actors
  the scenario needs, drive verbs, ``tick``/``wait`` to advance, then assert on
  ``player_pos``, ``npcs()``, ``disposition()``, an NPC's ``health.hp``, or
  ``messages``. See ``tests/test_sim_harness_*.py`` for worked examples.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from brileta import config
from brileta.events import MessageEvent, reset_event_bus_for_testing, subscribe_to_event
from brileta.game.action_plan import WalkToPlan
from brileta.game.actions.combat import MeleeAttackPlan
from brileta.game.actors import NPC, Actor, Character
from brileta.game.actors.npc_core import NPCType
from brileta.input_handler import InputHandler
from brileta.types import RandomSeed, WorldTileCoord, WorldTilePos
from brileta.util import rng

# Default Chebyshev radius for "near the player" queries. FOV_RADIUS is the
# player's sight range, so it's the natural scope for "NPCs the player can see".
_NEARBY_RADIUS = config.FOV_RADIUS


class SimHarness:
    """Drives a real, headless game world as the player and observes its state.

    Construct with a seed to boot a fresh, fully deterministic world (same seed
    reproduces the same world, NPC placement, and dice rolls). Then drive verbs
    (:meth:`walk_to`, :meth:`attack`, ...) and advance the simulation with
    :meth:`tick` / :meth:`wait`.
    """

    def __init__(self, seed: RandomSeed = None) -> None:
        """Boot a headless world for ``seed``.

        Seeding wires all three determinism sources the engine reads: the
        ``config.RANDOM_SEED`` that ``Controller.new_world`` consumes, the global
        ``random`` state (reseeded inside ``new_world``), and the isolated
        per-domain RNG streams (``rng.init``). A given seed therefore reproduces
        the world exactly.
        """
        # Import here so importing this module doesn't pull in test-only helpers
        # (and their heavy graphics stubs) unless a harness is actually built.
        from tests.helpers import (
            get_controller_with_player_and_map,
            reset_actor_id_counter,
        )

        # Reseed every determinism source before the world is built.
        config.RANDOM_SEED = seed
        rng.init(seed)
        # Fresh actor-id numbering so ids are reproducible across harnesses.
        reset_actor_id_counter()

        # Fresh event bus so a prior harness's message listener (or a stale
        # controller subscription) can't leak into this world. Reset BEFORE the
        # controller is built so the controller's own subscriptions (e.g. combat
        # auto-entry) register on this fresh bus.
        reset_event_bus_for_testing()

        self.controller = get_controller_with_player_and_map()

        # The dummy builder leaves input_handler as None, but
        # process_player_input() asserts it is set. The method only uses
        # overlay_system/turn_manager, so any non-None stub satisfies the guard.
        self.controller.input_handler = cast(InputHandler, SimpleNamespace())

        # Capture the in-game message log by subscribing to MessageEvent.
        self.messages: list[str] = []
        subscribe_to_event(MessageEvent, lambda e: self.messages.append(e.text))

    # -- Observation ---------------------------------------------------------

    @property
    def player(self) -> Character:
        """The player character."""
        return self.controller.gw.player

    @property
    def player_pos(self) -> WorldTilePos:
        """The player's current tile position."""
        return (self.player.x, self.player.y)

    def npcs(self, radius: int = _NEARBY_RADIUS) -> list[NPC]:
        """Return living NPCs within ``radius`` tiles of the player.

        Filters to real NPCs with an AI component - trees and other actors carry
        an ``ai`` attribute that is ``None`` and must be excluded.
        """
        px, py = self.player_pos
        nearby = self.controller.gw.actor_spatial_index.get_in_radius(px, py, radius)
        return [
            actor
            for actor in nearby
            if isinstance(actor, NPC)
            and actor.ai is not None
            and actor.health.is_alive()
        ]

    def disposition(self, npc: NPC) -> int:
        """Return ``npc``'s numeric disposition toward the player (-100..+100)."""
        return npc.ai.disposition_toward(self.player)

    # -- Scenario setup ------------------------------------------------------

    def spawn(
        self,
        npc_type: NPCType,
        x: WorldTileCoord,
        y: WorldTileCoord,
        name: str | None = None,
    ) -> NPC:
        """Create an NPC of ``npc_type`` at ``(x, y)`` and add it to the world."""
        npc = npc_type.create(
            x, y, name or npc_type.display_name, game_world=self.controller.gw
        )
        self.controller.gw.add_actor(npc)
        return npc

    # -- Verbs (driving layer) ----------------------------------------------

    def walk_to(self, x: WorldTileCoord, y: WorldTileCoord) -> None:
        """Start the player walking to ``(x, y)``.

        Sets a ``WalkToPlan`` (the same plan a right-click-to-move issues); the
        player advances one step per pumped tick. Call :meth:`wait`/:meth:`tick`
        to actually execute the plan.
        """
        self.controller.start_plan(self.player, WalkToPlan, target_position=(x, y))

    def attack(self, npc: Actor) -> None:
        """Enter combat mode and have the player melee-attack ``npc``.

        Drives the same pipeline as the combat-mode default action: enter combat,
        then start a ``MeleeAttackPlan`` that approaches to adjacency and attacks.
        Call :meth:`wait`/:meth:`tick` to run the approach and the strike.
        """
        self.controller.enter_combat_mode()
        self.controller.start_plan(
            self.player,
            MeleeAttackPlan,
            target_actor=npc,
            target_position=(npc.x, npc.y),
        )

    # -- Loop pump -----------------------------------------------------------

    def tick(self, steps: int = 1) -> None:
        """Advance the simulation by ``steps`` logic steps (player + world)."""
        self._pump(steps)

    def wait(self, turns: int = 1) -> None:
        """Alias for :meth:`tick` - advance ``turns`` logic steps.

        Named for readability at call sites where the player is idling or letting
        a queued plan run to completion.
        """
        self._pump(turns)

    def _pump(self, steps: int) -> None:
        """Run ``steps`` iterations of (player input, one logic step).

        Presentation timing is cleared before each step so the wall-clock pacing
        gate (``duration_ms``) never stalls autopilot or NPC reactions - each
        pumped step commits at most one player action and advances NPCs by their
        energy, deterministically and without sleeping.
        """
        tm = self.controller.turn_manager
        for _ in range(steps):
            tm.clear_presentation_timing()
            self.controller.process_player_input()
            self.controller.update_logic_step()
