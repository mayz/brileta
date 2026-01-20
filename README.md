# Catley

Catley is an experimental post-apocalyptic game. The goal is to build interesting systems - environmental, social, tactical - and see what emerges from their interactions.

The project prioritizes:
- **Mechanical experimentation** over genre purity
- **Feel and atmosphere** over feature completeness
- **Emergent gameplay** from systemic interactions

## Inspirations

**Tone & Setting:**
- *Fallout 1* and *Fallout 2*
- Late 90s isometric RPGs
- *Disco Elysium*

**Mechanics:**
- Jason Tocci's TTRPG *Wastoid*
- *Brogue*

## Design Principles

**Systemic over scripted:** Prefer mechanics that emerge from system interactions over hand-authored special cases.

**Feel first:** Visual and audio feedback matter as much as mechanical correctness. Uncertain outcomes deserve dramatic presentation.

**Depth through interaction:** Interesting gameplay comes from systems interacting, not from adding more systems in isolation.

**Respect the Intent/Executor pattern:** All world-changing actions go through intents and executors. This architecture enables future extensibility.

## Getting Started

1. Install **Python 3.14**.
2. Run `uv sync` to create a virtual environment and install dependencies from `uv.lock`.
3. Run `make`.
4. Run the game with `make run`.

## Game Action Architecture

Catley uses an **Intent/Executor** pattern for all in-world actions.
Actors and UI components create lightweight `GameIntent` objects that
describe the desired action. These intents are queued through the
`TurnManager`, which dispatches them to specialized executors. Each
executor contains the implementation logic and returns a
`GameActionResult` describing the outcome. Executors should only be
created by `TurnManager` methods, and intents should never call
`execute()` directly.
