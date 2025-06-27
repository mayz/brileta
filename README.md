# Catley

A post-apocalyptic roguelike.

## Getting Started

1. Install **Python 3.13**.
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
