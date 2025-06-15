# Catley

A post-apocalyptic roguelike.

## Getting Started

1. Install **Python 3.13**.
2. Run `uv sync` to create a virtual environment and install dependencies from `uv.lock`.
3. Install the project in editable mode:
   ```sh
   uv pip install -e .
   ```
4. Execute the test suite:
   ```sh
  pytest -q
   ```

## Game Action Architecture

Catley uses an **Intent/Executor** pattern for all in-world actions.
Actors and UI components create lightweight `GameIntent` objects that
describe the desired action. These intents are queued through the
`TurnManager`, which dispatches them to specialized executors. Each
executor contains the implementation logic and returns a
`GameActionResult` describing the outcome. Executors should only be
created by `TurnManager` methods, and intents should never call
`execute()` directly.
