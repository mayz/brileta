## Overview

This is an in-progress post-apocalyptic roguelike CRPG inspired by Fallout 1 & 2, Jason Tocci's TTRPG Wastoid, and the roguelike classic Brogue.

## Dev Environment

My local Python version is 3.13.

The Python code is set up as a Python package that was set up with `uv`.

To create a virtual environment and install all dependencies, run `uv sync`. Everything is in the local `uv.lock` file.

I've been using `ruff check` as a linter, and Pylance/Pyright as a language server in VSCode and type checker. I've also been experimenting with the new `ty` type checker from Astral.

To run unit tests:
`uv pip install -e .`        # Install catley as an editable package
`pytest`                  # Run the tests


## Style

- I've been using `ruff format` to format code. Wherever possible, I try to annotate params and variable with their types, as you can see throughout the code.
- All new code must be fully type-hinted and pass static analysis.
- Good Documentation: Comments should describe what code is doing and (if particularly complex) why it is doing it.

##  Dev Notes

- Identical Outcome: Ideally, there should be no performance regressions and no visual regressions (unless deliberately making visual changes).
