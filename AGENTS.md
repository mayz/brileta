For design discussions, read README.md for project context and design principles.

## Dev Environment

My local Python version is 3.14.

The Python code is set up as a Python package that was set up with `uv`.

To create a virtual environment and install all dependencies, run `uv sync`. Everything is in the local `uv.lock` file.

I've been using `uv run ruff check` as a linter, and Pylance/Pyright as a language server in VSCode and type checker.

To run unit tests:
`make test`

This command, defined in the `Makefile`, will automatically sync dependencies and run the test suite within the correct environment.

## Headless Linux / CI (Ubuntu)

When running in a headless Linux container, ModernGL and WGPU tests require an X11 display. Install these system packages and run tests under Xvfb, rather than running `make` directly:

```bash
sudo apt-get update
sudo apt-get install -y \
  libgl1-mesa-dri libgl1 libglx-mesa0 libegl1 \
  libx11-6 libxext6 libxrandr2 libxi6 libxxf86vm1 \
  libvulkan1 mesa-vulkan-drivers vulkan-tools \
  xvfb xauth

xvfb-run -a make
```

  **CRITICAL**: This project uses `uv` for dependency management. Always run Python commands with `uv run`:
  - Use `uv run python script.py` NOT `python script.py`
  - Use `uv run pyright` NOT `pyright`
  - Use `uv run pytest` NOT `pytest`

  The only exception is the Makefile targets which already handle this automatically.

## Quality Checks

**IMPORTANT: After finishing any code changes, always run:**
```bash
make
```

This runs the full quality pipeline: `uv run ruff format`, `uv run ruff check`, `uv run pyright`, and `uv run pytest`. All checks must pass before considering work complete.

Individual commands are available as `make lint`, `make format`, `make typecheck`, or `make test`.

## Style

- I've been using `uv run ruff format` to format code. Wherever possible, I try to annotate params and variable with their types, as you can see throughout the code.
- All new code must be fully type-hinted and pass static analysis.
- Good Documentation: Comments should describe what code is doing and (if particularly complex) why it is doing it.

## Dev Notes

- Identical Outcome: Ideally, there should be no performance regressions and no visual regressions (unless deliberately making visual changes).

- When modifying existing architectures, preserve the intended separation of concerns. Look at module names and existing patterns to understand the intended responsibility boundaries.

- When implementing new code or fixing bugs, if the functionality isn't already covered by unit testing, implement one or more unit tests to test it as you see fit.

- As you make changes, add docstrings and line comments to clarify to a human reader *what* the code is doing and, if needed, *why* it's doing it that way.

## Design Discussions

When discussing game design, architecture, or implementation trade-offs:
- Present the landscape of options before recommending one
- Argue from first principles, not just pattern-matching to what I seem to want
- Make clear recommendations with explicit reasoning
- Push back on assumptions that don't hold up
