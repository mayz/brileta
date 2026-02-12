.PHONY: all format ruff-check lint typecheck test check clean run

# Globally silence `make` output
MAKEFLAGS += --silent

NATIVE_STAMP := .venv/.native-build-stamp
NATIVE_SOURCES := setup.py pyproject.toml $(wildcard brileta/util/native/*.c brileta/util/native/*.h)

# Point git at the version-controlled hooks directory.
$(shell git config core.hooksPath .githooks)

# Default target - run all quality checks
all: native-build lint test
	@echo "✅ All checks passed!"

# Build native extension modules in editable mode.
native-build: $(NATIVE_STAMP)
	@if ! find brileta/util -maxdepth 1 -type f \( -name "_native*.so" -o -name "_native*.pyd" -o -name "_native*.dll" \) | grep -q .; then \
		uv pip install -e .; \
		mkdir -p $(dir $(NATIVE_STAMP)); \
		touch $(NATIVE_STAMP); \
	fi

$(NATIVE_STAMP): $(NATIVE_SOURCES)
	uv pip install -e .
	mkdir -p $(dir $@)
	touch $@

# Run all static analysis (format + ruff + types)
lint: format ruff-check typecheck

# Format code
format:
	uv run ruff format .

# Run ruff linting (auto-fix what it can)
ruff-check:
	uv run ruff check --fix .

# Run type checking
typecheck:
	uv run ty check --error-on-warning

# Run tests. Installs/syncs dependencies AND runs pytest in the same logical command.
# Reset stack size to the shell default. GNU make on macOS inflates it to ~64 MB,
# which causes memory pressure when pytest-xdist spawns many worker processes.
test: native-build
	ulimit -s 8176 && \
	uv sync && \
	if [ "$(shell uname)" = "Linux" ] && [ -z "$$DISPLAY" ]; then \
		if command -v xvfb-run >/dev/null 2>&1 ; then \
			xvfb-run -s "-screen 0 1024x768x24" uv run pytest tests/ ; \
		else \
			echo "Headless run detected (DISPLAY is empty). Install Xvfb and Mesa/Vulkan packages, then run again with xvfb-run -a make." ; \
			exit 1 ; \
		fi ; \
	else \
		uv run pytest tests/ ; \
	fi

# Alias for 'all'
check: all

# Clean up cache files
clean:
	rm -f $(NATIVE_STAMP)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Run the game inside the virtual environment
run:
	uv run python -m brileta
