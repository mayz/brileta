.PHONY: all format ruff-check lint typecheck test check clean run

# Globally silence `make` output
MAKEFLAGS += --silent

# Default target - run all quality checks
all: lint test
	@echo "✅ All checks passed!"

# Run all static analysis (format + ruff + types)
lint: format ruff-check typecheck

# Format code
format:
	uv run ruff format .

# Run ruff linting
ruff-check:
	uv run ruff check .

# Run type checking
typecheck:
	uv run pyright

# Run tests. Installs/syncs dependencies AND runs pytest in the same logical command.
test:
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
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Run the game inside the virtual environment
run:
	uv run python -m catley
