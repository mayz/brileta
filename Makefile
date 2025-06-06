.PHONY: all format ruff-check lint typecheck test check clean

# Default target - run all quality checks
all: lint test
	@echo "✅ All checks passed!"

# Run all static analysis (format + ruff + types)
lint: format ruff-check typecheck

# Format code
format:
	ruff format .

# Run ruff linting
ruff-check:
	ruff check .

# Run type checking
typecheck:
	ty check

# Run tests
test:
	python -m pytest tests/

# Alias for 'all'
check: all

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +