.PHONY: help install data lint format test clean

help:  ## Show available commands
	@grep -E "^[a-zA-Z_-]+:.*?## " $(MAKEFILE_LIST) | awk "BEGIN {FS = \":.*?## \"}; {printf \"  \033[36m%-14s\033[0m %s\n\", \$$1, \$$2}"

install:  ## Create .venv and install all dependencies from uv.lock
	uv sync

data:  ## Download the BAF base dataset from Kaggle (needs a Kaggle API token)
	uv run python -m vigil.data.download

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Run the test suite
	uv run pytest -q

clean:  ## Remove caches (keeps data and .venv)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
