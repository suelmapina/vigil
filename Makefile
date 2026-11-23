.PHONY: help install data drift model decision explain fairness all lint format test mlflow clean

help:  ## Show available commands
	@grep -E "^[a-zA-Z_-]+:.*?## " $(MAKEFILE_LIST) | awk "BEGIN {FS = \":.*?## \"}; {printf \"  \033[36m%-14s\033[0m %s\n\", \$$1, \$$2}"

install:  ## Create .venv and install all dependencies from uv.lock
	uv sync

data:  ## Download the BAF base dataset from Kaggle (needs a Kaggle API token)
	uv run python -m vigil.data.download

drift:  ## Fraud rate and feature drift (PSI) per month
	uv run python -m vigil.drift

model:  ## Fit and compare fraud models on the temporal split (~4 min)
	uv run python -m vigil.model.train

decision:  ## Compare screening policies in euros on the test months
	uv run python -m vigil.decision.simulate

explain:  ## Reason codes and global feature importance (TreeSHAP)
	uv run python -m vigil.explain

fairness:  ## Burden by age group and the cost of three fairness options (~2 min)
	uv run python -m vigil.fairness

all: drift model decision explain fairness  ## Run the full pipeline

lint:  ## Lint with ruff
	uv run ruff check src tests

format:  ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Run the test suite
	uv run pytest -q

mlflow:  ## Open the MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:  ## Remove caches (keeps data and .venv)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
