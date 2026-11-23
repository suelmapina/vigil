.PHONY: help install data drift model decision explain fairness monitor all api loadtest docker demo-data demo lint format test mlflow clean

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

monitor:  ## Monthly monitoring report, retraining backtest and shadow test (~9 min)
	uv run python -m vigil.monitor

all: drift model decision explain fairness monitor  ## Run the full pipeline

api:  ## Serve the scoring API on http://localhost:8000/docs
	uv run uvicorn vigil.api.main:app --port 8000

loadtest:  ## Latency under load (start make api in another terminal first)
	uv run python -m vigil.api.loadtest --requests 2000 --concurrency 1
	uv run python -m vigil.api.loadtest --requests 3000 --concurrency 8

docker:  ## Build the API image (after make model)
	docker build -t vigil-api .

demo-data:  ## Write the small files the hosted demo needs into app/data
	uv run python -m vigil.exportdemo

demo:  ## Open the Streamlit demo on http://localhost:8501
	uv run streamlit run src/vigil/demo.py

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
